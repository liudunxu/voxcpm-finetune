"""TensorBoard 与既有训练产物 → SwanLab 旁路桥接。

官方训练脚本只写 tensorboardX 日志；此模块周期读取 event 文件，
并转发派生训练指标、数据覆盖和已完成阶段门，不侵入 submodule 代码。
"""
from __future__ import annotations

import json
import math
import statistics
import threading
import time
from pathlib import Path

from ..paths import CONFIG_DIR, env

_METRIC_SECTIONS = {
    'train/*': '01 Training',
    'val/*': '02 Validation (random subset)',
    'data/*': '03 Data coverage',
    'gate/run/*': '04 Gate status',
    'gate/content/*': '05 Content risk',
    'gate/acoustic/*': '06 Acoustic and timbre',
    'gate/lang/*': '07 Gate by language',
    'gate/seed/*': '08 Gate by seed',
}


def _ensure_swanlab(run_name: str):
    if not env("SWANLAB_API_KEY"):
        return None
    import swanlab
    try:
        swanlab.get_run()
        return swanlab
    except RuntimeError:
        swanlab.init(project=env("SWANLAB_PROJ_NAME", "voxcpm-finetune"), name=run_name)
    for pattern, section in _METRIC_SECTIONS.items():
        swanlab.define_metric(pattern, section_name=section)
    return swanlab


def _put(metrics: dict, key: str, value) -> None:
    if value is None:
        return
    value = float(value)
    if math.isfinite(value):
        metrics[key] = value


def _run_context(tb_dir, run_name: str):
    work = Path(tb_dir).parent.parent
    config_path = work / 'train.yaml'
    if not config_path.exists():
        config_path = CONFIG_DIR / f'{run_name}.yaml'
    if not config_path.exists():
        return work, {}
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Invalid training config: {config_path}')
    return work, config


def _add_training_metrics(pending: dict[int, dict], config: dict) -> None:
    weights = config.get('lambdas') or {}
    max_norm = float(config.get('max_grad_norm', 0) or 0)
    for values in pending.values():
        losses = [key for key in values if key.startswith('train/loss/')
                  and key not in {'train/loss/total', 'train/loss/weighted_total'}]
        if losses and 'train/loss/weighted_total' not in values:
            _put(values, 'train/loss/weighted_total', sum(
                values[key] * float(weights.get(key.removeprefix('train/'), 1.0)) for key in losses))
        norm = values.get('train/grad_norm')
        if norm is not None and max_norm > 0:
            _put(values, 'train/grad_clip_ratio', 1.0 if norm <= 0 else min(1.0, max_norm / norm))
            _put(values, 'train/grad_clipped', norm > max_norm)


def _data_metrics(work: Path, config: dict) -> dict:
    manifest = config.get('train_manifest')
    if not manifest or not Path(manifest).is_file():
        return {}
    with Path(manifest).open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return {}
    mix = {}
    for candidate in (work / 'mix.json', Path(manifest).parent / 'mix.json'):
        if candidate.is_file():
            mix = json.loads(candidate.read_text(encoding='utf-8'))
            break
    metrics = {}

    def summarize(prefix, group):
        durations = [float(row['duration']) for row in group
                     if isinstance(row.get('duration'), (int, float))
                     and math.isfinite(row['duration']) and row['duration'] >= 0]
        _put(metrics, f'{prefix}/rows', len(group))
        _put(metrics, f'{prefix}/hours', sum(durations) / 3600)
        _put(metrics, f'{prefix}/duration_p50_sec', statistics.median(durations) if durations else None)
        _put(metrics, f'{prefix}/ref_rows', sum(bool(row.get('ref_audio')) for row in group))
        _put(metrics, f'{prefix}/ref_rate', sum(bool(row.get('ref_audio')) for row in group) / len(group))
        short = sum(3 <= duration <= 8 for duration in durations)
        _put(metrics, f'{prefix}/short_3_8s_rows', short)
        _put(metrics, f'{prefix}/short_3_8s_rate', short / len(group))

    summarize('data/train', rows)
    shares = mix.get('language_shares') or {}
    total_seconds = sum(float(row['duration']) for row in rows
                        if isinstance(row.get('duration'), (int, float))
                        and math.isfinite(row['duration']) and row['duration'] >= 0)
    for lang in sorted({row.get('lang') or 'unknown' for row in rows}):
        group = [row for row in rows if (row.get('lang') or 'unknown') == lang]
        prefix = f'data/lang/{lang}'
        summarize(prefix, group)
        share = shares.get(lang) or {}
        actual = share.get('actual')
        if actual is None and total_seconds:
            actual = sum(float(row['duration']) for row in group
                         if isinstance(row.get('duration'), (int, float))
                         and math.isfinite(row['duration']) and row['duration'] >= 0) / total_seconds
        _put(metrics, f'{prefix}/actual_share', actual)
        _put(metrics, f'{prefix}/requested_share', share.get('requested'))
    train = mix.get('train') or {}
    if train.get('with_control') is not None:
        _put(metrics, 'data/train/control_rate', train['with_control'] / len(rows))
    _put(metrics, 'data/train/max_exposure', train.get('max_exposure'))
    return metrics


def _gate_metrics(result: dict, checkpoint: dict) -> dict:
    from ..eval import _agg, _norm

    rows, base = result['rows'], result['matched_base']
    if not rows or len(rows) != len(base):
        raise ValueError('Gate rows and matched_base must be non-empty and paired')
    pairs = list(zip(rows, base, strict=True))
    for row, old in pairs:
        if any(row.get(key) != old.get(key) for key in ('case_id', 'seed', 'lang', 'text')):
            raise ValueError('Gate baseline conditions differ')
    metrics = {}
    content = ('mean_cer', 'mean_cer_non_numeric', 'mean_wer',
               'over_read_rate', 'suspected_truncation_rate')
    acoustic = ('mean_audio_sec_non_numeric', 'mean_speech_ratio', 'mean_tail_silence',
                'p90_tail_silence', 'mean_speaker_sim')
    languages = [None] + sorted({row['lang'] for row in rows})
    for lang in languages:
        selected = pairs if lang is None else [pair for pair in pairs if pair[0]['lang'] == lang]
        candidate = _agg([row for row, _ in selected])
        baseline = _agg([old for _, old in selected])
        for name in (*content, *acoustic):
            if lang is None:
                family = 'content' if name in content else 'acoustic'
                prefix = f'gate/{family}'
            else:
                prefix = f'gate/lang/{lang}'
            _put(metrics, f'{prefix}/candidate/{name}', candidate.get(name))
            _put(metrics, f'{prefix}/base/{name}', baseline.get(name))
            if candidate.get(name) is not None and baseline.get(name) is not None:
                _put(metrics, f'{prefix}/delta/{name}', candidate[name] - baseline[name])
        if lang is not None:
            _put(metrics, f'gate/lang/{lang}/cases', candidate['cases'])

    def severe(pair):
        row, old = pair
        return row['cer'] > .20 and row['cer'] - old['cer'] > .10

    def new_overread(pair):
        return pair[0]['over_read'] and not pair[1]['over_read']

    def new_truncation(pair):
        return pair[0]['suspected_truncation'] and not pair[1]['suspected_truncation']

    def extreme_acoustic(pair):
        row, old = pair
        return (not row['numeric'] and old.get('audio_sec') and row.get('audio_sec') is not None
                and row['audio_sec'] > 2 * old['audio_sec']
                and (old.get('speech_ratio') or 0) - (row.get('speech_ratio') or 0) > .10)

    _put(metrics, 'gate/run/completed', result['completed'])
    _put(metrics, 'gate/run/skipped', result['skipped'])
    _put(metrics, 'gate/run/rejected', result['decision'] == 'rejected')
    _put(metrics, 'gate/run/optimizer_updates', checkpoint['optimizer_updates'])
    _put(metrics, 'gate/run/max_len_ratio', max(row.get('len_ratio') or 0 for row in rows))
    _put(metrics, 'gate/run/large_paired_cer_regressions', sum(map(severe, pairs)))
    _put(metrics, 'gate/run/new_overread', sum(map(new_overread, pairs)))
    _put(metrics, 'gate/run/new_suspected_truncation', sum(map(new_truncation, pairs)))
    _put(metrics, 'gate/run/new_empty_asr', sum(
        not _norm(row.get('hyp') or '') and bool(_norm(old.get('hyp') or '')) for row, old in pairs))
    _put(metrics, 'gate/run/near_silence', sum(
        row.get('rms_dbfs', 0) < -60 and (row.get('speech_ratio') or 0) < .05 for row in rows))
    _put(metrics, 'gate/run/paired_extreme_acoustics', sum(map(extreme_acoustic, pairs)))
    _put(metrics, 'gate/run/acoustic_failure_count', sum(
        len(values) for values in (result.get('acoustic_failures') or {}).values()))
    for seed in sorted({row['seed'] for row in rows}):
        selected = [pair for pair in pairs if pair[0]['seed'] == seed]
        prefix = f'gate/seed/s{seed}'
        _put(metrics, f'{prefix}/cases', len(selected))
        _put(metrics, f'{prefix}/max_len_ratio', max(row.get('len_ratio') or 0 for row, _ in selected))
        _put(metrics, f'{prefix}/mean_cer_delta', statistics.fmean(
            row['cer'] - old['cer'] for row, old in selected))
        _put(metrics, f'{prefix}/large_paired_cer_regressions', sum(map(severe, selected)))
        _put(metrics, f'{prefix}/new_overread', sum(map(new_overread, selected)))
        _put(metrics, f'{prefix}/new_suspected_truncation', sum(map(new_truncation, selected)))
        _put(metrics, f'{prefix}/paired_extreme_acoustics', sum(map(extreme_acoustic, selected)))
    return metrics


def _sync_backend(reader, tb_dir, backend, state: dict, run_name: str) -> int:
    n = 0
    max_step = state.get('_max_step', -1)
    work, config = _run_context(tb_dir, run_name)
    pending: dict[int, dict[str, float]] = {}
    marks: dict[int, list[str]] = {}

    def queue(step, values, mark):
        if not values:
            return
        if step < max_step:
            raise RuntimeError(f'Late monitoring artifact at step {step} after {max_step}; use a fresh backfill')
        pending.setdefault(step, {}).update(values)
        marks.setdefault(step, []).append(mark)

    if not state.get('_data'):
        queue(max(0, max_step), _data_metrics(work, config), '_data')
    gate_paths = sorted(Path(work).glob('gate_*/result.json'),
                        key=lambda path: int(path.parent.name.removeprefix('gate_')))
    for path in gate_paths:
        key = f"gate:{path.parent.name}"
        if state.get(key):
            continue
        step = int(path.parent.name.removeprefix('gate_'))
        result = json.loads(path.read_text(encoding='utf-8'))
        checkpoint = json.loads((path.parent / 'checkpoint.json').read_text(encoding='utf-8'))
        queue(step, _gate_metrics(result, checkpoint), key)

    scalars = reader.scalars
    for _, row in scalars.iterrows():
        tag, step, value = row['tag'], int(row['step']), float(row['value'])
        if step <= state.get(tag, -1):
            continue
        if step <= max_step:
            state[tag] = step
            continue
        pending.setdefault(step, {})[tag] = value
    _add_training_metrics(pending, config)
    audio_pending: dict[int, dict] = {}
    audio_marks: dict[int, list[str]] = {}
    import io
    import soundfile as sf
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    for path in sorted(Path(tb_dir).rglob('events.out.tfevents.*')):
        for event in EventFileLoader(str(path)).Load():
            step = int(event.step)
            for value in event.summary.value:
                if value.WhichOneof('value') != 'tensor' or not value.tensor.string_val:
                    continue
                raw = value.tensor.string_val[0]
                if not raw.startswith(b'RIFF'):
                    continue
                key = f'audio:{value.tag}:{step}'
                if state.get(key) or step < max_step:
                    continue
                arr, sample_rate = sf.read(io.BytesIO(raw), dtype='float32')
                audio_pending.setdefault(step, {})[value.tag] = backend.Audio(arr, sample_rate=sample_rate)
                audio_marks.setdefault(step, []).append(key)

    for step in sorted(set(pending) | set(audio_pending)):
        if pending.get(step):
            backend.log(pending[step], step=step)
            for tag in pending[step]:
                state[tag] = step
            for key in marks.get(step, []):
                state[key] = True
            n += len(pending[step])
        if audio_pending.get(step):
            backend.log(audio_pending[step], step=step)
            for key in audio_marks[step]:
                state[key] = True
        max_step = max(max_step, step)
        state['_max_step'] = max_step
    return n


def sync_once(tb_dir: str | Path, run_name: str, state: dict) -> int:
    """增量同步训练、数据与阶段门到 SwanLab，失败保留下次重试进度。"""
    state['_backend_errors'] = {}
    try:
        backend = _ensure_swanlab(run_name)
        if backend is None:
            return 0
        from tbparse import SummaryReader
        reader = SummaryReader(str(tb_dir))
        return _sync_backend(reader, tb_dir, backend, state, run_name)
    except Exception as error:
        state['_backend_errors'] = {'swanlab': str(error)}
        raise RuntimeError(f'SwanLab synchronization failed: {error}') from error


def start_bridge(tb_dir: str | Path, run_name: str, interval: float = 30.0,
                 stop_event: threading.Event | None = None, progress=None
                 ) -> threading.Thread | None:
    """后台线程周期同步，直到 stop_event 被置位或 tb_dir 消失。"""
    if not env("SWANLAB_API_KEY"):
        if progress:
            progress("[monitor] 未配置 SWANLAB_API_KEY，跳过监控桥接")
        return None
    stop_event = stop_event or threading.Event()

    def loop():
        state: dict = {}
        while not stop_event.is_set():
            if Path(tb_dir).exists():
                try:
                    sync_once(tb_dir, run_name, state)
                    if progress and state.get('_backend_errors'):
                        progress(f"[monitor] SwanLab 异常: {state['_backend_errors']}")
                except Exception as exc:
                    if progress:
                        progress(f"[monitor] 桥接异常: {exc}")
            stop_event.wait(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python -m voxft.train.tb_wandb_bridge <tb_dir> <run_name>")
        raise SystemExit(1)
    state: dict = {}
    while True:
        n = sync_once(sys.argv[1], sys.argv[2], state)
        if state.get('_backend_errors'):
            print(f"后端异常: {state['_backend_errors']}", file=sys.stderr, flush=True)
        if n:
            print(f"已转发 {n} 条指标")
        time.sleep(30)
