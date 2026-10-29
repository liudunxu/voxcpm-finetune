from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('full_corpus_clean')


def test_asr_disagreement_is_suspect_without_language_label_veto(module):
    result = {'hyp': 'a clear transcript', 'detected_language': 'en',
              'comparisons': [{'cer': .15}],
              'segments': [{'start': 0., 'end': 3., 'avg_logprob': -.2, 'no_speech_prob': .01}]}
    assert module.asr_reasons(result) == []
    result['comparisons'][0]['cer'] = .1501
    assert module.asr_reasons(result) == ['asr_text_mismatch']
    result['segments'][0].update(avg_logprob=-1.1, no_speech_prob=.7)
    assert set(module.asr_reasons(result)) == {'asr_text_mismatch', 'asr_avg_logprob', 'asr_no_speech_prob'}
    result.update(hyp='', segments=[])
    assert {'empty_asr', 'no_asr_speech_duration'} <= set(module.asr_reasons(result))


def test_exclusion_covers_reference_origin_and_pcm_without_mutation(module):
    rows = [{'audio': '/keep.wav', 'origin_audio': '/raw/keep.wav'},
            {'audio': '/a.wav', 'ref_audio': '/alias.wav'},
            {'audio': '/b.wav', 'origin_audio': '/raw/./bad.wav'},
            {'audio': '/c.wav', 'ref_audio': '/ref.wav', 'ref_origin_audio': '/raw/bad.wav'},
            {'audio': '/d.wav'}]
    before = deepcopy(rows)
    files = {p: {'reasons': [], 'pcm_sha256': p}
             for row in rows for p in (row.get('audio'), row.get('ref_audio')) if p}
    files['/alias.wav']['pcm_sha256'] = 'bad_pcm'
    files['/d.wav']['reasons'] = ['asr_text_mismatch']
    kept, rejected = module.filter_rows(rows, {}, files, {'/raw/bad.wav'}, {'bad_pcm'}, set(), Counter())
    assert kept == [1]
    assert len(rejected) == 4
    assert rows == before


def test_protected_pcm_dedup_same_ref_and_origin_budget(module):
    rows = [{'audio': f'/{i}.wav', 'origin_audio': '/raw/shared.wav'} for i in range(7)]
    files = {r['audio']: {'reasons': [], 'pcm_sha256': str(i)} for i, r in enumerate(rows)}
    files['/1.wav']['pcm_sha256'] = '0'
    rows[2]['ref_audio'] = '/2.wav'
    kept, rejected = module.filter_rows(rows, {}, files, set(), set(), {'6'}, Counter())
    assert kept == [1, 4, 5]
    assert {reason for row in rejected for reason in row['reasons']} >= {
        'duplicate_target_pcm', 'identical_target_reference_pcm', 'validation_pcm_audio', 'origin_exposure_above_3'}


def test_audio_changes_are_fatal_but_corruption_is_quarantined(module, tmp_path):
    audio = tmp_path / 'bad.wav'
    audio.write_bytes(b'not audio')
    task = {'id': 0, 'audio': str(audio), 'file_sha256': module.sha256(audio),
            'durations': [4.], 'pre_reasons': []}
    result = module.acoustic(task, lambda a, r: [])
    assert result['reasons'] and result['pcm_sha256'] is None
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        module.acoustic(task, lambda a, r: [])


def test_policy_and_existing_runs_cannot_be_silently_changed(module, tmp_path):
    with pytest.raises(ValueError, match='policy'):
        module.check_inputs({'policy': {**module.POLICY, 'max_cer': .5}, 'inputs_sha256': {}})
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
    with pytest.raises(FileExistsError):
        module.run(tmp_path)


def test_collect_requires_every_asr_and_keeps_source_bytes_and_old_val(module, monkeypatch, tmp_path):
    work = tmp_path / 'run'
    rows = []
    for i in range(4):
        row = {'audio': str(tmp_path / f'{i}.wav'), 'origin_audio': str(tmp_path / f'raw{i}.wav'),
               'text': f'A sufficiently long and distinct test sentence number {i}',
               'source_id': 'cv22_id', 'lang': 'id', 'duration': 4., 'speaker_verified': False}
        rows.append(row)
    # Source/current are alternative candidate views, not extra training exposure.
    for source in ('cv22_id', 'current'):
        folder = work / 'manifests' / source
        folder.mkdir(parents=True)
        for split, subset in (('train', rows[:3]), ('val', rows[3:])):
            (folder / f'{split}.jsonl').write_bytes(b''.join((json.dumps(r) + '\r\n').encode() for r in subset))
    manifests = {f'{source}/{split}': {} for source in ('cv22_id', 'current') for split in ('train', 'val')}
    plan = {'policy': module.POLICY, 'inputs_sha256': {}, 'manifests': manifests, 'inherited_pcm': ['2']}
    module.save_new(work / 'plan.json', plan)
    module.save_new(work / 'metadata_issues.json', {key: [] for key in manifests})
    module._write_jsonl(rows[3:], work / 'protected_validation.jsonl')
    tasks = [{'audio': r['audio'], 'origins': [r['origin_audio']]} for r in rows]
    module._write_jsonl(tasks, work / 'tasks.jsonl')
    measurements = [{'audio': r['audio'], 'file_sha256': str(i), 'pcm_sha256': str(i), 'reasons': []}
                    for i, r in enumerate(rows)]
    module._write_jsonl(measurements, work / 'acoustic.jsonl')
    asr = [{**r, 'reasons': ['asr_text_mismatch'] if i == 1 else []} for i, r in enumerate(measurements)]
    module._write_jsonl(asr[:-1], work / 'asr.jsonl')
    monkeypatch.setattr(module, 'load_exclusions', lambda *a: (set(), {}))
    with pytest.raises(ValueError, match='Incomplete ASR'):
        module.collect(work)
    assert not (work / 'candidate').exists()
    module._write_jsonl(asr, work / 'asr.jsonl')
    before = {p: p.read_bytes() for p in (work / 'manifests').rglob('*.jsonl')}
    module.collect(work)
    assert all(p.read_bytes() == data for p, data in before.items())
    expected = (json.dumps(rows[0]) + '\r\n').encode()
    assert (work / 'candidate/current/train.jsonl').read_bytes() == expected
    assert (work / 'candidate/current/val.jsonl').read_bytes() == before[work / 'manifests/current/val.jsonl']
    result = module.read_json(work / 'summary.json')
    assert result['outputs']['current/train']['excluded'] == 2
    assert result['training_started'] is False
    assert not any(result['checks']['current']['pcm'].values())
    with pytest.raises(FileExistsError):
        module.collect(work)


def test_prepare_inventories_source_and_current_refs_without_changing_manifests(module, monkeypatch, tmp_path):
    parent, processed, raw = (tmp_path / name for name in ('parent', 'processed', 'raw'))
    parent.mkdir()
    (processed / 'cv22_id').mkdir(parents=True)
    (raw / 'cv22_id').mkdir(parents=True)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.bin').write_bytes(b'test model fingerprint; never loaded')
    rows = []
    for i in range(3):
        path = tmp_path / f'{i}.wav'
        sf.write(path, np.full(64000, .01 * (i + 1)), 16000)
        rows.append({'audio': str(path), 'origin_audio': str(raw / f'{i}.wav'),
                     'source_id': 'cv22_id', 'lang': 'id', 'duration': 4.,
                     'text': f'This is a sufficiently long source sentence number {i}',
                     'speaker_verified': False})
    module._write_jsonl([{'audio': r['origin_audio']} for r in rows], raw / 'cv22_id/manifest.jsonl')
    rows[0].update(ref_audio=rows[1]['audio'], ref_duration=4., ref_lang='id',
                   ref_origin_audio=rows[1]['origin_audio'])
    for folder in (parent, processed / 'cv22_id'):
        module._write_jsonl(rows[:2], folder / 'train.jsonl')
        module._write_jsonl(rows[2:], folder / 'val.jsonl')
    old = {'sample': {'audio': rows[1]['audio'], 'origin_audio': rows[1]['origin_audio']}}
    module._write_jsonl([old], parent / 'excluded.jsonl')
    inputs = {str(p): module.sha256(p) for p in (parent / 'train.jsonl', parent / 'val.jsonl', parent / 'excluded.jsonl')}
    cpu = tmp_path / 'cpu'
    cpu.mkdir()
    module.save_new(cpu / 'audio_audit.json', {'files': {row['audio']: module.inspect_audio(row) for row in rows}})
    inputs[str(cpu / 'audio_audit.json')] = module.sha256(cpu / 'audio_audit.json')
    old_run = tmp_path / 'old_run'
    old_run.mkdir()
    module.save_new(old_run / 'plan.json', {'additional_excluded': [{'sample': rows[0]}]})
    inputs[str(old_run / 'plan.json')] = module.sha256(old_run / 'plan.json')
    module.save_new(parent / 'plan.json', {'sources': {'cv22_id': {}}, 'asr_path': str(model), 'inputs_sha256': inputs})
    monkeypatch.setattr(module, 'PARENT', parent)
    monkeypatch.setattr(module, 'DATA_PROCESSED', processed)
    monkeypatch.setattr(module, 'DATA_RAW', raw)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'load_exclusions', lambda *args: (set(), {}))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda p: type('Usage', (), {'free': 10 * 1024**3})())
    def unused_timestamps(*args):
        raise AssertionError('Inventory must not run VAD or ASR')
    monkeypatch.setitem(sys.modules, 'faster_whisper.utils', SimpleNamespace(get_assets_path=lambda: str(tmp_path)))
    monkeypatch.setitem(sys.modules, 'faster_whisper.vad', SimpleNamespace(get_speech_timestamps=unused_timestamps))
    output = tmp_path / 'audit'
    module.prepare(output)
    tasks = module._read_manifest(output / 'tasks.jsonl')
    assert len(tasks) == 3
    assert any(use['role'] == 'ref_audio' for task in tasks for use in task['uses'])
    assert 'inherited_exclusion' in next(t for t in tasks if t['audio'] == rows[1]['audio'])['pre_reasons']
    assert 'inherited_exclusion' in next(t for t in tasks if t['audio'] == rows[0]['audio'])['pre_reasons']
    assert len(module.read_json(output / 'plan.json')['inherited_pcm']) == 2
    assert all(module.sha256(path) == digest for path, digest in inputs.items())
    module.check_inputs(module.read_json(output / 'plan.json'))
