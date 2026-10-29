import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('audit_codec_padding')


def test_native_hop_and_training_patch_have_distinct_extra_padding(module):
    assert module.pad_samples(51200, 2560) == 0
    for extra in range(4):
        samples = 51200 + (4 - extra) * 640 - 1
        assert module.pad_samples(samples, 640) == 1
        assert module.pad_samples(samples, 2560) - 1 == extra * 640
    with pytest.raises(ValueError):
        module.pad_samples(-1, 640)


def test_stratified_selection_requires_completed_pass_and_covers_all_cells(module):
    rows, acoustic, asr = [], {}, {}
    for lang in ('id', 'th', 'tl'):
        for extra in range(4):
            for i in range(9):
                path = f'/{lang}_{extra}_{i}.wav'
                rows.append({'audio': path, 'lang': lang, 'source_id': 'source'})
                acoustic[path] = {'duration': (51200 + (4-extra)*640-1)/16000,
                                  'reasons': [], 'file_sha256': str(i)}
                asr[path] = {'file_sha256': str(i), 'reasons': [], 'result': {
                    'hyp': 'test', 'comparisons': [{'cer': 0.}],
                    'segments': [{'start': 0., 'end': 3., 'avg_logprob': -.1, 'no_speech_prob': .01}]}}
                if i == 0:
                    acoustic[path]['reasons'] = ['head_silence']
    cases = module.select_cases(rows, acoustic, asr, 640, 2560)
    assert len(cases) == 96 and len({r['audio'] for r in cases}) == 96
    assert {(r['lang'], r['extra_hops']) for r in cases[:12]} == {
        (lang, extra) for lang in ('id', 'th', 'tl') for extra in range(4)}
    asr.pop(cases[0]['audio'])
    with pytest.raises(ValueError, match='Need eight'):
        module.select_cases(rows, acoustic, asr, 640, 2560)


def test_reconstruction_metrics_reject_nonfinite_and_keep_unknown_tail(module):
    measured = module.metrics(np.zeros(48000, dtype=np.float32), 48000)
    assert measured['tail_silence_sec'] is None and measured['duration'] == 1
    with pytest.raises(ValueError):
        module.metrics(np.array([float('nan')]), 16000)


def test_summary_uses_matched_deltas_and_does_not_turn_unknown_into_zero(module):
    names = ('original', 'hop_input', 'patch_input', 'hop_codec', 'patch_codec')
    items = [{'case': {'lang': lang}, 'measurements': {name: {'tail_silence_sec': value}
               for name, value in zip(names, (.1, .12, .24, .08, .20))}} for lang in ('id', 'th', 'tl')]
    items.append({'case': {'lang': 'id'}, 'measurements': {name: {'tail_silence_sec': None} for name in names}})
    summary = module.summarize(items)
    delta = summary['id']['paired_tail_delta']['patch_codec_minus_hop_codec']
    assert delta['count'] == 1 and delta['mean'] == pytest.approx(.12)
