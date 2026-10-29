import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('audit_training_quality')


def test_padding_has_no_extra_block_at_exact_boundary(module):
    assert module.padding_seconds(51200, 2560, 16000) == 0
    assert module.padding_seconds(51201, 2560, 16000) == pytest.approx(2559 / 16000)
    assert module.padding_seconds(53759, 2560, 16000) == pytest.approx(1 / 16000)
    with pytest.raises(ValueError):
        module.padding_seconds(0, 2560, 16000)


def test_official_packer_stops_on_last_target_patch_with_and_without_ref(module):
    # Isolate native torch/einops teardown from the suite's other native runtimes on macOS.
    code = """
from audit_training_quality import packer_check
result = packer_check({'audio_vae_config': {'sample_rate': 16000, 'encoder_rates': [2, 5, 8, 8]},
                       'patch_size': 4, 'feat_dim': 64})
assert set(result) == {'plain', 'reference'}
assert all(r['stop_on_last_target_patch'] and not r['end_token_has_loss'] for r in result.values())
"""
    subprocess.run([sys.executable, '-c', code], cwd=Path(module.__file__).parent,
                   check=True, capture_output=True, text=True, timeout=30)


def test_nominal_controls_exclude_same_identity_origin_and_pcm(module):
    files = {}
    for path, speaker, origin, pcm in [('/a', 'A', 'a', 'a'), ('/ref', 'A', 'ref', 'ref'),
            ('/same', 'A', 'same', 'same'), ('/origin', 'B', 'a', 'other'),
            ('/pcm', 'C', 'x', 'ref'), ('/negative', 'D', 'n', 'n')]:
        files[path] = {'audio': path, 'speaker': speaker, 'origin_audio': origin,
                       'pcm_sha256': pcm, 'source_id': 'cv22_id'}
    vectors = {path: np.array([1., 0.]) for path in files}
    vectors['/negative'] = np.array([0., 1.])
    vectors['/ref'] = np.array([.6, .8])
    pairs = [{'audio': '/a', 'ref_audio': '/ref', 'speaker': 'A', 'source_id': 'cv22_id'}]
    rows, summary = module.summarize_pairs(pairs, files, vectors)
    assert rows[0]['cosine'] == .6
    assert rows[0]['nominal_negative_cosines'] == [0.]
    assert rows[0]['identity_verdict'] == 'unverified_score_only'
    assert summary['cv22_id']['nominal_different_speaker']['count'] == 1


def test_frozen_diagnostics_fail_on_changed_inputs(module, tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{}', encoding='utf-8')
    module.save_new(tmp_path / 'plan.json', {'inputs_sha256': {str(path): module.sha256(path)}})
    module.checked(tmp_path)
    path.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='input changed'):
        module.checked(tmp_path)


def test_missing_metric_is_not_zero_and_nonfinite_fails(module):
    assert module.stats([None]) == {'count': 0}
    assert module.stats([None, .2])['mean'] == .2
    with pytest.raises(ValueError, match='Nonfinite'):
        module.stats([float('nan')])
