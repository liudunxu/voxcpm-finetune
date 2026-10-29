import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('audit_music')


@pytest.mark.parametrize('samples', [48000, 160000, 160001, 240001, 480000])
def test_windows_cover_tail_without_gaps_or_duplicate_starts(module, samples):
    result = module.windows(samples)
    assert result[0][0] == 0 and result[-1][1] == samples
    assert len({a for a, _ in result}) == len(result)
    assert all(0 <= a < b <= samples and b-a <= 160000 for a, b in result)
    assert all(right[0] <= left[1] for left, right in zip(result, result[1:]))
    assert len(result) <= 5


def test_multilabel_scores_preserve_speech_and_late_music(module):
    speech = dict(zip(module.LABELS, (.99, .01, .01, .01, .01)))
    late = {**speech, 'Music': .92}
    combined = module.aggregate([speech, late])
    assert combined['Speech'] == .99 and combined['Music'] == .92
    assert module.aggregate(iter([speech, late])) == combined
    for value in (-.01, 1.1, float('nan')):
        with pytest.raises(ValueError):
            module.aggregate([{**speech, 'Music': value}])
    with pytest.raises(ValueError):
        module.aggregate([])


def test_selection_deduplicates_roles_skips_existing_flags_and_interleaves_sources(module):
    rows = [{'audio': f'/{lang}{i}', 'ref_audio': f'/{lang}ref', 'lang': lang, 'source_id': lang}
            for lang in ('th', 'id', 'tl') for i in range(3)]
    acoustic = {path: {'reasons': [], 'file_sha256': path, 'pcm_sha256': path, 'duration': 3.5}
                for row in rows for path in (row['audio'], row['ref_audio'])}
    acoustic['/id0']['reasons'] = ['head_silence']
    cases, skipped = module.select_files(rows, acoustic)
    assert skipped == {'/id0': ['head_silence']}
    assert len(cases) == 11 and len({c['audio'] for c in cases}) == 11
    assert cases[0]['lang'] == 'id' and {c['lang'] for c in cases[:3]} == {'id', 'th', 'tl'}
    assert len(next(c for c in cases if c['audio'] == '/idref')['uses']) == 3
    with pytest.raises(ValueError, match='Conflicting audio provenance'):
        module.select_files([rows[0], {**rows[0], 'lang': 'en'}], acoustic)


def test_summary_keeps_review_priorities_distinct_from_exclusions(module):
    scores = dict(zip(module.LABELS, (.99, .85, .1, .1, .1)))
    summary = module.summarize([{'case': {'lang': 'id', 'audio': '/a'}, 'scores': scores}], .8)
    assert summary['review_priority_files'] == 1
    assert summary['automatic_exclusions'] == 0
    assert summary['training_started'] is False
