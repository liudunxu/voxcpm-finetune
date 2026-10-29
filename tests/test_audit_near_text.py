import importlib
from pathlib import Path
import random

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('audit_near_text')


def full_distance(a, b):
    previous = list(range(len(b)+1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(char != other)))
        previous = current
    return previous[-1]


def test_banded_distance_matches_unbounded_reference_including_unicode(module):
    rng = random.Random(42)
    samples = ['', 'ก'*45, 'abc', 'ab', '中', 'กขคง']
    samples += [''.join(rng.choices('abก中', k=rng.randrange(1, 50))) for _ in range(30)]
    for a in samples:
        for b in samples:
            distance = full_distance(a, b)
            assert module.distance_at_most_two(a, b) == (distance if distance <= 2 else None)


def test_trigram_candidates_keep_two_edits_and_repeated_grams(module):
    originals = ['กขคงจฉชซญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'*2, 'a'*60,
                 'abcdefghijklmnop'*4]
    for original in originals:
        index = module.HoldoutIndex([{'lang': 'th', 'text': original, 'location': 'heldout'}])
        variants = [original[:10]+'Z'+original[11:30]+'Y'+original[31:],
                    original[:10]+original[11:30]+original[31:],
                    original[:10]+'Z'+original[10:30]+'Y'+original[30:]]
        for text in variants:
            matches = index.matches({'lang': 'th', 'text': text})
            assert len(matches) == 1 and matches[0]['distance'] == 2


def test_short_common_phrases_and_other_languages_are_not_near_matches(module):
    index = module.HoldoutIndex([{'lang': 'id', 'text': 'Selamat pagi', 'location': 'heldout'},
                                {'lang': 'id', 'text': 'a'*50, 'location': 'long'}])
    assert index.matches({'lang': 'id', 'text': 'Selamat pagii'}) == []
    assert index.matches({'lang': 'th', 'text': 'a'*49+'b'}) == []
    assert index.matches({'lang': 'id', 'text': '(calm)SELAMAT, PAGI!', 'control': 'calm'})[0]['distance'] == 0
