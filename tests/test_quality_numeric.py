import importlib
from pathlib import Path

import pytest


@pytest.fixture
def check(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quality_numeric').numeric_equivalence


@pytest.mark.parametrize('lang,reference,hypothesis',[
    ('id','Album ini tetap selama empat minggu.','Album ini tetap selama 4 minggu.'),
    ('id','standar lima puluh lima gram','standar 55 gram'),
    ('en','About three thousand years later','About 3000 years later'),
    ('th','ยัดห้านัดแล้วยิงทีละนัด','ยัด 5 นัดแล้วยิงทีละนัด'),
    ('th','หนึ่ง. ท่านอนหงาย','1. ท่านอนหงาย'),
    ('tl','tumatagal ng 17 araw','tumatagal ng labing pitong araw'),
    ('zh','公元前 5000 年','公元前五千年'),
    ('zh','大约 24,000 块陨石','大约两万四千块陨石'),
    ('zh','周一晚 10：08，发生地震','周一晚十点零八分，发生地震'),
    ('zh','周一晚 10：08，发生地震','周一晚10点08分，发生地震'),
    ('zh','周一晚10：08发生地震','周一晚十点零八分发生地震'),
    ('en','About 3,000 years later','About three thousand years later'),
    ('id','ada 1.000 orang','ada seribu orang'),
])
def test_equivalent_cardinal_spelling_is_not_a_bad_label(check,lang,reference,hypothesis):
    assert check(reference,hypothesis,lang)['numeric_only_difference']
    assert check(hypothesis,reference,lang)['numeric_only_difference']


@pytest.mark.parametrize('lang,reference,hypothesis',[
    ('id','empat minggu','5 minggu'),
    ('id','empat minggu sebelum ini','4 minggu'),
    ('id','dua atom hidrogen','2 atom nitrogen'),
    ('en','one person','21 people'),
    ('en','1.23 grams','12.3 grams'),
    ('en','-4 degrees','four degrees'),
    ('en','0012','twelve'),
    ('en','R3','R three'),
    ('th','ห้าม','5 ม'),
    ('th','ห้าสิบคน','5 คน'),
    ('tl','17 araw','labing walong araw'),
    ('zh','四川','4川'),
    ('zh','周一晚10：08发生地震','周一晚十点零九分发生地震'),
    ('en','12,000','twelve'),
    ('id','1,000','seribu'),
])
def test_values_missing_words_identifiers_and_thai_negation_are_not_ignored(check,lang,reference,hypothesis):
    assert not check(reference,hypothesis,lang)['numeric_only_difference']
