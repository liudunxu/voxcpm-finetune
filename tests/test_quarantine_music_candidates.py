import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quarantine_music_candidates')


def test_exclusion_keeps_bytes_and_catches_target_reference_origin_and_pcm(module, tmp_path):
    rows = [{'audio': '/good', 'origin_audio': '/raw/good'}, {'audio': '/bad'},
            {'audio': '/alias'}, {'audio': '/target', 'ref_audio': '/alias'},
            {'audio': '/same_origin', 'origin_audio': '/raw/bad'},
            {'audio': '/t2', 'ref_audio': '/other_ref', 'ref_origin_audio': '/raw/bad'}]
    rows = [{**r, 'duration': 4., 'lang': 'id', 'source_id': 'gigaspeech2_id', 'text': 'sample'} for r in rows]
    path = tmp_path/'input.jsonl'
    before = b''.join((json.dumps(r, ensure_ascii=False)+'\r\n').encode() for r in rows)
    path.write_bytes(before)
    audio = {r[k]: {'pcm_sha256': r[k]} for r in rows for k in ('audio', 'ref_audio') if r.get(k)}
    audio['/alias']['pcm_sha256'] = 'bad_pcm'
    flags = [{'audio': '/bad', 'pcm_sha256': 'bad_pcm', 'uses': [{'origin_audio': '/raw/bad'}]}]
    output = tmp_path/'output/train.jsonl'
    result = module.filter_manifest(path, output, flags, audio)
    assert result['excluded'] == 5 and path.read_bytes() == before
    assert output.read_bytes() == before.splitlines(keepends=True)[0]
    removed = module._read_manifest(output.with_name('train_excluded.jsonl'))
    assert all('music_score_suspect_user_excluded' in row['reasons'] for row in removed)
    with pytest.raises(FileExistsError):
        module.filter_manifest(path, output, flags, audio)


def test_finish_waits_for_complete_parent_then_preserves_all_existing_exclusions(module, monkeypatch, tmp_path):
    main, work = tmp_path/'main', tmp_path/'work'
    main.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'CLEAN', main)
    module.save_new(work/'plan.json', {'inputs_sha256': {}})
    module.save_new(main/'status.json', {'stage': 'asr'})
    with pytest.raises(RuntimeError, match='complete'):
        module.finish(work)
    assert not (work/'candidate').exists()
    rows = [{'audio': f'/{i}', 'origin_audio': f'/raw/{i}', 'lang': 'id', 'source_id': 'gigaspeech2_id',
             'text': f'A sufficiently long and distinct test sentence number {i}', 'duration': 4.,
             'speaker_verified': False} for i in range(4)]
    # Row 0 was already excluded by the main cleaner and must never return.
    outputs = {f'{source}/{split}': {} for source in ('current', 'gigaspeech2_id') for split in ('train', 'val')}
    for key in outputs:
        path = main/f'candidate/{key}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        module._write_jsonl(rows[1:3] if key.endswith('train') else rows[3:], path)
    module._write_jsonl(rows[3:], main/'protected_validation.jsonl')
    module._write_jsonl([{'audio': r['audio'], 'pcm_sha256': r['audio']} for r in rows], main/'acoustic.jsonl')
    module.save_new(work/'flagged.json', [{'audio': '/1', 'pcm_sha256': '/1', 'uses': [{'origin_audio': '/raw/1'}]}])
    module.save_new(main/'summary.json', {'status': 'complete', 'training_started': False, 'outputs': outputs,
        'outputs_sha256': {str(p.relative_to(main)): module.sha256(p) for p in (main/'candidate').rglob('*.jsonl')}})
    (main/'status.json').write_text(json.dumps({'stage': 'complete'}))
    protected = (main/'protected_validation.jsonl').read_bytes()
    module.finish(work)
    assert module._read_manifest(work/'candidate/current/train.jsonl') == rows[2:3]
    assert module._read_manifest(work/'candidate/current/val.jsonl') == rows[3:]
    assert (main/'protected_validation.jsonl').read_bytes() == protected
    summary = module.read(work/'summary.json')
    assert summary['outputs']['current/train']['excluded'] == 1 and not summary['training_started']
    assert not summary['checks']['current']['metadata']['violations']
    with pytest.raises(FileExistsError):
        module.finish(work)


def test_priority_selection_requires_all_62_frozen_files_and_matching_pcm(module):
    summary = {'review_priority': []}
    tasks, acoustic = {}, {}
    for i in range(62):
        path = f'/{i}'
        case = {'audio': path, 'file_sha256': str(i), 'pcm_sha256': str(i)}
        summary['review_priority'].append({'case': case, 'scores': {k: .9 for k in module.EVENTS}})
        tasks[path] = {'file_sha256': str(i), 'origins': [f'/raw/{i}']}
        acoustic[path] = case.copy()
    flags = module.selected_flags(summary, tasks, acoustic)
    assert len(flags) == 62 and flags[0]['uses'] == [{'origin_audio': '/raw/0'}]
    acoustic['/1']['pcm_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='evidence'):
        module.selected_flags(summary, tasks, acoustic)
    with pytest.raises(ValueError, match='62'):
        module.selected_flags({'review_priority': []}, tasks, acoustic)
