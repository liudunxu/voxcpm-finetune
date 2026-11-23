"""Build tl_tail_cases.jsonl: TL sentence-final proper nouns + Taglish (development cases, 2026-11-23).

Why: UAT 128/239 showed r50-era output repeating a sentence-final name ("...Zoe Sullivan? Sullivan, Sullivan...")
and Taglish words ("CEO" -> "Siyo"); neither form is in any eval set, so the risk is unmeasured.
Texts written by the assistant, NOT native-verified (development only, never a final holdout). Production TL text
goes through DIS contraction expansion ('yung -> iyong), so the UAT line is given in expanded form.
Same refs / cfg / adaptive steps as prod_review short_cases (3 ref arms x 5 seeds).

  python scripts/build_tl_tail_cases.py <short_cases.jsonl> <out tl_tail_cases.jsonl>
"""
import json
import sys
from pathlib import Path

NAME_FINAL = [
    "Ikaw ba talaga iyong dating aroganteng Zoe Sullivan?",  # UAT 128/239, contraction-expanded
    "Hindi kita mapapatawad, Lin Feng!",
    "Salamat po, Ginoong Chen.",
    "Nasaan na si Mark Johnson?",
    "Huwag kang mag-alala, Xiao Mei.",
    "Tumawag ulit si Attorney Reyes.",
    "Ito na ba ang anak ni Madam Huang?",
    "Magkita tayo bukas sa Makati.",
    "Pinsan ko si Daniel Cruz.",
    "Akala ko patay ka na, Wei Chen!",
    "Kasal na pala si Sofia.",
    "Siya ang bagong CEO ng Lim Group.",
]
TAGLISH = [
    "Sorry, late na ako sa meeting.",
    "Okay lang, ako na ang bahala sa project.",
    "Grabe, ang ganda ng presentation mo!",
    "Na-cancel ang flight natin bukas.",
    "I-check mo muna ang email mo.",
    "Ang CEO mismo ang nag-approve nito.",
    "Teka, nasaan ang phone ko?",
    "Busy ako ngayon, mamaya na lang.",
]


def build(short_cases, out):
    arms = {}
    for line in Path(short_cases).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        arms.setdefault(r['ref_arm'], {k: r[k] for k in ('ref_audio', 'ref_arm', 'ref_lang', 'ref_sec',
                                                          'cfg_value', 'inference_timesteps')})
    rows = []
    for kind, texts in (('name_final', NAME_FINAL), ('taglish', TAGLISH)):
        for i, text in enumerate(texts):
            for arm, ref in sorted(arms.items()):
                sid = f'tltail_{kind}_{i:02d}'
                rows.append({'case_id': f'{sid}_{arm}', 'sentence_id': sid, 'lang': 'tl', 'text': text, **ref,
                             'control': '', 'numeric': False, 'kind': kind, 'cohort': 'tl_tail_20261123',
                             'evaluation_use': 'development', 'note': 'assistant-written, not native-verified'})
    Path(out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return len(rows)


if __name__ == '__main__':
    print(build(sys.argv[1], sys.argv[2]), 'cases')
