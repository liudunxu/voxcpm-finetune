"""Remote-only: reserve official FLEURS test texts absent from known local training/evaluation pools."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import random

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.data.registry import TARGET_LANGS, get_source
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT, env

from quality_probe import write_json
from ref_ab import sha256

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def parse_tsv(contents):
    rows = []
    for position, fields in enumerate(csv.reader(io.StringIO(contents), delimiter="\t",
                                                quoting=csv.QUOTE_NONE), 1):
        if len(fields) != 7 or not fields[0].isdigit() or not fields[5].isdigit():
            raise ValueError(f"Unexpected FLEURS TSV schema at row {position}")
        if not fields[2].strip() or evaluation._norm(fields[2]) != evaluation._norm(fields[3]):
            raise ValueError(f"Raw/normalized transcript mismatch at row {position}")
        rows.append({"source_sentence_id": fields[0], "source_audio_filename": fields[1],
                     "text": fields[2]})
    return rows


def select_texts(rows, excluded, lang):
    unique = {}
    counts = Counter()
    for row in rows:
        text = row["text"]
        key = evaluation._norm(text)
        reason = ("existing_text" if key in excluded else "numeric" if evaluation._is_numeric(row)
                  else "length" if not 10 <= len(text) <= 300 else
                  "duplicate_text" if key in unique else "")
        if reason:
            counts[reason] += 1
        else:
            unique[key] = row
    pool = sorted(unique.values(), key=lambda row: (int(row["source_sentence_id"]), row["text"]))
    selected = random.Random(f"20260917:{lang}").sample(pool, min(30, len(pool)))
    return selected, {"input_rows": len(rows), "eligible_unique_texts": len(pool),
                      "selected": len(selected), "shortfall": max(0, 30 - len(selected)),
                      "excluded": dict(counts)}


def prepare(output, experiment):
    from huggingface_hub import hf_hub_download
    import yaml

    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(set(DATA_PROCESSED.glob("*/train.jsonl")) |
                   set(DATA_PROCESSED.glob("*/val.jsonl")) |
                   set(DATA_RAW.glob("*/manifest.jsonl")) |
                   set(ROOT.glob("eval_cases/*.jsonl")) |
                   set((CHECKPOINT_DIR.parent / "training_pool_20260917").glob("*/targets.jsonl")) |
                   {experiment / "cases.jsonl"})
    excluded, inputs = set(), {}
    for path in paths:
        for row in _read_manifest(path):
            if row.get("text"):
                excluded.add(evaluation._norm(row["text"]))
        inputs[str(path)] = sha256(path)
    for path in sorted(evaluation.EVAL_DIR.glob("*.json")):
        for row in json.loads(path.read_text()).get("items", []):
            if row.get("text"):
                excluded.add(evaluation._norm(row["text"]))
        inputs[str(path)] = sha256(path)
    references = {}
    for row in _read_manifest(experiment / "cases.jsonl"):
        if row.get("ref_audio") and row.get("ref_lang"):
            references.setdefault(row["ref_lang"], set()).add(row["ref_audio"])
    refs = {}
    for lang in ("zh", "en", "tl"):
        if not references.get(lang):
            raise ValueError(f"Missing existing references for {lang}")
        refs[lang] = sorted(references[lang])
        for reference in refs[lang]:
            if not Path(reference).is_file():
                raise FileNotFoundError(reference)
            inputs[reference] = sha256(reference)
    download = dict(repo_id="google/fleurs", repo_type="dataset", revision=REVISION,
                    token=False, endpoint=env("HF_ENDPOINT", "https://huggingface.co"),
                    cache_dir=output / "cache", local_dir=output / "sources")
    card_path = Path(hf_hub_download(filename="README.md", **download))
    header = card_path.read_text().split("---", 2)
    if len(header) != 3 or header[0].strip():
        raise ValueError("Missing dataset card frontmatter")
    license_value = yaml.safe_load(header[1]).get("license")
    if license_value not in ("cc-by-4.0", ["cc-by-4.0"]):
        raise ValueError(f"Unexpected source license: {license_value}")
    inputs[str(card_path)] = sha256(card_path)
    cases, summary = [], {}
    for lang in TARGET_LANGS:
        config = get_source(f"fleurs_{lang}").config
        source = Path(hf_hub_download(filename=f"data/{config}/test.tsv", **download))
        inputs[str(source)] = sha256(source)
        rows, summary[lang] = select_texts(parse_tsv(source.read_text()), excluded, lang)
        for position, row in enumerate(rows):
            ref_lang = ("zh", "en", "tl")[position % 3]
            reference = refs[ref_lang][(position // 3) % len(refs[ref_lang])]
            cases.append({
                **row, "case_id": f"fleurs_test_{lang}_{row['source_sentence_id']}",
                "lang": lang, "ref_audio": reference, "ref_lang": ref_lang,
                "numeric": False, "cohort": "fleurs_test_unseen_local_text",
                "source_dataset": "google/fleurs", "source_revision": REVISION,
                "source_config": config, "source_split": "test", "source_license": "CC-BY-4.0",
                "note": "Official raw transcript, absent from audited local text pools. "
                        "Pretraining exposure/speaker novelty/native quality unverified. "
                        "Read-speech text diagnostic, not dialogue/emotion validation.",
            })
            excluded.add(evaluation._norm(row["text"]))
        print(f"{lang}: {summary[lang]}", flush=True)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate source sentence IDs in selected cases")
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("Audited source changed during preparation")
    _write_jsonl(cases, output / "cases.jsonl")
    write_json(output / "plan.json", {
        "source_dataset": "google/fleurs", "source_revision": REVISION,
        "source_license": "CC-BY-4.0", "by_lang": summary, "inputs_sha256": inputs,
        "reference_pool": refs,
        "cases_sha256": sha256(output / "cases.jsonl"),
        "evaluation_seeds": [42, 43, 44, 45, 49], "cfg_value": 1.8,
        "inference_timesteps": 20, "retry_badcase": False, "evaluated": False,
        "note": "Text-only retrieval; no source audio downloaded. Exact normalized text exclusion "
                "against available raw/processed/train/val/candidate/evaluation manifests/reports. "
                "Not proof of pretraining novelty, semantic non-overlap, independent speakers, "
                "or source-audio correctness. Reference files reused; no new identity claims. "
                "Existing r10 cases/training inputs and production unchanged. "
                "Reserve these texts from future training; inference has NOT been run.",
    })
    (output / "done").write_text("Text holdout prepared; no inference started.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR.parent / "holdout_texts_20260917")
    parser.add_argument("--experiment", type=Path, default=CHECKPOINT_DIR.parent / "ref_ab_20260917")
    args = parser.parse_args()
    prepare(args.output, args.experiment)
