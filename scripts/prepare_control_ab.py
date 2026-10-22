"""Prepare paired control-prefix manifests and configs; never launch training."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary
from voxft.paths import ROOT
from voxft.train.yaml_builder import steps_for_epochs

from incremental_mix import audit_mix, load_exclusions, target_text, validate_mix
from ref_ab import sha256


def prepare(source, output, template_path, eval_dir):
    if output.exists():
        raise FileExistsError(output)
    excluded, inputs = load_exclusions(eval_dir)
    train_path, val_path = source / "train.jsonl", source / "val.jsonl"
    inputs.update({str(path): sha256(path) for path in (train_path, val_path, template_path)})
    train, validation = _read_manifest(train_path), _read_manifest(val_path)
    audit = audit_mix(train, validation, excluded)
    if audit["origins_above_3x"]:
        raise ValueError("Source exceeds target exposure limit")
    dropped = {item["row"] for item in audit["violations"]}
    on = [{**row, "control": row.get("control") or ""}
          for position, row in enumerate(train, 1) if position not in dropped]
    controlled = [row for row in on if row["control"]]
    if not controlled:
        raise ValueError("No verified control examples")
    for row in controlled:
        if not (row.get("emotion_verified") is True or row.get("control_verified") is True):
            raise ValueError("Control prefix lacks verified annotation")
        if not row["text"].startswith(f"({row['control']})"):
            raise ValueError("Control metadata/text prefix mismatch")
    off = [{**row, "text": target_text(row), "control": ""} for row in on]
    for rows in (off, on):
        validate_mix(rows, validation, excluded)
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not config.get("lora") or config["lora"].get("enable_proj") is not False:
        raise ValueError("Expected existing LoRA recipe with projection disabled")
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Frozen source changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(val_path, output / "validation.jsonl")
    artifacts = {}
    for arm, rows in (("off", off), ("on", on)):
        manifest = output / f"{arm}.jsonl"
        _write_jsonl(rows, manifest)
        steps = steps_for_epochs(str(manifest), 3, config["batch_size"],
                                 config["grad_accum_steps"], 1)
        arm_config = {**config, "train_manifest": str(manifest),
                      "val_manifest": str(output / "validation.jsonl"),
                      "num_iters": steps, "max_steps": steps,
                      "save_path": str(output / f"checkpoint_{arm}"),
                      "tensorboard": str(output / f"checkpoint_{arm}" / "logs")}
        config_path = output / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(arm_config, allow_unicode=True), encoding="utf-8")
        artifacts[arm] = {"manifest_sha256": sha256(manifest),
                          "config_sha256": sha256(config_path),
                          "summary": dataset_summary(rows), "num_iters": steps}
    plan = {
        "status": "prepared_not_authorized_for_training", "training_started": False,
        "hypothesis": "Effect of verified English control prefixes on fixed joint-language data.",
        "initialization": "Same original base, fresh optimizer/scheduler; no r23 continuation.",
        "only_changed_fields": ["text_control_prefix", "control"],
        "inputs_sha256": inputs, "artifacts": artifacts,
        "validation_sha256": sha256(output / "validation.jsonl"),
        "removed_source_rows": audit["violations"], "control_rows": len(controlled),
        "training_seed": 42, "evaluation_seeds": [42, 43, 44, 45, 46],
        "pending_gates": [
            "Fresh held-out texts: >=30 independent texts per target language; freeze before training.",
            "Verify target/ref PCM, provenance and actual base weight hashes.",
            "Verify equal training RNG initialization and reject implicit resume in the runner.",
        ],
        "note": "Both arms include the same THAI-SER audio. This isolates prefixes, not source effects. "
                "Only existing verified labels are used; no ID/TL emotion labels invented. "
                "Existing reports/checkpoints/manifests/audio are unchanged. "
                "Acoustic/native-language review remains unverified.",
    }
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path,
                        default=ROOT / "configs/lora_omni5_r20_emofix_3ep.yaml")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval_cases")
    args = parser.parse_args()
    plan = prepare(args.source.resolve(), args.output.resolve(), args.template, args.eval_dir)
    print(json.dumps({key: value for key, value in plan.items()
                      if key not in {"inputs_sha256", "removed_source_rows"}}, ensure_ascii=False, indent=2))
