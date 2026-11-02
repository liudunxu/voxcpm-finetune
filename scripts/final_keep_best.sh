#!/bin/bash
set -euo pipefail

ROOT=/root/autodl-tmp/r46_keepbest_final_20261102
PROJECT=/root/voxcpm-finetune
PY=$PROJECT/.venv/bin/python
BASE=/root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915
CANDIDATE=/root/autodl-tmp/r46_micro1_20261031/run/checkpoint/step_0000009
CASES=$PROJECT/eval_cases/reserved_final_20261030.jsonl
mkdir -p "$ROOT/logs"
cd "$PROJECT"
export PATH="$PROJECT/.venv/bin:/root/.local/bin:$PATH"
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VOXFT_WHISPER_MODEL_LARGE=/root/autodl-tmp/hf_home/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478
export VOXFT_SPK_EMB_MODEL=/root/autodl-tmp/hf_home/hub/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

"$PY" - <<'PY'
import hashlib, json, os, shutil, subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/r46_keepbest_final_20261102')
cases=Path('/root/voxcpm-finetune/eval_cases/reserved_final_20261030.jsonl')
candidate=Path('/root/autodl-tmp/r46_micro1_20261031/run/checkpoint/step_0000009')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(cases)=='cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c'
rows=[json.loads(line) for line in cases.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(rows)==90 and len({r['case_id'] for r in rows})==90
assert {lang:sum(r['lang']==lang for r in rows) for lang in ('th','tl','id')}=={'th':30,'tl':30,'id':30}
assert sha(candidate/'lora_weights.safetensors')=='4cd25558723cfe3db468bf45b61f2d94d50816e1b8f072dd7ccebfcd01531115'
assert sha(candidate/'lora_config.json')=='3ed83c23c2b55b2e9c4ebedb2f92bfe553712ca80aa6f033fd2b8bf71aa6922b'
assert not (root/'run.started').exists(), 'Existing final run must be inspected, not repeated'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
free=shutil.disk_usage(root).free
assert free >= int(5.3*1024**3), f'Need 5.3GiB conservative start budget, have {free}'
(root/'run.started').write_text(json.dumps({'cases_sha256':sha(cases),'outputs_per_arm':360,
    'seeds':[42,43,44,45],'candidate':str(candidate),'free_bytes':free}),encoding='utf-8')
print('FINAL_PREFLIGHT_OK',len(rows),free,flush=True)
PY

run_arm() {
  local name=$1 target=$2 shard pid failed=0
  local -a pids=()
  for shard in 0 1 2 3; do
    local directory=$ROOT/$name/shard$shard
    mkdir -p "$directory/runtime/inductor" "$directory/runtime/triton"
    VOXFT_CKPT_ROOT="$directory" TORCHINDUCTOR_CACHE_DIR="$directory/runtime/inductor" \
      TRITON_CACHE_DIR="$directory/runtime/triton" \
      "$PY" -X utf8 -u -m voxft.eval "$target" --base "$BASE" --texts-file "$CASES" \
      --seeds 42 43 44 45 --cfg-value 1.8 --inference-timesteps 20 --shard "$shard/4" \
      > "$ROOT/logs/${name}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed )); then
    echo "$name shard failed; preserve all completed outputs" >&2
    return 1
  fi
  local merged=$ROOT/$name/merged
  mkdir -p "$merged"
  local reports=("$ROOT"/$name/shard*/eval/*.json)
  VOXFT_CKPT_ROOT="$merged" "$PY" -X utf8 -u -m voxft.eval --merge "${reports[@]}" \
    > "$ROOT/logs/${name}_merge.log" 2>&1
  "$PY" - "$name" <<'PY'
import shutil,sys
from pathlib import Path
root=Path('/root/autodl-tmp/r46_keepbest_final_20261102')
reports=list((root/sys.argv[1]/'merged/eval').glob('*.json'))
assert len(reports)==1,reports
print('ARM_COMPLETE',sys.argv[1],reports[0],shutil.disk_usage(root).free,flush=True)
PY
}

run_arm base base
run_arm candidate "$CANDIDATE"
base_reports=("$ROOT"/base/merged/eval/*.json)
candidate_reports=("$ROOT"/candidate/merged/eval/*.json)
if (( ${#base_reports[@]} != 1 || ${#candidate_reports[@]} != 1 )); then
  echo 'Expected exactly one merged report per arm' >&2
  exit 1
fi
"$PY" -X utf8 -u scripts/keep_best_eval.py --base "${base_reports[0]}" \
  --candidate "${candidate_reports[0]}" --output "$ROOT/keep_best.json" \
  > "$ROOT/logs/keep_best.log" 2>&1
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/root/autodl-tmp/r46_keepbest_final_20261102')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
result=json.loads((root/'keep_best.json').read_text(encoding='utf-8'))
assert result['cases']==90 and result['selection_policy']['seed_order']==[42,43,44,45]
paths=[root/name for name in ('run.started','keep_best.json')]
paths += [Path(result['base_report']),Path(result['candidate_report'])]
completion={'status':'final_outputs_complete','cases':90,'outputs_per_arm':360,
    'seeds':[42,43,44,45],'sha256':{str(p):sha(p) for p in paths},
    'final_holdout_used':True,'production_changed':False,'native_quality':'unverified'}
(root/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL_KEEPBEST_COMPLETE',json.dumps(result['by_lang'],ensure_ascii=False),flush=True)
PY
