#!/bin/bash
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH PYTHONUNBUFFERED=1 TORCH_HOME=/root/autodl-tmp/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/voxcpm-finetune
P="python -X utf8 -u"
O=/root/autodl-tmp/seg_th_20261123; A="--sources fleurs_th --src $O/input.jsonl --out $O/run"
L=/root/autodl-tmp/driver_v2_logs; mkdir -p $L
step() { echo "[$(date +%F' '%T)] $*"; }
die() { step "FAILED: $*"; exit 1; }

step "r55 build+train start (bg)"
( $P scripts/r55_ratio_trim_run.py build > $L/r55_build.log 2>&1 && $P scripts/r55_ratio_trim_run.py train > $L/r55_train.log 2>&1 ) &
R55=$!

step "th qc x3"
QC=(); for k in 0 1 2; do $P scripts/fleurs_segment_20261103.py qc $k/3 $A > $L/qc_$k.log 2>&1 & QC+=($!); done
for j in "${QC[@]}"; do wait $j || die "qc shard"; done
rm -f $O/run/qc_part0of1.jsonl  # sample-run file; would double-count against qc_part*of3
$P scripts/fleurs_segment_20261103.py finalize $A > $L/finalize.log 2>&1 || die finalize
step "finalize: $(tail -n 1 $L/finalize.log)"
$P scripts/r54_th_segments_run.py music > $L/r54_music.log 2>&1 || die music
$P scripts/r54_th_segments_run.py build > $L/r54_build.log 2>&1 || die r54_build
step "r54 built: $(grep -E '"th_segment_rows"|"th_segment_hours"|"steps"' $L/r54_build.log | tr -d ' \n')"

wait $R55 || die "r55 build/train (see $L/r55_*.log)"
step "r55 trained: $(tail -n 1 $L/r55_train.log)"

step "r54 train (bg) + r55 short screen (2 gen workers)"
$P scripts/r54_th_segments_run.py train > $L/r54_train.log 2>&1 &
R54=$!
STAGE_SHORT_ONLY=1 $P scripts/candidate_stage_20261123.py r55 /root/autodl-tmp/r55_ratio_trim_20261123 2 > $L/r55_stage_short.log 2>&1 || die "r55 short stage"
step "r55 short: $(grep 'short screen' $L/r55_stage_short.log | tail -1)"
wait $R54 || die "r54 train"
step "r54 trained: $(tail -n 1 $L/r54_train.log)"

$P scripts/candidate_stage_20261123.py r54 /root/autodl-tmp/r54_th_segments_20261123 > $L/r54_stage.log 2>&1 || die "r54 stage"
step "r54 stage: $(grep 'short screen' $L/r54_stage.log | tail -1)"
$P scripts/candidate_stage_20261123.py r55 /root/autodl-tmp/r55_ratio_trim_20261123 > $L/r55_stage.log 2>&1 || die "r55 stage"
step ALL_DONE
