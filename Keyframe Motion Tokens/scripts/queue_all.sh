#!/usr/bin/env bash
# Full training queue on one GPU, in the order results are needed. Skips finished runs and
# evaluations, so it can be stopped and relaunched. WAIT_PID: start after that process exits.
# STEPS=n before a tok call overrides the step budget for that run.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TSTEPS=${TSTEPS:-48000}
GSTEPS=${GSTEPS:-20000}
[ -n "$WAIT_PID" ] && tail --pid="$WAIT_PID" -f /dev/null
finished() { [ -f "$1/model.pt" ] && grep -q '"step": '"$2"',' "$1/log.jsonl" 2>/dev/null; }
tok() {  # name, args...
  local name=$1; shift
  local steps=${STEPS:-$TSTEPS}
  grep -q mpjpe_local_cm runs/tok/$name/eval_recon.json 2>/dev/null || rm -f runs/tok/$name/eval_recon.json
  if ! finished runs/tok/$name $steps; then
    echo "[$(date +%H:%M)] start tok $name"
    $PY scripts/10_train_tokenizer.py --steps $steps --out runs/tok/$name "$@" > runs/tok/$name.log 2>&1 || echo "[$(date +%H:%M)] FAILED tok $name"
  fi
  if [ ! -f runs/tok/$name/eval_recon.json ]; then
    $PY scripts/11_eval_tokenizer.py runs/tok/$name > runs/tok/$name.eval.log 2>&1 || echo "[$(date +%H:%M)] FAILED eval $name"
    echo "[$(date +%H:%M)] done tok $name"
  fi
}
gen() {  # name, args...
  local name=$1; shift
  if ! finished runs/gen/$name $GSTEPS; then
    echo "[$(date +%H:%M)] start gen $name"
    $PY scripts/20_train_generator.py --steps $GSTEPS --out runs/gen/$name "$@" > runs/gen/$name.log 2>&1 || echo "[$(date +%H:%M)] FAILED gen $name"
  fi
  if [ ! -f runs/gen/$name/eval_inbetween.json ]; then
    $PY scripts/21_eval_generator.py runs/gen/$name > runs/gen/$name.eval.log 2>&1 || echo "[$(date +%H:%M)] FAILED eval gen $name"
    echo "[$(date +%H:%M)] done gen $name"
  fi
}
mkdir -p runs/tok runs/gen
# 12k-step ablations, step-matched within each group (finished; re-evaluated if the metrics changed):
# keyframes coded context-free (I-frames) vs predictively (P-frames); fixed-rate trajectory
# 0 integrated velocities (HumanML3D-style, drifts), 1 absolute, 2 integrated + learned correction
STEPS=12000 tok kf_g4_12k --kind kf --groups 4
STEPS=12000 tok kf_g4p_12k --kind kf --groups 4 --predictive 1
STEPS=12000 tok fr_fsq_g4_vel_12k --kind fr --quant fsq --groups 4
STEPS=12000 tok fr_fsq_g4_abs_12k --kind fr --quant fsq --groups 4 --abs_traj 1
STEPS=12000 tok fr_fsq_g4_12k --kind fr --quant fsq --groups 4 --abs_traj 2
[ -f runs/gen/baselines.json ] || $PY scripts/21_eval_generator.py --baselines --out runs/gen/baselines.json > runs/gen/baselines.log 2>&1
# core, 48k steps (12k left the fixed-rate tokenizer far from converged): ~300 b/s tokenizers and
# the 2x2 {keyframe, fixed-rate} x {discrete, continuous} generators on the same backbone
tok fr_fsq_g4 --kind fr --quant fsq --groups 4 --abs_traj 2
# generators use per-dimension FSQ heads (--code_head digits, the default). Joint-softmax ablations,
# trained earlier with the same backbone: runs/gen/fr_fsq_g4_joint, runs/gen/kf_g4p_joint (the keyframe
# one also without decoded-key inputs; it barely learned the codes)
gen fr_fsq_g4 --tok runs/tok/fr_fsq_g4 --continuous 0
tok kf_g4p --kind kf --groups 4 --predictive 1
gen kf_g4p --tok runs/tok/kf_g4p --continuous 0
# placement control: the same keyframe tokens, keys every 4 frames instead of adaptive
gen kf_g4p_uniform4 --tok runs/tok/kf_g4p --continuous 0 --stride 4
tok kf_none --kind kf --quant none
tok fr_none --kind fr --quant none --abs_traj 2
gen kf_cont --tok runs/tok/kf_none --continuous 1
gen fr_cont --tok runs/tok/fr_none --continuous 1
# rate sweep and fixed-rate quantizer controls
tok fr_fsq_g2 --kind fr --quant fsq --groups 2 --abs_traj 2
tok kf_g2p --kind kf --groups 2 --predictive 1
tok fr_fsq_g8 --kind fr --quant fsq --groups 8 --abs_traj 2
tok kf_g8p --kind kf --groups 8 --predictive 1
tok fr_fsq_g1 --kind fr --quant fsq --groups 1 --abs_traj 2
tok fr_vq_g2 --kind fr --quant vq --groups 2 --abs_traj 2
tok fr_rvq_q2 --kind fr --quant rvq --groups 2 --abs_traj 2
echo "[$(date +%H:%M)] queue finished"
