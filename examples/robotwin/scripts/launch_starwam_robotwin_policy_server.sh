#!/bin/bash
# Launch StarWAM RoboTwin policy inference servers (one per GPU) in the
# Torch/StarWAM environment. Each server binds SERVER_PORT_BASE+gpu and serves
# action-chunk inference over a socket; the RoboTwin SAPIEN client connects to
# these ports (see launch_starwam_robotwin_mot_rollout.sh).
#
# Use this when SAPIEN and Torch/StarWAM cannot share one environment (the
# common case). If they CAN share one env, use policy_mode=local instead and
# skip the server entirely.
#
# All machine-specific paths come from environment variables (no hardcoding):
#   REPO=/path/to/starWAM \
#   PY=/path/to/starwam-env/bin/python \
#   CKPT=/path/to/output/starwam_robotwin_mot_wan22_5b/checkpoint-XXXX/pytorch_model \
#   BACKBONE=/path/to/Wan2.2-TI2V-5B \
#   ACTION_STATS=/path/to/output/.../action_stats.json \
#   TEXTCACHE=/path/to/output/.../eval_text_cache \
#   ACTION_INIT=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
#   SERVER_BIND=0.0.0.0 SERVER_PORT_BASE=8765 NGPU=8 \
#   bash examples/robotwin/scripts/launch_starwam_robotwin_policy_server.sh
set -u

REPO="${REPO:?set REPO=/path/to/starWAM}"
PY="${PY:?set PY=/path/to/starwam-env/bin/python}"
CKPT="${CKPT:?set CKPT=/path/to/checkpoint-XXXX/pytorch_model}"
BACKBONE="${BACKBONE:?set BACKBONE=/path/to/Wan2.2-TI2V-5B}"
ACTION_STATS="${ACTION_STATS:?set ACTION_STATS=/path/to/action_stats.json}"
STATE_STATS="${STATE_STATS:-$ACTION_STATS}"
TEXTCACHE="${TEXTCACHE:?set TEXTCACHE=/path/to/eval_text_cache}"
ACTION_INIT="${ACTION_INIT:?set ACTION_INIT=/path/to/preprocessed/starwam_action_dit_init_wan22.pt}"
RECIPE="${RECIPE:-examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml}"
SERVER_BIND="${SERVER_BIND:-0.0.0.0}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-8765}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"   # Fast-WAM RoboTwin eval = 4
NGPU="${NGPU:-8}"
LOGDIR="${LOGDIR:-$REPO/robotwin_server_logs}"
mkdir -p "$LOGDIR" "$TEXTCACHE"

OVR="backbone.pretrained_model_id=$BACKBONE data.action_stats_path=$ACTION_STATS data.state_stats_path=$STATE_STATS data.text_embedding_cache_dir=$TEXTCACHE framework.action_expert_init_from=$ACTION_INIT"

cd "$REPO" || exit 1
for (( g=0; g<NGPU; g++ )); do
  port=$(( SERVER_PORT_BASE + g ))
  echo "[server] gpu$g -> $SERVER_BIND:$port (log $LOGDIR/server_${g}.log)"
  CUDA_VISIBLE_DEVICES=$g PYTHONDONTWRITEBYTECODE=1 "$PY" -m examples.robotwin.policy_server \
    --config "$RECIPE" \
    --checkpoint "$CKPT" \
    --override $OVR \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --host "$SERVER_BIND" --port "$port" \
    > "$LOGDIR/server_${g}.log" 2>&1 &
done
echo "[server] launched $NGPU servers on ports ${SERVER_PORT_BASE}..$(( SERVER_PORT_BASE + NGPU - 1 ))"
echo "[server] ready when each log prints: 'model ready ... listening on'"
wait
