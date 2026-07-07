#!/bin/bash
# Precompute RoboTwin text-embedding cache across multiple GPUs.
#
# Splits the full instruction set (meta/tasks.jsonl) into NUM_SHARDS shards and
# runs one precompute process per GPU. Cache writes are atomic (tmp+rename) and
# keyed by prompt hash, so shards can safely share OUTPUT_DIR. Already-cached
# entries are skipped, so the job is resumable.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-vace-a800-0408}
RECIPE=${RECIPE:-examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml}
BACKBONE=${BACKBONE:-/root/paddlejob/bosdata/guanjiazhi/pretrain_weights/hf_cache/Wan2.2-TI2V-5B}
OUTPUT_DIR=${OUTPUT_DIR:-/root/paddlejob/robotwin-textcache}
NUM_SHARDS=${NUM_SHARDS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

cd "$REPO_DIR"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
mkdir -p "$OUTPUT_DIR/logs"

read -r -a GPU_ARR <<< "$GPUS"
idx=0
for gpu in "${GPU_ARR[@]}"; do
  echo "[launch] shard $idx/$NUM_SHARDS on GPU $gpu -> $OUTPUT_DIR/logs/shard_${idx}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m starwam.tools.precompute_text_cache \
    --config "$RECIPE" \
    --pretrained-model-id "$BACKBONE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --dtype bf16 \
    --batch-size "$BATCH_SIZE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$idx" \
    > "$OUTPUT_DIR/logs/shard_${idx}.log" 2>&1 &
  idx=$((idx + 1))
done
wait
echo "ALL SHARDS DONE"
