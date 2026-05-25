#!/usr/bin/env bash
# Run the full PAN + WildChat sweep for one encoder, including both TF-IDF
# and SBERT hardness mining. Skips any cell whose result.json already exists.
#
# Usage:
#   bash scripts/run_full_sweep_one_model.sh
#   bash scripts/run_full_sweep_one_model.sh --dry-run
#
# To switch models: edit MODEL_NAME / TRAIN_BATCH / EVAL_BATCH / GRAD_ACCUM
# at the top. The defaults below are for distilbert-base-uncased.
# For bert-base-uncased use: TRAIN_BATCH=8 EVAL_BATCH=16 GRAD_ACCUM=2.

set -euo pipefail

# ---------- per-model knobs (edit these for each encoder) ----------
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
EVAL_BATCH="${EVAL_BATCH:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# ---------- fixed across models ----------
DATASETS=(pan wildchat)
SEEDS=(13 21 42)
HARDNESS_MINER_MODEL="sentence-transformers/all-MiniLM-L6-v2"
NUM_EPOCHS=3
TRIPLETS_PER_AUTHOR=16
MAX_LENGTH=256
MAX_CHARS_PER_TEXT=2000
HARDNESS_MINING_BATCH_SIZE=64

PYTHON=.venv/bin/python
SWEEP_SCRIPT=scripts/run_minilm_contrastive_sweep.py
OUTPUT_ROOT=outputs/experiments/contrastive_profile_transformers
MODEL_CACHE=outputs/model_cache
SUMMARY_DIR=outputs/summaries/contrastive_profile_transformers

DRY_RUN_FLAG=""
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN_FLAG="--dry-run"
  echo "[orchestrator] dry-run mode: commands will be printed but not executed"
fi

# Each configuration is a single sweep call. Together they produce the five
# (train_regime, train_miner, eval_miner) cells per (dataset, seed):
#   (RO,    tfidf, tfidf)     — original baseline (likely already done)
#   (mixed, tfidf, tfidf)     — original hardness-aware (likely already done)
#   (RO,    tfidf, sbert)     — SBERT-robustness check on the RO baseline
#   (mixed, tfidf, sbert)     — cross-miner: train TF-IDF, test SBERT
#   (mixed, sbert, sbert)     — SBERT-native upper bound
#
# The sweep skips any cell whose result.json already exists, so re-running this
# script after partial completion is safe and idempotent.

run_sweep() {
  local regimes_arg="$1"      # space-separated regimes
  local train_miner="$2"
  local eval_miner="$3"
  local label="$4"

  echo
  echo "=============================================="
  echo "[orchestrator] $label"
  echo "  model=$MODEL_NAME  train_miner=$train_miner  eval_miner=$eval_miner"
  echo "  regimes=$regimes_arg  seeds=${SEEDS[*]}  datasets=${DATASETS[*]}"
  echo "=============================================="

  # shellcheck disable=SC2086
  "$PYTHON" "$SWEEP_SCRIPT" \
    --model-name "$MODEL_NAME" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --regimes $regimes_arg \
    --output-root "$OUTPUT_ROOT" \
    --cache-dir "$MODEL_CACHE" \
    --summary-dir "$SUMMARY_DIR" \
    --train-hardness-miner "$train_miner" \
    --eval-hardness-miner "$eval_miner" \
    --hardness-miner-model "$HARDNESS_MINER_MODEL" \
    --hardness-mining-batch-size "$HARDNESS_MINING_BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --triplets-per-author "$TRIPLETS_PER_AUTHOR" \
    --train-batch-size "$TRAIN_BATCH" \
    --eval-batch-size "$EVAL_BATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --max-chars-per-text "$MAX_CHARS_PER_TEXT" \
    --local-files-only \
    --no-summary \
    $DRY_RUN_FLAG
}

# 1. Original TF-IDF baseline + hardness-aware (skipped if already done).
run_sweep "random_only mixed" tfidf tfidf "Phase A: TF-IDF train, TF-IDF eval"

# 2. SBERT-eval cross-miner check on both regimes (TF-IDF-trained models).
run_sweep "random_only mixed" tfidf sbert "Phase B: TF-IDF train, SBERT eval"

# 3. SBERT-native upper bound (mixed regime only; RO doesn't use the train miner).
run_sweep "mixed" sbert sbert "Phase C: SBERT train, SBERT eval"

# Refresh the summary once at the end (cheap, deterministic).
if [ -z "$DRY_RUN_FLAG" ]; then
  echo
  echo "[orchestrator] refreshing summary"
  "$PYTHON" scripts/summarize_contrastive_results.py \
    --results-root "$OUTPUT_ROOT" \
    --output-dir "$SUMMARY_DIR"
fi

echo
echo "[orchestrator] sweep complete for $MODEL_NAME"
