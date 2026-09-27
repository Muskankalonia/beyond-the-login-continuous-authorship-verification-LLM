# Beyond the Login: Continuous Behavioral Authorship Verification for LLM Agent Sessions

Code and benchmark-construction scripts for *Beyond the Login: Continuous Behavioral
Authorship Verification for LLM Agent Sessions* (Muskan Sharma, Seba Susan).

The paper studies whether a text encoder can serve as an agent-layer identity check:
given an enrolled **profile** of a user's past messages and an incoming **query** message,
decide whether both were written by the same person. We show that verifiers trained and
evaluated against *random* imposters can look strong (ROC-AUC > 0.95), while relying
heavily on topic and vocabulary shortcuts. Performance drops sharply against
**topic-similar** imposters, and a 50/30/20 mix of random, semi-hard, and hard negatives
(hardness-aware training) closes much of that gap. This repository reproduces the
benchmark construction, training, and evaluation pipeline end to end.

## Data availability

The benchmark is built from two **public** datasets:

- **WildChat-1M** — the AllenAI release (primary dataset).
- **PAN-20** authorship-verification training data (secondary dataset).

We do **not** redistribute raw user text, because both corpora contain user-sensitive
content and are covered by their own licenses. Instead, the scripts reconstruct the exact
author-disjoint splits **deterministically from the released seeds (13, 21, 42)**, so
anyone with access to the original datasets can regenerate the profiles, splits, and
hardness indexes used in the paper. The model is never given metadata such as `hashed_ip`;
it only sees raw user text.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,data]'
```

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and edit the paths for your
machine. All experiments in the paper run on a single 8 GB GPU (RTX 4060 Ti); FP16 mixed
precision is used to fit BERT and ELECTRA.

## Reproducing the benchmark

### 1. Download the datasets

```bash
.venv/bin/python scripts/download_dataset.py --dataset all
```

### 2. Build author-disjoint profile–query splits

Both builders use the paper settings by default: seeds 13/21/42, a 60/20/20
author-disjoint train/validation/test split stratified by text count, a 70/30
profile/query split within each author, and a minimum of 10 texts per identity.

```bash
# PAN-20: regroup pairwise data by author into per-author profiles
.venv/bin/python scripts/build_pan_profiles.py

# WildChat-1M: English users only, hashed_ip as the identity proxy
.venv/bin/python scripts/build_wildchat_profiles.py
```

Splits are written to `data/processed/{pan_profiles,wildchat_profiles}/`, with per-seed
summaries in `outputs/profiles/`.

### 3. Training-free baselines

TF-IDF cosine, frozen SBERT, and frozen transformer encoders:

```bash
.venv/bin/python scripts/evaluate_profile_baseline_sweep.py
```

Results are written to `outputs/summaries/profile_baselines/`.

### 4. Contrastive bi-encoders (random-only vs. hardness-aware)

The orchestrator runs the full PAN + WildChat sweep for one encoder across all three
seeds and produces every (train regime, train miner, eval miner) cell used in the paper,
including the SBERT cross-miner ablation. It skips any cell whose `result.json` already
exists, making it safe to rerun.

```bash
# DistilBERT / MiniLM (default batch settings)
bash scripts/run_full_sweep_one_model.sh

# BERT / ELECTRA (larger models: smaller batch + grad accumulation to fit 8 GB)
MODEL_NAME=bert-base-uncased            TRAIN_BATCH=8 EVAL_BATCH=16 GRAD_ACCUM=2 bash scripts/run_full_sweep_one_model.sh
MODEL_NAME=google/electra-base-discriminator TRAIN_BATCH=8 EVAL_BATCH=16 GRAD_ACCUM=2 bash scripts/run_full_sweep_one_model.sh
```

Encoders used in the paper: `distilbert-base-uncased`, `bert-base-uncased`,
`sentence-transformers/all-MiniLM-L6-v2`, and `google/electra-base-discriminator`.
Fixed hyperparameters (held constant across all cells): 3 epochs, 16 triplets/author/epoch,
`max_length=256`, texts char-truncated to 2000, AdamW at `lr=2e-5`, margin 0.3, triplet
margin loss with mean pooling + L2 normalization.

To run a single cell instead of the full sweep, use
`scripts/run_contrastive_profile_experiment.py --dataset {pan,wildchat} --seed <s>
--model-name <hf-id> ...`.

### 5. Aggregate results

```bash
.venv/bin/python scripts/summarize_contrastive_results.py \
  --results-root outputs/experiments/contrastive_profile_transformers \
  --output-dir   outputs/summaries/contrastive_profile_transformers
```

This regenerates the per-run table and aggregate summaries
(`runs.csv`, `summary.csv`, `summary.json`) that back the tables in the paper. Every
reported number is the mean over seeds 13/21/42 with sample standard deviation.

## Repository layout

```text
configs/                        Experiment and path configuration files
data/                           Local datasets and generated splits (not tracked by Git)
outputs/                        Result JSONs, summaries, and profile stats
scripts/                        Command-line entrypoints (download, build, train, evaluate)
src/identity_verification/      Project package (datasets, training, evaluation, metrics)
tests/                          Tests and small sanity checks
paper/                          LaTeX source of the paper
```

## Metrics

Evaluation is security-oriented: alongside ROC-AUC and PR-AUC, we report EER,
TAR@FAR=1%, FAR@FRR=1%, F1 (at a fixed validation-selected threshold), and streaming
metrics over the first query messages (catch@3, genuine survival rate, median/P90
messages-to-detect). Genuine (same-author) pairs are the positive class, so TAR = TPR and
FAR = FPR. See `src/identity_verification/metrics.py` and
`src/identity_verification/evaluation/`.

## Development

```bash
.venv/bin/python -m pytest -q
```

## Citation

If you use this benchmark or code, please cite:

```bibtex
@inproceedings{sharma-susan-beyond-the-login,
  title     = {Beyond the Login: Continuous Behavioral Authorship Verification for LLM Agent Sessions},
  author    = {Sharma, Muskan and Susan, Seba},
  year      = {2026},
  note      = {Findings of the Association for Computational Linguistics: AACL 2026}
}
```
