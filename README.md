# Research Paper

This repository contains the code for a conversational identity verification project.

The current Phase 1 plan focuses on:

- PAN profile-query verification derived from PAN training data.
- WildChat-1M profile-query identity verification.
- Random-only vs hardness-aware training.
- Encoder-based verification models and robustness-gap analysis.

See `milestones.txt` for the implementation roadmap.

## Development Setup

Use a project virtual environment for all commands:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,data]'
```

Then run scripts and tests with `.venv/bin/python`, for example:

```bash
.venv/bin/python -m pytest -q
```

## Repository Layout

```text
configs/                 Experiment and path configuration files
data/                    Local datasets, not tracked by Git
outputs/                 Generated reports and artifacts, not tracked by Git
scripts/                 Command-line entrypoints
src/identity_verification/  Project package
tests/                   Tests and small sanity checks
```

## Local Paths

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and edit paths for your machine.
