# Data Directory

Local datasets should be placed here, but raw and processed data are intentionally not tracked by Git.

Suggested layout:

```text
data/
  raw/
    pan/
    wildchat/
  processed/
```

## Download

Use the reproducible downloader from the repository root:

```bash
python scripts/download_dataset.py --dataset all
```

It skips existing files by default. Use `--dry-run` to preview actions and
`--verify-existing` to verify the PAN zip checksum.
