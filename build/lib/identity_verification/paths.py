"""Path helpers for project-local artifacts."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

