#!/usr/bin/env python
"""Audit PAN authorship verification data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_verification.datasets.pan import audit_pan_dataset, format_pan_audit
from identity_verification.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Path to PAN pairs JSONL file.")
    parser.add_argument("--truth", required=True, help="Path to PAN truth JSONL file.")
    parser.add_argument("--output-json", help="Optional path to save the audit report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = audit_pan_dataset(args.pairs, args.truth)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print(format_pan_audit(audit))

    if args.output_json:
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON audit report to: {output_path}")


if __name__ == "__main__":
    main()
