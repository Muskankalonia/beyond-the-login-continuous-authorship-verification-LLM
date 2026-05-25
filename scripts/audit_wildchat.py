#!/usr/bin/env python
"""Audit WildChat-1M style conversation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_verification.datasets.wildchat import audit_wildchat_dataset, format_wildchat_audit
from identity_verification.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to WildChat file or directory.")
    parser.add_argument(
        "--user-field",
        default="hashed_ip",
        help="Field used only for grouping conversations by user. Default: hashed_ip.",
    )
    parser.add_argument("--output-json", help="Optional path to save the audit report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = audit_wildchat_dataset(args.input, user_field=args.user_field)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print(format_wildchat_audit(audit))

    if args.output_json:
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON audit report to: {output_path}")


if __name__ == "__main__":
    main()
