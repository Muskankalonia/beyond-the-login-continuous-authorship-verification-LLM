#!/usr/bin/env python
"""Download raw PAN and WildChat datasets used by this project.

Defaults:
  - PAN profile-query experiments use the PAN20 large training release,
    hosted on Zenodo record 5106099.
  - WildChat-1M is downloaded from the public Hugging Face dataset
    allenai/WildChat-1M as 14 parquet shards.

The script uses only the Python standard library. It skips files that already
exist unless --force is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zipfile


PAN_RECORD_ID = "5106099"
PAN_TRAIN_ZIP_NAME = "pan20-authorship-verification-training-large.zip"
PAN_TRAIN_ZIP_MD5 = "5dd736507017bbb4f61871d90f12c4e1"
PAN_TRAIN_EXTRACTED_DIR = "pan20-authorship-verification-training-large"
PAN_TRAIN_EXPECTED_FILES = (
    "pan20-authorship-verification-training-large.jsonl",
    "pan20-authorship-verification-training-large-truth.jsonl",
)
WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_REVISION = "main"
WILDCHAT_SHARDS = tuple(f"data/train-{index:05d}-of-00014.parquet" for index in range(14))

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "pan", "pan-train", "wildchat"),
        default="all",
        help="Dataset to download. `pan` means PAN train/calibration data. Default: all.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw datasets are stored. Default: data/raw.",
    )
    parser.add_argument(
        "--pan-record-id",
        default=PAN_RECORD_ID,
        help=f"Zenodo record ID for PAN large data. Default: {PAN_RECORD_ID}.",
    )
    parser.add_argument(
        "--pan-url",
        help="Optional direct PAN zip URL. If omitted, Zenodo API lookup is used.",
    )
    parser.add_argument(
        "--wildchat-repo",
        default=WILDCHAT_REPO,
        help=f"Hugging Face dataset repo. Default: {WILDCHAT_REPO}.",
    )
    parser.add_argument(
        "--wildchat-revision",
        default=WILDCHAT_REVISION,
        help=f"Hugging Face revision/branch. Default: {WILDCHAT_REVISION}.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download PAN zip without extracting it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify checksum for an existing PAN zip instead of only skipping it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading or extracting.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        args.data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"all", "pan", "pan-train"}:
        download_pan_train(args)

    if args.dataset in {"all", "wildchat"}:
        download_wildchat(args)


def download_pan_train(args: argparse.Namespace) -> None:
    pan_root = args.data_root / "pan"
    if not args.dry_run:
        pan_root.mkdir(parents=True, exist_ok=True)
    zip_path = pan_root / PAN_TRAIN_ZIP_NAME

    if zip_path.exists() and not args.force:
        print(f"skip existing: {zip_path}")
        if args.verify_existing:
            verify_md5(zip_path, PAN_TRAIN_ZIP_MD5)
    else:
        pan_url = args.pan_url or resolve_pan_zenodo_url(
            record_id=args.pan_record_id,
            filename=PAN_TRAIN_ZIP_NAME,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        download_file(
            url=pan_url,
            destination=zip_path,
            timeout=args.timeout,
            force=args.force,
            dry_run=args.dry_run,
            expected_md5=PAN_TRAIN_ZIP_MD5,
        )

    if not args.no_extract:
        extract_zip(
            zip_path=zip_path,
            destination=pan_root,
            extracted_dir=PAN_TRAIN_EXTRACTED_DIR,
            expected_files=PAN_TRAIN_EXPECTED_FILES,
            force=args.force,
            dry_run=args.dry_run,
        )


def download_wildchat(args: argparse.Namespace) -> None:
    wildchat_root = args.data_root / "wildchat"
    if not args.dry_run:
        wildchat_root.mkdir(parents=True, exist_ok=True)
    headers = huggingface_headers()

    for shard in WILDCHAT_SHARDS:
        filename = Path(shard).name
        url = huggingface_resolve_url(
            repo=args.wildchat_repo,
            revision=args.wildchat_revision,
            path=shard,
        )
        download_file(
            url=url,
            destination=wildchat_root / filename,
            timeout=args.timeout,
            force=args.force,
            dry_run=args.dry_run,
            headers=headers,
        )


def resolve_pan_zenodo_url(
    record_id: str,
    filename: str,
    timeout: int,
    dry_run: bool,
) -> str:
    fallback_url = (
        f"https://zenodo.org/records/{record_id}/files/"
        f"{urllib.parse.quote(filename)}?download=1"
    )
    if dry_run:
        return fallback_url

    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with open_url(api_url, timeout=timeout) as response:
            record = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"warning: could not read Zenodo API ({exc}); using fallback URL", file=sys.stderr)
        return fallback_url

    for file_info in record.get("files", []):
        if not isinstance(file_info, dict):
            continue
        key = str(file_info.get("key") or file_info.get("filename") or "")
        if key != filename:
            continue
        links = file_info.get("links")
        if isinstance(links, dict):
            for link_key in ("self", "download"):
                url = links.get(link_key)
                if isinstance(url, str) and url:
                    return url

    print(f"warning: {filename} not found in Zenodo API; using fallback URL", file=sys.stderr)
    return fallback_url


def huggingface_resolve_url(repo: str, revision: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path)
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/datasets/{repo}/resolve/{quoted_revision}/{quoted_path}?download=true"


def base_headers() -> dict[str, str]:
    return {"User-Agent": "identity-verification-dataset-downloader/0.1"}


def huggingface_headers() -> dict[str, str]:
    headers = base_headers()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


def open_url(
    url: str,
    timeout: int,
    headers: dict[str, str] | None = None,
    resume_from: int = 0,
):
    request_headers_for_url = base_headers()
    if headers:
        request_headers_for_url.update(headers)
    if resume_from > 0:
        request_headers_for_url["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=request_headers_for_url)
    return urllib.request.urlopen(request, timeout=timeout)


def download_file(
    url: str,
    destination: Path,
    timeout: int,
    force: bool,
    dry_run: bool,
    expected_md5: str | None = None,
    verify_existing: bool = False,
    headers: dict[str, str] | None = None,
) -> None:
    if destination.exists() and not force:
        print(f"skip existing: {destination}")
        if expected_md5 and verify_existing:
            verify_md5(destination, expected_md5)
        return

    if dry_run:
        action = "re-download" if destination.exists() and force else "download"
        print(f"dry-run: {action} {url} -> {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    mode = "ab" if partial_path.exists() and not force else "wb"
    resume_from = partial_path.stat().st_size if mode == "ab" else 0

    print(f"download: {url}")
    print(f"      -> {destination}")
    if resume_from:
        print(f"resuming from byte {resume_from}")

    with open_url(url, timeout=timeout, headers=headers, resume_from=resume_from) as response:
        if resume_from and getattr(response, "status", None) != 206:
            mode = "wb"
            resume_from = 0

        total_size = content_length(response)
        if total_size is not None:
            total_size += resume_from

        write_response_to_file(response, partial_path, mode, resume_from, total_size)

    partial_path.replace(destination)

    if expected_md5:
        verify_md5(destination, expected_md5)


def content_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def write_response_to_file(
    response: Any,
    partial_path: Path,
    mode: str,
    initial_bytes: int,
    total_size: int | None,
) -> None:
    downloaded = initial_bytes
    last_progress = time.monotonic()

    with partial_path.open(mode) as handle:
        while True:
            chunk = response.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
                print_progress(downloaded, total_size)
                last_progress = now

    print_progress(downloaded, total_size, final=True)


def print_progress(downloaded: int, total_size: int | None, final: bool = False) -> None:
    downloaded_gib = downloaded / (1024**3)
    if total_size:
        total_gib = total_size / (1024**3)
        percent = downloaded / total_size * 100
        message = f"progress: {downloaded_gib:.2f}/{total_gib:.2f} GiB ({percent:.1f}%)"
    else:
        message = f"progress: {downloaded_gib:.2f} GiB"
    if final:
        message = f"done: {message}"
    print(message)


def verify_md5(path: Path, expected_md5: str) -> None:
    actual_md5 = hash_file_md5(path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Checksum mismatch for {path}: expected md5 {expected_md5}, got {actual_md5}"
        )
    print(f"verified md5: {path}")


def hash_file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip(
    zip_path: Path,
    destination: Path,
    extracted_dir: str,
    expected_files: tuple[str, ...],
    force: bool,
    dry_run: bool,
) -> None:
    expected_paths = [destination / extracted_dir / name for name in expected_files]
    extracted_path = destination / extracted_dir
    if expected_paths:
        extraction_exists = all(path.exists() for path in expected_paths)
    else:
        extraction_exists = extracted_path.exists() and any(extracted_path.iterdir())

    if extraction_exists and not force:
        print(f"skip existing extraction: {destination / extracted_dir}")
        return

    if dry_run:
        print(f"dry-run: extract {zip_path} -> {destination}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")

    print(f"extract: {zip_path} -> {destination}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)

    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"PAN extraction did not produce expected files: {missing_list}")


if __name__ == "__main__":
    main()
