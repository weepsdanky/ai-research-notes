"""Preflight and verify a license-authorized Ego4D UID download."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Callable, Mapping
from urllib.parse import urlparse

from ego4d_media import (
    GetObject,
    HeadObject,
    OFFICIAL_MANIFEST_PATH,
    ProbeVideo,
    build_download_plan,
    download_exact_media,
    verify_downloaded_media,
)


SnapshotObject = Callable[[str], tuple[bytes, Mapping[str, object]]]


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _aws_access(
    profile: str | None,
) -> tuple[HeadObject, SnapshotObject, GetObject]:
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "boto3 is required for Ego4D preflight; install the official Ego4D CLI"
        ) from error
    client = boto3.Session(profile_name=profile).client("s3")

    def location(s3_path: str) -> tuple[str, str]:
        parsed = urlparse(s3_path)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"invalid Ego4D S3 path: {s3_path}")
        return parsed.netloc, parsed.path.lstrip("/")

    def normalized(response: Mapping[str, object]) -> dict[str, object]:
        return {
            "version_id": response.get("VersionId"),
            "content_length": response.get("ContentLength"),
            "etag": str(response.get("ETag", "")).strip('"'),
        }

    def head(s3_path: str) -> dict[str, object]:
        bucket, key = location(s3_path)
        return normalized(client.head_object(Bucket=bucket, Key=key))

    def get(s3_path: str, version_id: str) -> dict[str, object]:
        bucket, key = location(s3_path)
        response = client.get_object(Bucket=bucket, Key=key, VersionId=version_id)
        return {**normalized(response), "body": response.get("Body")}

    def snapshot(s3_path: str) -> tuple[bytes, Mapping[str, object]]:
        identity = head(s3_path)
        length = identity.get("content_length")
        if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= 100_000_000:
            raise ValueError("Ego4D manifest must be between 1 byte and 100 MB")
        response = get(s3_path, str(identity["version_id"]))
        if any(response.get(field) != identity.get(field) for field in (
            "version_id", "content_length", "etag",
        )):
            raise ValueError("Ego4D manifest changed between HEAD and version-specific GET")
        body = response.get("body")
        if not hasattr(body, "read"):
            raise ValueError("Ego4D manifest GET lacks a streaming body")
        try:
            data = body.read(length + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(data, bytes) or len(data) != length:
            raise ValueError("Ego4D manifest GET returned the wrong byte count")
        return data, {"s3_path": s3_path, **identity}

    return head, snapshot, get


def _validate_output(output: Path, inputs: list[Path], media_dir: Path | None = None) -> None:
    resolved = output.resolve()
    if resolved in {path.resolve() for path in inputs}:
        raise ValueError("Ego4D output must differ from every input")
    if media_dir is not None:
        root = media_dir.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError("Ego4D verification output must be outside the media directory")


def main(
    argv: list[str] | None = None,
    *,
    head_object: HeadObject | None = None,
    snapshot_object: SnapshotObject | None = None,
    get_object: GetObject | None = None,
    sdk_version: str | None = None,
    probe: ProbeVideo | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("candidates", type=Path)
    preflight.add_argument("uid_file", type=Path)
    preflight.add_argument("scan_report", type=Path)
    preflight.add_argument("output", type=Path)
    preflight.add_argument("--aws-profile")
    download = commands.add_parser("download")
    download.add_argument("download_plan", type=Path)
    download.add_argument("media_dir", type=Path)
    download.add_argument("receipt", type=Path)
    download.add_argument("--aws-profile")
    verify = commands.add_parser("verify")
    verify.add_argument("candidates", type=Path)
    verify.add_argument("uid_file", type=Path)
    verify.add_argument("scan_report", type=Path)
    verify.add_argument("metadata", type=Path)
    verify.add_argument("download_plan", type=Path)
    verify.add_argument("download_receipt", type=Path)
    verify.add_argument("version_manifest", type=Path)
    verify.add_argument("media_dir", type=Path)
    verify.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "preflight":
        inputs = [args.candidates, args.uid_file, args.scan_report]
        _validate_output(args.output, inputs)
        frozen = [path.read_bytes() for path in inputs]
        if head_object is None or snapshot_object is None:
            aws_head, aws_snapshot, _ = _aws_access(args.aws_profile)
            head_object = head_object or aws_head
            snapshot_object = snapshot_object or aws_snapshot
        manifest_bytes, manifest_identity = snapshot_object(OFFICIAL_MANIFEST_PATH)
        plan = build_download_plan(
            frozen[0], frozen[1], frozen[2], manifest_bytes, manifest_identity,
            head_object,
        )
        _atomic_json(args.output, plan)
        return 0

    if args.command == "download":
        _validate_output(args.receipt, [args.download_plan], args.media_dir)
        plan_bytes = args.download_plan.read_bytes()
        if get_object is None:
            _, _, get_object = _aws_access(args.aws_profile)
        receipt = download_exact_media(
            plan_bytes, args.media_dir, get_object,
            sdk_version=sdk_version or importlib.metadata.version("boto3"),
        )
        _atomic_json(args.receipt, receipt)
        return 0

    inputs = [
        args.candidates, args.uid_file, args.scan_report, args.metadata,
        args.download_plan, args.download_receipt, args.version_manifest,
    ]
    _validate_output(args.output, inputs, args.media_dir)
    frozen = [path.read_bytes() for path in inputs]
    keyword = {} if probe is None else {"probe": probe}
    report = verify_downloaded_media(
        frozen[0], frozen[1], frozen[2], frozen[3], frozen[4], frozen[5], frozen[6],
        args.media_dir,
        **keyword,
    )
    _atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
