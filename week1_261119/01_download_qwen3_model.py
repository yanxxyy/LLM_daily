#!/usr/bin/env python3
"""Download and verify the complete Qwen3-8B model from ModelScope.

Install only the downloader dependency:
    python -m pip install -U modelscope

Examples on the server:
    python 01_download_qwen3_model.py
    python 01_download_qwen3_model.py --cache-dir /data/modelscope_cache
    python 01_download_qwen3_model.py --verify-only --model-dir /path/to/snapshots/master

The script prints the exact directory to pass to
`05_qwen3_grpo_hotpotqa.py train --model-dir ...`. ModelScope reuses its cache
when the same model/revision is downloaded again.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def verify_safetensors(path: Path) -> set[str]:
    """Check the header, indexed tensors, and exact expected shard length."""
    size = path.stat().st_size
    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise RuntimeError(f"Incomplete safetensors header: {path}")
        header_size = int.from_bytes(header_size_bytes, "little")
        if header_size < 2 or header_size > 64 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length: {path}")
        header = file.read(header_size)
        if len(header) != header_size:
            raise RuntimeError(f"Truncated safetensors header: {path}")
    try:
        entries = json.loads(header)
        names = set(entries) - {"__metadata__"}
        if not names:
            raise ValueError("no tensors")
        data_end = max(int(entries[name]["data_offsets"][1]) for name in names)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise RuntimeError(f"Invalid safetensors index: {path}") from exc
    expected_size = 8 + header_size + data_end
    if size != expected_size:
        raise RuntimeError(
            f"Incomplete/corrupt shard {path.name}: actual {size:,} bytes, "
            f"expected {expected_size:,} bytes"
        )
    return names


def verify_model(model_dir: Path) -> tuple[int, int]:
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"Missing config.json in: {model_dir}")
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing model.safetensors.index.json in: {model_dir}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Invalid model shard index: {index_path}") from exc
    if not weight_map:
        raise RuntimeError(f"Empty model shard index: {index_path}")
    shard_names = set(weight_map.values())
    total_bytes = 0
    for shard_name in sorted(shard_names):
        if Path(shard_name).name != shard_name or not shard_name.endswith(".safetensors"):
            raise RuntimeError(f"Unexpected shard path in index: {shard_name}")
        shard = model_dir / shard_name
        if not shard.is_file():
            raise RuntimeError(f"Missing shard: {shard}")
        actual_names = verify_safetensors(shard)
        indexed_names = {tensor for tensor, file in weight_map.items() if file == shard_name}
        if actual_names != indexed_names:
            raise RuntimeError(f"Shard tensor names do not match index: {shard_name}")
        total_bytes += shard.stat().st_size
        print(f"Verified {shard_name}: {shard.stat().st_size / 1024**3:.2f} GiB", flush=True)
    return len(shard_names), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B", help="ModelScope repository ID")
    parser.add_argument("--revision", help="Optional fixed ModelScope revision")
    parser.add_argument("--cache-dir", help="Optional ModelScope cache directory")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without downloading")
    parser.add_argument("--model-dir", help="Existing snapshot directory; required with --verify-only")
    args = parser.parse_args()

    if args.verify_only:
        if not args.model_dir:
            parser.error("--verify-only requires --model-dir")
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        if args.model_dir:
            parser.error("--model-dir is only used with --verify-only")
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise SystemExit("Missing modelscope. Run: python -m pip install -U modelscope") from exc
        options: dict[str, str] = {}
        if args.revision:
            options["revision"] = args.revision
        if args.cache_dir:
            cache_dir = Path(args.cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            options["cache_dir"] = str(cache_dir)
        print(f"Downloading {args.model_id} from ModelScope...", flush=True)
        model_dir = Path(snapshot_download(args.model_id, **options)).resolve()

    shard_count, total_bytes = verify_model(model_dir)

    print(f"\nModel directory: {model_dir}")
    print(f"Verified weight files: {shard_count} ({total_bytes / 1024**3:.2f} GiB)")
    print("Train with the local model:")
    print(f"python 05_qwen3_grpo_hotpotqa.py train --model-dir {shlex.quote(str(model_dir))}")


if __name__ == "__main__":
    main()
