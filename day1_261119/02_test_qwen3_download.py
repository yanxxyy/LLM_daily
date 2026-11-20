#!/usr/bin/env python3
"""Verify a local Qwen3-8B download, then run one offline GPU inference.

Run on the RTX 5090 server after installing transformers and accelerate:
    python 02_test_qwen3_download.py
    python 02_test_qwen3_download.py --model-dir /path/to/snapshots/master

This script does not download a model. It reads the safetensors shard headers
and their expected lengths, loads every weight with Transformers, and asks
the model one short question. A successful load is stronger evidence than a
file count or a stale download progress bar.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_MODEL_DIR = "/root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master"


def check_shard(path: Path) -> None:
    actual_bytes = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"Incomplete safetensors header: {path}")
        header_length = int.from_bytes(raw_length, "little")
        if not 2 <= header_length <= 64 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header length: {path}")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise RuntimeError(f"Truncated safetensors header: {path}")
    try:
        tensors = json.loads(header)
        ends = [int(info["data_offsets"][1]) for name, info in tensors.items() if name != "__metadata__"]
        if not ends:
            raise ValueError("no tensors")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise RuntimeError(f"Invalid safetensors metadata: {path}") from exc
    expected_bytes = 8 + header_length + max(ends)
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Truncated/corrupt shard {path.name}: {actual_bytes:,} bytes on disk, "
            f"{expected_bytes:,} bytes expected"
        )
    print(f"OK {path.name}: {actual_bytes / 1024**3:.2f} GiB", flush=True)


def check_files(model_dir: Path) -> None:
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json: {model_dir}")
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors.index.json: {model_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    if not shard_names:
        raise RuntimeError("The model shard index is empty")
    print(f"Checking {len(shard_names)} weight shards...", flush=True)
    for name in shard_names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise RuntimeError(f"Unexpected shard name in index: {name}")
        shard = model_dir / name
        if not shard.is_file():
            raise FileNotFoundError(f"Missing weight shard: {shard}")
        check_shard(shard)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    check_files(model_dir)

    # Guarantee the test cannot silently fetch missing files from the network.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on the 5090 server")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16")
    device = torch.cuda.get_device_properties(0)
    print(f"Loading Qwen3-8B on {device.name} ({device.total_memory / 1024**3:.1f} GiB)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "请只回答数字：1 + 1 等于几？"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = result[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("Weights loaded, but generation returned an empty answer")
    print(f"Model response: {response}")
    print("PASS: all shards are structurally complete and local GPU inference worked.")


if __name__ == "__main__":
    main()
