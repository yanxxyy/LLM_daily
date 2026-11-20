#!/usr/bin/env python3
"""Download QA datasets once, then use local files for training.

HotpotQA (default):
    python 03_download_qa_datasets.py hotpotqa --output-dir ./hotpotqa_distractor
    python 04_qwen3_sft_hotpotqa.py --model-dir /path/to/Qwen3-8B --dataset-dir ./hotpotqa_distractor
    python 05_qwen3_grpo_hotpotqa.py train --model-dir /path/to/Qwen3-8B --dataset-dir ./hotpotqa_distractor

FinQA:
    python 03_download_qa_datasets.py finqa --output-dir ./finqa_data
    python qwen3_finqa_pipeline.py sft --model-dir /path/to/Qwen3-8B --data-dir ./finqa_data

HotpotQA uses Hugging Face `datasets` and respects HF_ENDPOINT. FinQA comes
from the official FinQA GitHub project and uses no Hugging Face endpoint.
Neither command downloads the Qwen model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"


def download_hotpot(output_dir: Path) -> None:
    from datasets import load_dataset, load_from_disk

    if output_dir.exists() and (output_dir / "dataset_dict.json").is_file():
        data = load_from_disk(str(output_dir))
        print(f"Using existing local HotpotQA dataset: {output_dir}")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory is not an empty or valid saved dataset: {output_dir}")
        print(f"Downloading HotpotQA from HF endpoint: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}", flush=True)
        data = load_dataset("hotpotqa/hotpot_qa", "distractor")
        required = {"train", "validation"}
        if not required <= set(data):
            raise RuntimeError(f"HotpotQA is missing splits: {sorted(required - set(data))}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        data.save_to_disk(str(output_dir))
        data = load_from_disk(str(output_dir))
        print(f"Saved HotpotQA dataset to: {output_dir}")
    required_columns = {"question", "answer", "context", "supporting_facts"}
    for name in ("train", "validation"):
        if name not in data:
            raise RuntimeError(f"Saved HotpotQA dataset is missing {name}")
        missing = required_columns - set(data[name].column_names)
        if missing:
            raise RuntimeError(f"Saved HotpotQA {name} is missing columns: {sorted(missing)}")
    print("Split counts:", {name: len(split) for name, split in data.items()})
    print(f"Use --dataset-dir {output_dir} for HotpotQA SFT/GRPO.")


def fetch_json(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "qwen3-qa-dataset-downloader/1.0"})
    print(f"Downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    except Exception as exc:
        raise RuntimeError(f"Could not download {url}; check network access to GitHub") from exc


def download_finqa(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train", "train.json"), ("validation", "dev.json")):
        path = output_dir / filename
        fetch_json(f"{FINQA_URL}/{filename}", path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) and isinstance(row.get("qa"), dict) for row in rows
        ):
            raise RuntimeError(f"Invalid FinQA {split} JSON: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{split}: {len(rows)} rows; {path}; SHA256 {digest}...")
    print(f"Use --data-dir {output_dir} for the FinQA pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("hotpotqa", "finqa"), nargs="?", default="hotpotqa")
    parser.add_argument("--output-dir", help="Local dataset folder")
    parser.add_argument("--hf-endpoint", help="Override HF_ENDPOINT for HotpotQA, e.g. https://hf-mirror.com")
    args = parser.parse_args()
    if args.hf_endpoint and args.dataset != "hotpotqa":
        parser.error("--hf-endpoint applies only to hotpotqa")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    default = "./hotpotqa_distractor" if args.dataset == "hotpotqa" else "./finqa_data"
    output_dir = Path(args.output_dir or default).expanduser().resolve()
    if args.dataset == "hotpotqa":
        download_hotpot(output_dir)
    else:
        download_finqa(output_dir)


if __name__ == "__main__":
    main()
