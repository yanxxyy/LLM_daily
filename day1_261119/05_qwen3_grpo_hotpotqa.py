#!/usr/bin/env python3
"""Qwen3-8B + TRL GRPO for multi-hop HotpotQA (distractor).

Run on a Linux/WSL server with an RTX 5090 (32 GB), not on Windows Python.
The model is downloaded from ModelScope; HotpotQA is loaded with `datasets`.

For an environment that already has CUDA-enabled PyTorch (including
torch 2.13.0+cu132), keep that installation and add the missing packages:
    python -m pip install -U transformers trl datasets accelerate peft bitsandbytes modelscope
Do not install the CUDA 12.8 PyTorch wheel over an existing CUDA 13.2 wheel.
Run `doctor` before training to verify actual 4-bit CUDA kernel compatibility.

Examples:
    python 05_qwen3_grpo_hotpotqa.py self-test
    python 05_qwen3_grpo_hotpotqa.py doctor
    python 05_qwen3_grpo_hotpotqa.py preview
    python 05_qwen3_grpo_hotpotqa.py train --output-dir ./hotpot_grpo_run
    python 05_qwen3_grpo_hotpotqa.py train --model-dir /path/to/Qwen3-8B --init-adapter-dir ./hotpot_sft_run/final_adapter
    python 05_qwen3_grpo_hotpotqa.py train --model-dir /path/to/Qwen3-8B --dataset-dir ./hotpotqa_distractor --init-adapter-dir ./hotpot_sft_run/final_adapter
    python 05_qwen3_grpo_hotpotqa.py evaluate --predictions-file ./baseline.jsonl
    python 05_qwen3_grpo_hotpotqa.py evaluate --adapter-dir ./hotpot_grpo_run/final_adapter

Use `evaluate` twice (base and SFT adapter) with separate --predictions-file
paths, then use `compare --baseline ... --candidate ...` on the JSONL files.

Training uses every HotpotQA distractor training row by default. The evaluation
command uses every validation row by default. --train-limit, --eval-limit and
--max-steps are explicit smoke-test overrides. Each prompt contains only the
question and supplied context; gold answers and supporting facts are passed
only to the reward function. Overlong contexts are packed without looking at
gold labels, and the fraction of retained gold evidence is reported.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-8B"
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
SYSTEM_PROMPT = (
    "Answer the multi-hop question using only the provided context. "
    "Find evidence from the relevant documents, including the bridge fact. "
    "Use the exact document titles and zero-based sentence numbers shown in the context. "
    "Return at most six evidence references and one short final answer. "
    "Output exactly this format, with no extra text or thinking tags:\n"
    "<evidence>Document Title#0; Other Document#2</evidence>\n"
    "<answer>short answer</answer>"
)
USER_TEMPLATE = "Question: {question}\n\nContext:\n{context}\n\nEvidence and answer:"
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.I | re.S)
FULL_FORMAT_RE = re.compile(
    r"\s*<evidence>\s*.*?\s*</evidence>\s*<answer>\s*.+?\s*</answer>\s*",
    re.I | re.S,
)
WORDS_RE = re.compile(r"[a-z0-9]+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = ARTICLES_RE.sub(" ", value)
    return " ".join(value.split())


def answer_scores(prediction: str, gold: str) -> tuple[float, float]:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    exact = float(pred == target)
    if not pred or not target:
        return exact, exact
    pred_words, target_words = pred.split(), target.split()
    shared = sum((Counter(pred_words) & Counter(target_words)).values())
    if not shared:
        return exact, 0.0
    precision = shared / len(pred_words)
    recall = shared / len(target_words)
    return exact, 2 * precision * recall / (precision + recall)


def parse_completion(completion: Any) -> tuple[str, set[str], bool]:
    # TRL may return a string or a conversational completion.
    if isinstance(completion, list):
        completion = completion[-1].get("content", "") if completion else ""
    content = str(completion)
    answer_match = ANSWER_RE.search(content)
    evidence_match = EVIDENCE_RE.search(content)
    answer = answer_match.group(1).strip() if answer_match else ""
    evidence: set[str] = set()
    if evidence_match:
        for item in re.split(r"[;\n]+", evidence_match.group(1)):
            title, marker, index = item.strip().rpartition("#")
            if marker and title.strip() and index.strip().isdigit():
                evidence.add(f"{title.strip().casefold()}#{int(index.strip())}")
    return answer, evidence, bool(FULL_FORMAT_RE.fullmatch(content))


def set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted or not gold:
        return float(not predicted and not gold)
    shared = len(predicted & gold)
    return 2 * shared / (len(predicted) + len(gold))


def score_completion(completion: Any, answer: str, gold_evidence: list[str]) -> dict[str, float]:
    predicted_answer, evidence, formatted = parse_completion(completion)
    exact, answer_f1 = answer_scores(predicted_answer, answer)
    gold = {str(item).casefold() for item in gold_evidence}
    sentence_f1 = set_f1(evidence, gold)
    predicted_titles = {item.rpartition("#")[0] for item in evidence}
    gold_titles = {item.rpartition("#")[0] for item in gold}
    title_f1 = set_f1(predicted_titles, gold_titles)
    evidence_f1 = 0.7 * sentence_f1 + 0.3 * title_f1
    # Answer remains dominant; exact supporting sentences reward both hops.
    reward = 0.55 * answer_f1 + 0.15 * exact + 0.25 * evidence_f1 + 0.05 * formatted
    return {
        "reward": reward,
        "answer_em": exact,
        "answer_f1": answer_f1,
        "evidence_f1": evidence_f1,
        "format_ok": float(formatted),
    }


def grpo_reward(completions: list[Any], answer: list[str], gold_evidence: list[list[str]], **_: Any) -> list[float]:
    if not (len(completions) == len(answer) == len(gold_evidence)):
        raise ValueError("TRL reward batch fields have different lengths")
    return [
        score_completion(completion, target, evidence)["reward"]
        for completion, target, evidence in zip(completions, answer, gold_evidence)
    ]


def extract_context(row: dict[str, Any]) -> list[tuple[str, int, str]]:
    context = row["context"]
    if not isinstance(context, dict) or "title" not in context or "sentences" not in context:
        raise ValueError("Unexpected HotpotQA context schema; expected title and sentences arrays")
    if len(context["title"]) != len(context["sentences"]):
        raise ValueError("HotpotQA context titles and sentence groups have different lengths")
    return [
        (str(title), index, str(sentence).strip())
        for title, sentences in zip(context["title"], context["sentences"])
        for index, sentence in enumerate(sentences)
        if str(sentence).strip()
    ]


def extract_gold(row: dict[str, Any]) -> list[str]:
    facts = row["supporting_facts"]
    if not isinstance(facts, dict) or "title" not in facts or "sent_id" not in facts:
        raise ValueError("Unexpected HotpotQA supporting_facts schema")
    if len(facts["title"]) != len(facts["sent_id"]):
        raise ValueError("HotpotQA supporting fact titles and indices have different lengths")
    return list(dict.fromkeys(
        f"{str(title).strip()}#{int(index)}"
        for title, index in zip(facts["title"], facts["sent_id"])
    ))


def render_prompt(tokenizer: Any, question: str, lines: list[str]) -> str:
    user = USER_TEMPLATE.format(question=question, context="\n".join(lines))
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def pack_context(
    tokenizer: Any, question: str, sentences: list[tuple[str, int, str]], max_tokens: int
) -> tuple[str, set[str], bool]:
    lines = [f"[{title}#{index}] {sentence}" for title, index, sentence in sentences]
    full_prompt = render_prompt(tokenizer, question, lines)
    all_keys = {f"{title.casefold()}#{index}" for title, index, _ in sentences}
    if token_count(tokenizer, full_prompt) <= max_tokens:
        return full_prompt, all_keys, False

    base_tokens = token_count(tokenizer, render_prompt(tokenizer, question, []))
    budget = max_tokens - base_tokens - 16  # separator/tokenizer safety margin
    if budget <= 0:
        raise ValueError("The question alone exceeds --max-prompt-length")
    qwords = set(WORDS_RE.findall(question.casefold()))
    costs = [token_count(tokenizer, line) + 2 for line in lines]
    first_in_title: dict[str, int] = {}
    for pos, (title, _, _) in enumerate(sentences):
        first_in_title.setdefault(title, pos)
    first = list(first_in_title.values())
    remaining = [i for i in range(len(lines)) if i not in set(first)]
    remaining.sort(
        key=lambda i: (
            len(qwords & set(WORDS_RE.findall((sentences[i][0] + " " + sentences[i][2]).casefold()))),
            -i,
        ),
        reverse=True,
    )
    priority = first + remaining
    selected: list[int] = []
    for pos in priority:
        if costs[pos] <= budget:
            selected.append(pos)
            budget -= costs[pos]
    if not selected:
        raise ValueError("No context sentence fits --max-prompt-length")
    while selected:
        selected_lines = [lines[i] for i in sorted(selected)]
        prompt = render_prompt(tokenizer, question, selected_lines)
        if token_count(tokenizer, prompt) <= max_tokens:
            visible = {f"{sentences[i][0].casefold()}#{sentences[i][1]}" for i in selected}
            return prompt, visible, True
        selected.pop()  # remove the lowest-priority retained sentence
    raise ValueError("Unable to pack this question within --max-prompt-length")


def prepare_row(row: dict[str, Any], tokenizer: Any, max_prompt_length: int) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError("HotpotQA training/validation row has an empty question or answer")
    gold = extract_gold(row)
    prompt, visible, compacted = pack_context(tokenizer, question, extract_context(row), max_prompt_length)
    gold_keys = {item.casefold() for item in gold}
    return {
        "prompt": prompt,
        "question": question,
        "answer": answer,
        "gold_evidence": gold,
        "context_compacted": compacted,
        "gold_evidence_visible": gold_keys <= visible,
    }


def load_hotpot(args: argparse.Namespace) -> Any:
    from datasets import DatasetDict, load_dataset, load_from_disk

    if args.dataset_dir:
        requested = Path(args.dataset_dir).expanduser().resolve()
        if requested.exists() and not (requested / "dataset_dict.json").is_file():
            raise ValueError(f"--dataset-dir is not a saved HotpotQA DatasetDict: {requested}")
        if not requested.exists():
            candidates = (
                Path.cwd() / "hotpotqa_distractor",
                Path(__file__).resolve().parent / "hotpotqa_distractor",
            )
            fallback = next((path.resolve() for path in candidates if (path / "dataset_dict.json").is_file()), None)
            if fallback is None:
                raise FileNotFoundError(
                    f"HotpotQA dataset directory does not exist: {requested}. "
                    "Run the dataset downloader with --output-dir set to this path, "
                    "or pass the actual saved directory to --dataset-dir."
                )
            print(f"Dataset directory {requested} is missing; using {fallback}", flush=True)
            requested = fallback
        data = load_from_disk(str(requested))
    else:
        kwargs: dict[str, Any] = {}
        if args.dataset_revision:
            kwargs["revision"] = args.dataset_revision
        try:
            data = load_dataset(args.dataset_id, args.dataset_config, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {args.dataset_id}/{args.dataset_config} with datasets. "
                "The dataset is fetched from Hugging Face, independently of the ModelScope model. "
                "Check server connectivity or set HF_ENDPOINT to a trusted mirror."
            ) from exc
    if not isinstance(data, DatasetDict) or "train" not in data or "validation" not in data:
        raise ValueError("Expected HotpotQA train and validation splits")
    required = {"question", "answer", "context", "supporting_facts"}
    for split in ("train", "validation"):
        missing = required - set(data[split].column_names)
        if missing:
            raise ValueError(f"{split} is missing HotpotQA columns: {sorted(missing)}")
    return data


def resolve_model(args: argparse.Namespace) -> str:
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        return str(model_dir)
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"Downloading/loading {args.model_id} from ModelScope...", flush=True)
    kwargs: dict[str, Any] = {}
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    return str(snapshot_download(args.model_id, **kwargs))


def load_tokenizer(model_dir: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on the RTX 5090 server with CUDA PyTorch.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16.")
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 1024**3
    print(f"CUDA device: {props.name}, {gib:.1f} GiB", flush=True)
    if gib < 20:
        raise RuntimeError("This 8B QLoRA GRPO configuration needs a GPU with at least 20 GiB VRAM.")


def doctor() -> None:
    """Check the actual server stack before an expensive model download."""
    import torch

    for package in ("torch", "transformers", "trl", "datasets", "accelerate", "peft", "bitsandbytes", "modelscope"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing package: {package}. Install the packages listed in this file's docstring.") from exc
        print(f"{package}: {version}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    check_gpu()
    from transformers import AutoConfig
    from trl import GRPOConfig, GRPOTrainer

    AutoConfig.for_model("qwen3")
    if "processing_class" not in inspect.signature(GRPOTrainer).parameters:
        raise RuntimeError("Installed TRL GRPOTrainer does not support processing_class")
    if "num_generations" not in inspect.signature(GRPOConfig).parameters:
        raise RuntimeError("Installed TRL GRPOConfig is incompatible with this script")
    try:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear4bit(
            16, 16, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
        ).to("cuda:0")
        with torch.inference_mode():
            result = layer(torch.ones((1, 16), device="cuda:0", dtype=torch.bfloat16))
        if result.shape != (1, 16) or not torch.isfinite(result).all():
            raise RuntimeError("4-bit CUDA test produced an invalid result")
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes 4-bit NF4 failed on this CUDA/PyTorch stack. "
            "Do not start GRPO yet; install a bitsandbytes build that supports the server's CUDA runtime."
        ) from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Doctor passed: CUDA, bf16 and bitsandbytes NF4 are usable.")


def load_quantized_model(model_dir: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


def prepare_split(args: argparse.Namespace, tokenizer: Any, split: Any, name: str) -> Any:
    limit = args.train_limit if name == "train" else args.eval_limit
    if limit is not None:
        split = split.select(range(min(limit, len(split))))
    prepared = split.map(
        lambda row: prepare_row(row, tokenizer, args.max_prompt_length),
        remove_columns=split.column_names,
        desc=f"Preparing {name} prompts",
    )
    compacted = sum(prepared["context_compacted"])
    visible = sum(prepared["gold_evidence_visible"])
    print(
        f"{name}: {len(prepared)} rows; compacted contexts: {compacted}; "
        f"all gold evidence visible: {visible}/{len(prepared)}",
        flush=True,
    )
    return prepared


def build_grpo_config(config_class: Any, args: argparse.Namespace) -> Any:
    """Keep essential GRPO settings and skip optional fields removed by TRL."""
    options: dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=args.max_completion_length,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        use_vllm=False,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    accepted = inspect.signature(config_class).parameters
    required = {
        "output_dir", "num_train_epochs", "max_steps", "learning_rate",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_generations", "max_completion_length", "beta", "bf16",
        "gradient_checkpointing", "remove_unused_columns",
    }
    missing = required - set(accepted)
    if missing:
        raise RuntimeError(
            f"Installed TRL GRPOConfig is missing essential settings {sorted(missing)}. "
            "Please send `python -m pip show trl transformers`."
        )
    # Prompts are packed to this limit before GRPOTrainer receives them.
    if "max_prompt_length" in accepted:
        options["max_prompt_length"] = args.max_prompt_length
    skipped = sorted(set(options) - set(accepted))
    if skipped:
        print(f"GRPOConfig does not support optional settings {skipped}; using library defaults.", flush=True)
    return config_class(**{key: value for key, value in options.items() if key in accepted})


def train(args: argparse.Namespace) -> None:
    doctor()
    from peft import LoraConfig, PeftModel
    from trl import GRPOConfig, GRPOTrainer

    model_dir = resolve_model(args)
    tokenizer = load_tokenizer(model_dir)
    data = load_hotpot(args)
    train_data = prepare_split(args, tokenizer, data["train"], "train")
    validation_data = prepare_split(args, tokenizer, data["validation"], "validation")
    model = load_quantized_model(model_dir)
    if args.init_adapter_dir:
        adapter_dir = Path(args.init_adapter_dir).expanduser().resolve()
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No SFT adapter_config.json in {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        lora = None  # Continue updating the SFT adapter, not a new random one.
        print(f"Initializing GRPO from SFT adapter: {adapter_dir}", flush=True)
    else:
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    config = build_grpo_config(GRPOConfig, args)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward,
        args=config,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = Path(args.output_dir) / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter to {final_dir.resolve()}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    doctor()
    import torch
    from peft import PeftModel

    model_dir = resolve_model(args)
    tokenizer = load_tokenizer(model_dir)
    data = load_hotpot(args)
    validation = prepare_split(args, tokenizer, data["validation"], "validation")
    model = load_quantized_model(model_dir)
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir).expanduser().resolve()
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No LoRA adapter_config.json in {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.config.use_cache = True
    model.eval()

    destination = Path(args.predictions_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    with destination.open("w", encoding="utf-8") as output:
        for index, row in enumerate(validation):
            inputs = tokenizer(row["prompt"], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            metrics = score_completion(completion, row["answer"], row["gold_evidence"])
            for key in ("answer_em", "answer_f1", "evidence_f1", "format_ok"):
                totals[key] += metrics[key]
            output.write(json.dumps({
                "question": row["question"],
                "gold_answer": row["answer"],
                "gold_evidence": row["gold_evidence"],
                "completion": completion,
                "metrics": metrics,
                "context_compacted": row["context_compacted"],
                "gold_evidence_visible": row["gold_evidence_visible"],
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0:
                print(f"Evaluated {index + 1}/{len(validation)}", flush=True)
    summary = {key: round(totals[key] / len(validation), 4) for key in totals}
    print(json.dumps({"samples": len(validation), "metrics": summary, "predictions": str(destination)}, indent=2))


def read_predictions(path: str) -> list[dict[str, Any]]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def compare_prediction_rows(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[dict[str, tuple[float, float]], list[int], list[int]]:
    if len(baseline) != len(candidate):
        raise ValueError("Prediction files have different sample counts")
    keys = ("answer_em", "answer_f1", "evidence_f1", "format_ok")
    before_totals: Counter[str] = Counter()
    after_totals: Counter[str] = Counter()
    wins: list[int] = []
    regressions: list[int] = []
    for index, (before, after) in enumerate(zip(baseline, candidate)):
        identity = ("question", "gold_answer", "gold_evidence")
        if any(before.get(key) != after.get(key) for key in identity):
            raise ValueError(f"Validation rows differ at index {index}; evaluate the same dataset and limit")
        config_keys = ("max_prompt_length", "max_completion_length")
        if any(
            key in before and key in after and before[key] != after[key]
            for key in config_keys
        ):
            raise ValueError(f"Generation settings differ at validation index {index}")
        before_metrics = score_completion(before["completion"], before["gold_answer"], before["gold_evidence"])
        after_metrics = score_completion(after["completion"], after["gold_answer"], after["gold_evidence"])
        for key in keys:
            before_totals[key] += before_metrics[key]
            after_totals[key] += after_metrics[key]
        if not before_metrics["answer_em"] and after_metrics["answer_em"]:
            wins.append(index)
        elif before_metrics["answer_em"] and not after_metrics["answer_em"]:
            regressions.append(index)
    summary = {
        key: (before_totals[key] / len(baseline), after_totals[key] / len(candidate))
        for key in keys
    }
    return summary, wins, regressions


def compare(args: argparse.Namespace) -> None:
    baseline = read_predictions(args.baseline)
    candidate = read_predictions(args.candidate)
    summary, wins, regressions = compare_prediction_rows(baseline, candidate)
    print(f"Comparing {len(baseline)} held-out validation questions")
    for key, (before, after) in summary.items():
        print(f"{key}: {before:.4f} -> {after:.4f} ({after - before:+.4f})")
    print(f"New exact answers: {len(wins)}; lost exact answers: {len(regressions)}")

    samples = random.Random(args.seed).sample(range(len(baseline)), min(args.show, len(baseline)))
    for label, indices in (("NEWLY CORRECT", wins[:args.show]),
                           ("REGRESSION", regressions[:args.show]),
                           ("RANDOM SAMPLE", samples)):
        for index in indices:
            before, after = baseline[index], candidate[index]
            print(
                f"\n{label} [{index + 1}] {before['question']}\n"
                f"Gold answer: {before['gold_answer']}\n"
                f"Gold evidence: {before['gold_evidence']}\n"
                f"Before: {before['completion']}\n"
                f"After: {after['completion']}\n"
            )


def self_test() -> None:
    gold = ["A#0", "B#1"]
    perfect = "<evidence>A#0; B#1</evidence>\n<answer>Paris</answer>"
    wrong = "<evidence>C#2</evidence>\n<answer>London</answer>"
    assert score_completion(perfect, "Paris", gold)["reward"] == 1.0
    assert score_completion(wrong, "Paris", gold)["reward"] < 0.1
    assert grpo_reward([perfect], ["Paris"], [gold]) == [1.0]
    row = {
        "question": "Where was the author born?",
        "answer": "Paris",
        "context": {"title": ["A", "B"], "sentences": [["A wrote B."], ["B was born in Paris."]]},
        "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
    }
    assert extract_gold(row) == ["A#0", "B#0"]
    assert len(extract_context(row)) == 2
    class FakeTokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
            return "\n".join(message["content"] for message in messages)

        def encode(self, content: str, **_: Any) -> list[str]:
            return content.split()

    fake = FakeTokenizer()
    prepared = prepare_row(row, fake, 512)
    assert prepared["gold_evidence_visible"] and not prepared["context_compacted"]
    assert "A#0" in prepared["prompt"] and "B#0" in prepared["prompt"]
    oversized = [("A", 0, "alpha " * 60), ("B", 0, "beta " * 60)]
    base_length = token_count(fake, render_prompt(fake, row["question"], []))
    packed, visible, compacted = pack_context(fake, row["question"], oversized, base_length + 80)
    assert compacted and len(visible) == 1 and token_count(fake, packed) <= base_length + 80
    baseline_row = {
        "question": "Where?", "gold_answer": "Paris", "gold_evidence": gold,
        "completion": wrong,
    }
    candidate_row = {**baseline_row, "completion": perfect}
    summary, wins, regressions = compare_prediction_rows([baseline_row], [candidate_row])
    assert summary["answer_em"] == (0.0, 1.0) and wins == [0] and not regressions
    print("Self-test passed: schema parsing and GRPO reward.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("self-test", "doctor", "preview", "train", "evaluate", "compare"))
    parser.add_argument("--model-id", default=MODEL_ID, help="ModelScope model ID")
    parser.add_argument("--model-dir", help="Existing local ModelScope snapshot; skips download")
    parser.add_argument("--model-revision", help="Optional ModelScope model revision")
    parser.add_argument("--dataset-id", default=DATASET_ID, help="Hugging Face datasets ID")
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-revision", help="Optional Hugging Face dataset revision")
    parser.add_argument("--dataset-dir", help="Local HotpotQA DatasetDict saved by 03_download_qa_datasets.py")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--train-limit", type=int, help="Smoke test only; default is the full train split")
    parser.add_argument("--eval-limit", type=int, help="Smoke test only; default is the full validation split")
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke test override; -1 trains one full epoch")
    parser.add_argument("--output-dir", default="./hotpot_grpo_run")
    parser.add_argument("--init-adapter-dir", help="SFT LoRA adapter used to initialize GRPO training")
    parser.add_argument("--adapter-dir", help="LoRA adapter for evaluate; omit to score the base model")
    parser.add_argument("--predictions-file", default="./hotpot_grpo_predictions.jsonl")
    parser.add_argument("--baseline", help="Base-model evaluation JSONL for compare")
    parser.add_argument("--candidate", help="SFT/GRPO evaluation JSONL for compare")
    parser.add_argument("--show", type=int, default=3, help="Examples per comparison category")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_prompt_length < 512 or args.max_completion_length < 32:
        parser.error("prompt/completion token limits are too small")
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be positive")
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("--eval-limit must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.command == "compare" and (not args.baseline or not args.candidate):
        parser.error("compare requires --baseline and --candidate")
    if args.show < 0:
        parser.error("--show must be nonnegative")
    return args


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "doctor":
        doctor()
    elif args.command == "preview":
        data = load_hotpot(args)
        print(json.dumps({
            "dataset": args.dataset_id,
            "config": args.dataset_config,
            "train_rows": len(data["train"]),
            "validation_rows": len(data["validation"]),
            "columns": data["train"].column_names,
            "sample_question": data["train"][0]["question"],
            "sample_supporting_facts": data["train"][0]["supporting_facts"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "train":
        train(args)
    elif args.command == "compare":
        compare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
