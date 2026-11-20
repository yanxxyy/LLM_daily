#!/usr/bin/env python3
"""SFT warm-up for the Qwen3-8B HotpotQA GRPO project.

This file is standalone: it does not import 05_qwen3_grpo_hotpotqa.py. It uses
the same question/context prompt and answer/evidence format as the GRPO
script. Training uses the full HotpotQA distractor train split by default;
--train-limit and --max-steps are for tests.

On the RTX 5090 server, after installing the dependencies listed in
05_qwen3_grpo_hotpotqa.py:

    python 04_qwen3_sft_hotpotqa.py --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master
    python 04_qwen3_sft_hotpotqa.py --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master --dataset-dir ./hotpotqa_distractor
    python 04_qwen3_sft_hotpotqa.py --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master --dataset-dir ./hotpotqa_distractor --eval-loss-steps 200 --eval-loss-limit 128 --qa-eval-steps 200 --qa-eval-limit 32 --save-steps 200
    python 05_qwen3_grpo_hotpotqa.py evaluate --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master --adapter-dir ./hotpot_sft_run/final_adapter
    python 05_qwen3_grpo_hotpotqa.py train --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master --init-adapter-dir ./hotpot_sft_run/final_adapter

The model directory is the local ModelScope download. No model download is
performed by this script. SFT uses only the HotpotQA training split.
Periodic QA evaluation uses a fixed random subset of the held-out validation
split at step 0, every --qa-eval-steps optimizer steps, and the final step.
It adds generation time; results are also saved to qa_eval_history.jsonl.
--eval-loss-steps separately enables teacher-forced validation loss on a fixed
validation subset. It is logged as eval_loss by TRL/Transformers.
Batch size, gradient accumulation and activation checkpointing can be tuned
from the command line; defaults remain 1, 8 and enabled respectively.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


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
WORDS_RE = re.compile(r"[a-z0-9]+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.I | re.S)
FULL_FORMAT_RE = re.compile(
    r"\s*<evidence>\s*.*?\s*</evidence>\s*<answer>\s*.+?\s*</answer>\s*",
    re.I | re.S,
)


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = ARTICLES_RE.sub(" ", value)
    return " ".join(value.split())


def score_qa_completion(completion: str, answer: str, gold_evidence: list[str]) -> dict[str, float]:
    """Match the answer/evidence/format scoring used by 05's evaluate command."""
    answer_match = ANSWER_RE.search(completion)
    evidence_match = EVIDENCE_RE.search(completion)
    predicted_answer = answer_match.group(1).strip() if answer_match else ""
    pred, gold_answer = normalize_answer(predicted_answer), normalize_answer(answer)
    exact = float(pred == gold_answer)
    if not pred or not gold_answer:
        answer_f1 = exact
    else:
        pred_words, gold_words = pred.split(), gold_answer.split()
        shared = sum((Counter(pred_words) & Counter(gold_words)).values())
        answer_f1 = 2 * shared / (len(pred_words) + len(gold_words)) if shared else 0.0

    evidence: set[str] = set()
    if evidence_match:
        for item in re.split(r"[;\n]+", evidence_match.group(1)):
            title, marker, index = item.strip().rpartition("#")
            if marker and title.strip() and index.strip().isdigit():
                evidence.add(f"{title.strip().casefold()}#{int(index.strip())}")
    gold = {str(item).casefold() for item in gold_evidence}

    def set_f1(predicted: set[str], expected: set[str]) -> float:
        if not predicted or not expected:
            return float(not predicted and not expected)
        return 2 * len(predicted & expected) / (len(predicted) + len(expected))

    sentence_f1 = set_f1(evidence, gold)
    title_f1 = set_f1(
        {item.rpartition("#")[0] for item in evidence},
        {item.rpartition("#")[0] for item in gold},
    )
    return {
        "answer_em": exact,
        "answer_f1": answer_f1,
        "evidence_f1": 0.7 * sentence_f1 + 0.3 * title_f1,
        "format_ok": float(bool(FULL_FORMAT_RE.fullmatch(completion))),
    }


def extract_context(row: dict[str, Any]) -> list[tuple[str, int, str]]:
    context = row["context"]
    if not isinstance(context, dict) or "title" not in context or "sentences" not in context:
        raise ValueError("Unexpected HotpotQA context schema")
    if len(context["title"]) != len(context["sentences"]):
        raise ValueError("HotpotQA titles and sentence groups have different lengths")
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
        raise ValueError("HotpotQA fact titles and indices have different lengths")
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
    budget = max_tokens - base_tokens - 16
    if budget <= 0:
        raise ValueError("The question alone exceeds --max-prompt-length")
    qwords = set(WORDS_RE.findall(question.casefold()))
    costs = [token_count(tokenizer, line) + 2 for line in lines]
    first_in_title: dict[str, int] = {}
    for pos, (title, _, _) in enumerate(sentences):
        first_in_title.setdefault(title, pos)
    first = list(first_in_title.values())
    first_set = set(first)
    remaining = [i for i in range(len(lines)) if i not in first_set]
    remaining.sort(
        key=lambda i: (
            len(qwords & set(WORDS_RE.findall((sentences[i][0] + " " + sentences[i][2]).casefold()))),
            -i,
        ),
        reverse=True,
    )
    selected: list[int] = []
    for pos in first + remaining:
        if costs[pos] <= budget:
            selected.append(pos)
            budget -= costs[pos]
    if not selected:
        raise ValueError("No context sentence fits --max-prompt-length")
    while selected:
        prompt = render_prompt(tokenizer, question, [lines[i] for i in sorted(selected)])
        if token_count(tokenizer, prompt) <= max_tokens:
            visible = {f"{sentences[i][0].casefold()}#{sentences[i][1]}" for i in selected}
            return prompt, visible, True
        selected.pop()
    raise ValueError("Unable to pack this question within --max-prompt-length")


def prepare_row(row: dict[str, Any], tokenizer: Any, max_prompt_length: int) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError("HotpotQA row has an empty question or answer")
    gold = extract_gold(row)
    prompt, visible, compacted = pack_context(tokenizer, question, extract_context(row), max_prompt_length)
    return {
        "prompt": prompt,
        "question": question,
        "answer": answer,
        "gold_evidence": gold,
        "context_compacted": compacted,
        "gold_evidence_visible": {fact.casefold() for fact in gold} <= visible,
    }


def load_data_split(args: argparse.Namespace, split_name: str) -> Any:
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
        if not isinstance(data, DatasetDict) or split_name not in data:
            raise ValueError(f"--dataset-dir must be a saved HotpotQA DatasetDict with a {split_name} split")
        split = data[split_name]
    else:
        kwargs: dict[str, str] = {}
        if args.dataset_revision:
            kwargs["revision"] = args.dataset_revision
        split = load_dataset(args.dataset_id, args.dataset_config, split=split_name, **kwargs)
    required = {"question", "answer", "context", "supporting_facts"}
    missing = required - set(split.column_names)
    if missing:
        raise ValueError(f"HotpotQA {split_name} split is missing columns: {sorted(missing)}")
    return split


def load_train_data(args: argparse.Namespace) -> Any:
    return load_data_split(args, "train")


def load_tokenizer(model_dir: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def check_environment() -> None:
    import torch
    from transformers import AutoConfig
    import bitsandbytes as bnb

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires the RTX 5090 server with CUDA and bfloat16")
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)
    AutoConfig.for_model("qwen3")
    layer = bnb.nn.Linear4bit(16, 16, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda:0")
    with torch.inference_mode():
        result = layer(torch.ones((1, 16), device="cuda:0", dtype=torch.bfloat16))
    if result.shape != (1, 16) or not torch.isfinite(result).all():
        raise RuntimeError("bitsandbytes NF4 test failed")
    torch.cuda.empty_cache()
    print("Environment check passed: CUDA, Qwen3, and bitsandbytes NF4.", flush=True)


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
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


def prepare_train_split(args: argparse.Namespace, tokenizer: Any, split: Any) -> Any:
    if args.train_limit is not None:
        split = split.select(range(min(args.train_limit, len(split))))
    prepared = split.map(
        lambda row: prepare_row(row, tokenizer, args.max_prompt_length),
        remove_columns=split.column_names,
        desc="Preparing SFT prompts",
    )
    compacted = sum(prepared["context_compacted"])
    visible = sum(prepared["gold_evidence_visible"])
    print(
        f"train: {len(prepared)} rows; compacted contexts: {compacted}; "
        f"all gold evidence visible: {visible}/{len(prepared)}",
        flush=True,
    )
    return prepared


def make_completion(
    answer: str,
    gold_evidence: list[str],
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
) -> str | None:
    if not answer or not gold_evidence:
        raise ValueError("SFT row needs an answer and supporting facts")
    # Keep the same output format and six-reference limit as the GRPO prompt.
    # Select across titles before taking more references from one document.
    by_title: dict[str, list[str]] = {}
    for fact in gold_evidence:
        title = fact.rpartition("#")[0]
        by_title.setdefault(title, []).append(fact)
    selected: list[str] = []
    while len(selected) < 6 and any(by_title.values()):
        for facts in by_title.values():
            if facts and len(selected) < 6:
                selected.append(facts.pop(0))
    def render(facts: list[str]) -> str:
        return f"<evidence>{'; '.join(facts)}</evidence>\n<answer>{answer}</answer>"

    completion = render(selected)
    if tokenizer is None or max_tokens is None:
        return completion

    # Long document titles can make six citations exceed the completion budget.
    # Keep a citation from each of the first two documents for multi-hop training.
    minimum_references = min(2, len(by_title))
    while (
        len(selected) > minimum_references
        and token_count(tokenizer, completion + tokenizer.eos_token) > max_tokens
    ):
        selected.pop()
        completion = render(selected)
    if token_count(tokenizer, completion + tokenizer.eos_token) > max_tokens:
        return None
    return completion


def build_sft_config(config_class: Any, args: argparse.Namespace) -> Any:
    """Pass only fields supported by the installed TRL/Transformers pair."""
    options = dict(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_prompt_length + args.max_completion_length + 8,
        completion_only_loss=True,
        packing=False,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    if args.gradient_checkpointing:
        options["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    accepted = inspect.signature(config_class).parameters
    required = {
        "output_dir", "num_train_epochs", "max_steps", "learning_rate",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "max_length", "completion_only_loss", "packing", "bf16",
        "gradient_checkpointing",
    }
    missing = required - set(accepted)
    if missing:
        raise RuntimeError(
            f"Installed TRL SFTConfig is too different for this script; missing {sorted(missing)}. "
            "Please send `python -m pip show trl transformers`."
        )
    if args.eval_loss_steps:
        strategy_key = next(
            (key for key in ("eval_strategy", "evaluation_strategy") if key in accepted), None
        )
        required_eval = {"eval_steps", "per_device_eval_batch_size"} - set(accepted)
        if strategy_key is None or required_eval:
            raise RuntimeError(
                "Installed SFTConfig cannot schedule validation loss; missing "
                f"evaluation strategy or {sorted(required_eval)}. "
                "Please send `python -m pip show trl transformers`."
            )
        options[strategy_key] = "steps"
        options["eval_steps"] = args.eval_loss_steps
        options["per_device_eval_batch_size"] = 1
        if "eval_accumulation_steps" in accepted:
            options["eval_accumulation_steps"] = 1
    skipped = sorted(set(options) - set(accepted))
    if skipped:
        print(f"SFTConfig does not support optional settings {skipped}; using library defaults.", flush=True)
    return config_class(**{key: value for key, value in options.items() if key in accepted})


def make_qa_eval_callback(args: argparse.Namespace, tokenizer: Any, rows: list[dict[str, Any]]) -> Any:
    import torch
    from transformers import TrainerCallback

    history = Path(args.output_dir) / "qa_eval_history.jsonl"
    eval_interval = args.qa_eval_steps

    class PeriodicQAEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.last_evaluated_step = -1

        def evaluate_step(self, state: Any, model: Any) -> None:
            if not state.is_world_process_zero or state.global_step == self.last_evaluated_step:
                return
            was_training = model.training
            previous_cache = model.config.use_cache
            totals: Counter[str] = Counter()
            example: dict[str, Any] | None = None
            try:
                model.eval()
                model.config.use_cache = True
                for row in rows:
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
                    scores = score_qa_completion(completion, row["answer"], row["gold_evidence"])
                    totals.update(scores)
                    if example is None:
                        example = {
                            "question": row["question"],
                            "gold_answer": row["answer"],
                            "completion": completion,
                        }
                    del inputs, generated
            finally:
                model.config.use_cache = previous_cache
                model.train(was_training)
            record = {
                "step": state.global_step,
                "samples": len(rows),
                "metrics": {key: round(totals[key] / len(rows), 4) for key in
                            ("answer_em", "answer_f1", "evidence_f1", "format_ok")},
                "example": example,
            }
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.last_evaluated_step = state.global_step
            print("\nQA validation: " + json.dumps(record, ensure_ascii=False), flush=True)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.is_world_process_zero and state.global_step == 0:
                history.parent.mkdir(parents=True, exist_ok=True)
                history.write_text("", encoding="utf-8")
            self.evaluate_step(state, kwargs["model"])
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step and state.global_step % eval_interval == 0:
                self.evaluate_step(state, kwargs["model"])
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.evaluate_step(state, kwargs["model"])
            return control

    return PeriodicQAEvalCallback()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Local ModelScope Qwen3-8B snapshot directory")
    parser.add_argument("--dataset-id", default="hotpotqa/hotpot_qa")
    parser.add_argument("--dataset-config", default="distractor")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-dir", help="Local HotpotQA DatasetDict saved by 03_download_qa_datasets.py")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Use more VRAM to avoid activation recomputation")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--train-limit", type=int, help="Smoke test only; default uses all train rows")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 trains one full epoch")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save a resumable checkpoint every N optimizer steps")
    parser.add_argument("--qa-eval-steps", type=int, default=0, help="Run generation metrics every N optimizer steps; 0 disables it")
    parser.add_argument("--qa-eval-limit", type=int, default=32, help="Fixed random validation sample size for periodic QA metrics")
    parser.add_argument("--eval-loss-steps", type=int, default=0, help="Log teacher-forced validation loss every N optimizer steps; 0 disables it")
    parser.add_argument("--eval-loss-limit", type=int, default=128, help="Fixed random validation sample size for eval_loss")
    parser.add_argument("--output-dir", default="./hotpot_sft_run")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_prompt_length < 512 or args.max_completion_length < 32:
        parser.error("prompt/completion token limits are too small")
    if args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("--per-device-train-batch-size and --gradient-accumulation-steps must be positive")
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.save_steps < 1 or args.qa_eval_steps < 0 or args.qa_eval_limit < 1:
        parser.error("--save-steps and --qa-eval-limit must be positive; --qa-eval-steps must be nonnegative")
    if args.eval_loss_steps < 0 or args.eval_loss_limit < 1:
        parser.error("--eval-loss-steps must be nonnegative and --eval-loss-limit must be positive")

    print(
        f"Training micro-batch: {args.per_device_train_batch_size}; "
        f"gradient accumulation: {args.gradient_accumulation_steps}; "
        f"effective single-GPU batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}; "
        f"gradient checkpointing: {args.gradient_checkpointing}",
        flush=True,
    )

    check_environment()
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    if "completion_only_loss" not in inspect.signature(SFTConfig).parameters:
        raise RuntimeError("Installed TRL SFTConfig does not support completion-only loss")
    if "processing_class" not in inspect.signature(SFTTrainer).parameters:
        raise RuntimeError("Installed TRL SFTTrainer does not support processing_class")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config.json in {model_dir}")
    tokenizer = load_tokenizer(str(model_dir))
    dataset = load_train_data(args)
    prepared = prepare_train_split(args, tokenizer, dataset)
    invisible = len(prepared) - sum(prepared["gold_evidence_visible"])
    if invisible:
        print(
            f"Warning: {invisible} prompts did not retain every gold evidence sentence. "
            "Increase --max-prompt-length if VRAM allows.",
            flush=True,
        )

    def to_sft(row: dict) -> dict:
        full_completion = make_completion(row["answer"], row["gold_evidence"])
        completion = make_completion(
            row["answer"], row["gold_evidence"], tokenizer, args.max_completion_length
        )
        return {
            "prompt": row["prompt"],
            "completion": completion or "",
            "target_too_long": completion is None,
            "target_reduced": completion is not None and completion != full_completion,
        }

    sft_data = prepared.map(
        to_sft,
        remove_columns=prepared.column_names,
        desc="Building completion-only SFT targets",
    )
    too_long = sum(sft_data["target_too_long"])
    reduced = sum(sft_data["target_reduced"])
    if too_long:
        sft_data = sft_data.filter(lambda row: not row["target_too_long"])
    if not len(sft_data):
        raise ValueError("No SFT targets fit --max-completion-length; increase the limit")
    sft_data = sft_data.remove_columns(["target_too_long", "target_reduced"])
    print(
        f"SFT targets: {len(sft_data)} kept; {reduced} shortened to fit "
        f"{args.max_completion_length} tokens; {too_long} skipped because even two "
        "document citations and the answer do not fit.",
        flush=True,
    )
    example = sft_data[0]
    print("Sample target (not included in the prompt):", example["completion"], flush=True)
    callbacks = []
    eval_dataset = None
    if args.qa_eval_steps or args.eval_loss_steps:
        validation = load_data_split(args, "validation")
        sample_count = max(
            args.qa_eval_limit if args.qa_eval_steps else 0,
            args.eval_loss_limit if args.eval_loss_steps else 0,
        )
        indices = random.Random(args.seed).sample(
            range(len(validation)), min(sample_count, len(validation))
        )
        validation_rows = [
            prepare_row(validation[index], tokenizer, args.max_prompt_length) for index in indices
        ]
    if args.qa_eval_steps:
        qa_rows = validation_rows[:args.qa_eval_limit]
        visible = sum(row["gold_evidence_visible"] for row in qa_rows)
        print(
            f"Periodic QA validation: {len(qa_rows)} fixed held-out rows; "
            f"all gold evidence visible: {visible}/{len(qa_rows)}; "
            f"evaluate at step 0, every {args.qa_eval_steps} steps, and the final step.",
            flush=True,
        )
        callbacks.append(make_qa_eval_callback(args, tokenizer, qa_rows))
    if args.eval_loss_steps:
        from datasets import Dataset

        loss_rows = []
        for row in validation_rows[:args.eval_loss_limit]:
            completion = make_completion(
                row["answer"], row["gold_evidence"], tokenizer, args.max_completion_length
            )
            if completion is not None:
                loss_rows.append({"prompt": row["prompt"], "completion": completion})
        if not loss_rows:
            raise ValueError("No validation targets fit --max-completion-length")
        eval_dataset = Dataset.from_list(loss_rows)
        print(
            f"Periodic eval_loss: {len(loss_rows)}/{min(args.eval_loss_limit, len(validation))} "
            f"held-out rows; every {args.eval_loss_steps} optimizer steps.",
            flush=True,
        )
    model = load_quantized_model(str(model_dir))
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = build_sft_config(SFTConfig, args)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_data,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=callbacks,
    )
    if args.eval_loss_steps and not args.resume_from_checkpoint:
        print("Baseline validation loss at step 0:", flush=True)
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.eval_loss_steps and trainer.state.global_step % args.eval_loss_steps:
        print("Final validation loss:", flush=True)
        trainer.evaluate()
    final_dir = Path(args.output_dir) / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"SFT LoRA adapter saved to {final_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
