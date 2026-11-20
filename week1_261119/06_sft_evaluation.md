# HotpotQA SFT 前后效果对比

先把修订后的 `04_qwen3_sft_hotpotqa.py` 和 `05_qwen3_grpo_hotpotqa.py` 上传到 5090 服务器，覆盖同名脚本。此次 SFT 报错发生在构造训练目标时，模型尚未开始训练；重新运行完整 SFT 即可。修订后的 04 脚本会优先保留两篇文档的证据引用，缩短过长目标，并报告少数仍放不下而被跳过的样本数。

```bash
python 04_qwen3_sft_hotpotqa.py \
  --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
  --dataset-dir /root/data/hotpotqa_distractor \
  --output-dir ./hotpot_sft_run
```

训练结束后，确认 `./hotpot_sft_run/final_adapter/adapter_config.json` 已生成。此前 `--train-limit 32 --max-steps 2` 只跑了两步，不能用它的 adapter 判断完整 SFT 效果。

## 同一批题测试基础模型与 SFT 模型

两次推理都读取同一个本地 HotpotQA validation 集，使用相同的 4096 token prompt 上限、192 token 生成上限与贪心解码。两次都用 NF4 加载 Qwen3-8B 底座，唯一变化是第二次加载 SFT LoRA。

```bash
python 05_qwen3_grpo_hotpotqa.py evaluate \
  --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
  --dataset-dir /root/data/hotpotqa_distractor \
  --max-prompt-length 4096 --max-completion-length 192 \
  --predictions-file ./eval_base.jsonl

python 05_qwen3_grpo_hotpotqa.py evaluate \
  --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
  --dataset-dir /root/data/hotpotqa_distractor \
  --adapter-dir ./hotpot_sft_run/final_adapter \
  --max-prompt-length 4096 --max-completion-length 192 \
  --predictions-file ./eval_sft.jsonl
```

上面的命令默认评估完整的 **7,405 道 validation 题**。首次检查流程时，可以在两条命令末尾都加 `--eval-limit 50`，并将结果文件改为 `eval_base_50.jsonl` 与 `eval_sft_50.jsonl`；正式结果再去掉限制重跑。

## 看指标和具体答案

```bash
python 05_qwen3_grpo_hotpotqa.py compare \
  --baseline ./eval_base.jsonl \
  --candidate ./eval_sft.jsonl \
  --show 5
```

- `answer_em`：标准化答案完全匹配的比例，最直观的答对率。
- `answer_f1`：答案词级 F1，反映部分正确。
- `evidence_f1`：证据句 F1 与证据文档标题 F1 的加权结果，用来检查是否找到两跳依据。
- `format_ok`：是否严格输出 `<evidence>...</evidence>` 和 `<answer>...</answer>`。

`compare` 会显示基础模型答错而 SFT 答对的 `NEWLY CORRECT`、反向退步的 `REGRESSION` 和随机题目。逐题看问题、标准答案、标准证据及两次原始输出，可以直观判断模型是否真正学会找证据并作答。若 validation 的答案 EM/F1 与证据 F1 一起提高，且退步题较少，才更有理由说 SFT 提高了多跳 QA 能力；只有 `format_ok` 提高，主要说明学会了输出格式。训练 loss 下降不等于泛化提升。

评估开始时会报告 `all gold evidence visible`。若很多题的标准证据未包含在 4096 token 的 prompt 中，证据分数会受到上下文截断影响；调整 prompt 长度后应把基础模型和 SFT 模型两组都重跑。服务器上的实际指标需等 GPU 推理完成；本地脚本自检不产生模型效果分数。
