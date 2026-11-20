### 计划

| 天数 | 任务                                                         | 当天产出                        |
| ---- | ------------------------------------------------------------ | ------------------------------- |
| 1    | 完成基础模型与 SFT 的固定题集评估                            | 指标表、各 10 道答对与答错案例  |
| 2    | 读懂奖励函数，检查答案、证据、格式各项得分及数据是否泄漏     | 奖励规则说明、边界案例          |
| 3    | GRPO 2 步试跑，再运行 100–300 步                             | 可加载的 GRPO adapter、训练日志 |
| 4    | 用同一批验证题比较 SFT 与 GRPO                               | EM/F1/证据 F1 变化和退步案例    |
| 5    | 根据错误案例只调整**一个**因素，例如奖励权重或 prompt 长度，再做一次短程实验 | 两组实验的对照表                |
| 6    | 在完整的 7,405 道 validation 题上评估候选 adapter            | 最终量化结果与人工抽检结论      |
| 7    | 用 vLLM 部署选中的 adapter，复测典型多跳问题并整理复现命令   | 可调用接口、实验报告            |



### day1-261119

#### 脚本使用顺序

1. `01_download_qwen3_model.py`：下载模型。
2. `02_test_qwen3_download.py`：检查模型权重并试运行推理。
3. `03_download_qa_datasets.py`：下载并保存数据集。
4. `04_qwen3_sft_hotpotqa.py`：进行 SFT。
5. `05_qwen3_grpo_hotpotqa.py`：从 SFT adapter 继续 GRPO。

#### 配置Codex_Deepseek_API版





#### 下载Qwen系列开源大模型

##### 让本地Codex生成下载Qwen模型py文件

注⭐：这里用的是ModelScope下载模型，而不是HF，下载速度是比较快的。

1. 见附件：01_download_qwen3_model.py，开启服务器的无卡模式，然后进行下载这个模式。

   * 如果有问题，还可以用Codex_API版本，询问修改。

2. 拖动，上传到服务器上面。

   * ![image-20261119115703050](C:\Users\14794\Desktop\大模型算法学习\大模型算法学习\day1_261119\week1\image-20261119115703050.png)
   * 命令行执行：
     附件：python 01_download_qwen3_model.py
     即可开启模型的下载。
   * 模型下载之后默认的路径：
     /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master
   * 可以通过--cache-dir参数指定模型下载的位置。

3. 清除文件，并重新下载：

   * ```
     rm -rf -- /root/.cache/modelscope/models/Qwen--Qwen3-8B
     python 01_download_qwen3_model.py
     ```

4. 使用测试脚本判断模型是否下载完成：

   ``` python
   # 测试模型下载情况
   python 02_test_qwen3_download.py
   ```



#### 使用TRL框架微调Qwen大模型-SFT

##### 知识点补充

1. 冷启动：只给题目，不让模型看答案 → 模型自己推理 → 只对了的留下来 → 用这些"自己推对的轨迹"做 SFT → 得到初始策略
2. 热启动：拿强模型（GPT-4/Qwen-72B）看题目+答案 → 生成完整推理过程 → 用这些"带答案生成的轨迹"做 SFT → 得到初始策略



##### HF镜像切换

1. 保证HF使用国内的镜像，加快数据/模型下载。
   ``` bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

   * 只针对当前会话的Linux终端有效。



##### 数据集下载和查看

1. 使用下载对应的数据集：
   ``` bash
   python 03_download_qa_datasets.py hotpotqa --output-dir /root/data/hotpotqa_distractor
   ```

   下载脚本会把数据保存为可离线读取的目录。若之前未指定 `--output-dir` 且在 `/root` 运行，默认目录是 `/root/hotpotqa_distractor`。

2. 查看对应数据集的大小：
   ``` bash
   python -c "from datasets import load_dataset; ds=load_dataset('hotpotqa/hotpot_qa','distractor'); print({name: len(rows) for name, rows in ds.items()})"
   ```

   



##### SFT

1. SFT的目的是让模型学会一些指令的，考虑大模型的特性，过度的SFT会让模型出现灾难性遗忘的特性，导致模型泛化能力下降。

2. 这里进行SFT的过程，我们使用TRL库进行，然后环境如下：

   * 这个库的版本到时候可以让Codex帮忙调整，因为不同的初始镜像的Torch版本不一样。

   * 我这里的库如下：
     ``` python
     transformers            5.23.0
     triton                  3.7.1
     trl                     1.18.1
     peft                    0.21.3
     torch                   2.13.0+cu132
     torchvision             0.28.0+cu132
     ```

3. 然后，使用附件的代码进行SFT操作：

   * 开启SFT训练：
     ``` python
     python 04_qwen3_sft_hotpotqa.py \
       --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
       --dataset-dir /root/data/hotpotqa_distractor \
       --train-limit 32 --max-steps 2 --output-dir ./sft_smoke
     ```

   * 这是试跑命令，主要是测试代码是否能正常运行，训练数据被限制在32条。

   * 开启完整正常SFT训练：
      ``` python
      python 04_qwen3_sft_hotpotqa.py \
        --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
        --dataset-dir /root/data/hotpotqa_distractor \
        --output-dir ./hotpot_sft_run
      ```



- [ ] 训练时日志内容解析：
  * {'loss': '0.08107', 'grad_norm': '0.2441', 'learning_rate': '4.983e-05', 'entropy': '0.07514', 'mean_token_accuracy': '0.9744', 'num_tokens': '6.201e+06', 'epoch': '0.03715'} 
    * loss：损失值。
    * grad_norm：梯度值。
      * 主要是分析训练是否稳定。没有出现梯度为0或无穷，基本都是正常的。
    * leraning_rate：学习率
    * entropy：熵值，衡量下一个token的不确定性
    * mean_token_accuracy：目标 token 预测正确率。是对token预测的正确率96.93%，不是96.93%的题答对了
    * num_tokens：目前为止大模型处理的token数
  * 缺少validation_loss，更新后的脚本加上了。



- [ ] SFT-训练时观察训练情况：

  1. 代码：
     ``` bash
     python 04_qwen3_sft_hotpotqa.py \
       --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
       --dataset-dir /root/data/hotpotqa_distractor \
       --max-steps 1000 --save-steps 200 \
       --eval-loss-steps 200 --eval-loss-limit 128 \
       --qa-eval-steps 200 --qa-eval-limit 32 \
       --output-dir ./hotpot_sft_probe
     ```

  2. 新训练命令：
     ``` bash
     python 04_qwen3_sft_hotpotqa.py \
       --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
       --dataset-dir /root/data/hotpotqa_distractor \
       --per-device-train-batch-size 2 \
       --gradient-accumulation-steps 4 \
       --max-steps 100 \
       --output-dir ./sft_batch2_test
     ```

  3. 



##### 效果评估-训练时

1. 看训练损失-training loss曲线：
   * epoch 1: loss 1.8 → 还在快速下降
     epoch 2: loss 0.9 → 还在降
     epoch 3: loss 0.6 → 趋平
     epoch 4: loss 0.55 → 基本不动了
     epoch 5: loss 0.54 → 开始过拟合（validation loss 回升）
   * 只要损失还在降低，可以考虑继续训练。
   * loss降低，也需要看对应的效果，而不是loss降低就继续训练，通过测试的效果不断调整。
2. 同时，考虑验证损失-validation loss曲线：
   * 综合考虑train和valid的损失。
3. 训练时分析训练损失和验证的损失。
   * 大模型的话可以看看：熵等.



##### 效果评估-测试时

1. SFT效果评估，LoRA加载前后的效果。

   * 如何判断模型是否学会这次的QA数据集。

   * 以及如何直观的感受模型微调前后的效果。

2. 每次训练的效果是要训练完之后才能看，所以必须要有验证validation的操作。

   * 训练的效果：
     ``` python
     QA validation: {"step": 0, "samples": 32, "metrics": {"answer_em": 0.5, "answer_f1": 0.6522, "evidence_f1": 0.6051, "format_ok": 1.0}, "example": {"question": "In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?", "gold_answer": "1755", "completion": "<evidence>Moscow State University#0; Moscow State University#3</evidence>\n<answer>1755</answer>"}}
     ```

   * 验证效果的分析。

   * 参于评估的指标，可以自行设计一些对于文本提取的规则和评分规则，一些特殊的打分规则。

     * answer_em：标准化后完全答对 **16/32** 道
     * answer_f1：逐题计算答案的词语重合 F1，再取平均；部分答对也可能得分。
     * evidence_f1：将引用的证据句与文档标题和标准证据比较，分别计算 F1，再按 0.7 × 证据句 F1 + 0.3 × 标题 F1 汇总。
     * format_ok：32 道题都按要求输出了 `<evidence>` 和 `<answer>` 标签





##### 问题QA

1. SFT的Batch是怎么设置的？
   * batch=4 时，并不是"4 个模型同时跑"，而是**把 4 个样本拼成一个张量，一次性送进同一份模型里做矩阵运算**。
   * 大模型训练时数据时已知的，所以这些训练过程，并完全是一个自回归的过程。但是，推理的时候是一个自回归的过程，需要一个token一个token解码
   * 虽然用了**因果掩码**保证每个位置只能看前面的 token，但所有位置的矩阵乘法可以**一次性并行完成**
2. 如何理解大模型训练时候的Batch可以随显存设置和但推理时一个请求只能将Batch=1?
3. 训练完得到的Lora适配器如何开启RLHF？
4. SFT过程？
   * 这是一个demo，目的在于理解这样的过程和跑通对应的流程。
   * 以及改进公共数据集。
5. batch=4和梯度累积4个回合更新是同一个作用吗？



### day2-261120

#### 基础补充

1. **NF4** 是 QLoRA 常用的 4 位权重量化格式（NormalFloat 4-bit）。你的脚本用 NF4 保存冻结的 Qwen3-8B 基础权重，计算时使用 bf16，只更新 LoRA 参数，因此显存占用较低。它不会把 LoRA 本身训练成 4 位。
   * 这里用的是NF4的LoRA量化模型。



#### 验证SFT结果

##### LoRA适配器加载

1. 具体的操作，可见06_sft_evaluation.md的《同一批题测试基础模型与 SFT 模型》章节。
   * 对比了加载适配器和不加载适配器的情况。
   * 通过--adapter-dir加载SFT的适配器，进行后续效果的测试。
   * 加载不同的Adapter进行测试。
2. 用SFT Adapter进行RLHF的训练。
   * 读懂奖励函数。
   * 加载SFT Adapter，进行训练得到GRPO Adapter。
     * 级联加载Adapter训练的思路。
     * SFT Adapter会被GRPO Adapter更新。



##### SFT评估

1. SFT计算奖励的位置：

   * SFT：损失计算的位置模型生成的句子和人工标注的参考句子得分的一个区别。
   * 可以拿模型产生句子的关键词和真实句子的关键词进行匹配，然后计算损失，当然这里只是部分例子

2. 使用对应的脚本，直接运行SFT前后进行操作。

   * 避免自己写代码和脚本，让AI帮忙把自己难以实现的脚本进行实现。

3. 评估SFT结果的脚本和指标。

   * 学术指标：EM(Exact Match完全匹配率，预测的答案和真实答案，是否相同)

     * | 指标                        | 衡量什么                                                     |
       | --------------------------- | ------------------------------------------------------------ |
       | **Answer EM / F1**          | 答案是否完全正确，以及部分词语是否答对。                     |
       | **Supporting Fact EM / F1** | 引用的证据句是否与标准证据一致。                             |
       | **Joint EM / F1**           | **答案和证据同时正确**；这是多跳任务最关键的主指标。         |
       | `eval_loss`                 | 验证集标准答案的预测损失，用来观察训练趋势和过拟合。         |
       | 格式正确率                  | 是否按要求输出 `<evidence>` 和 `<answer>`；属于辅助诊断指标。 |

   * 业务指标（多跳任务）：

     * | 指标               | 可操作的定义                                              |
       | ------------------ | --------------------------------------------------------- |
       | **一次可用率**     | 无需人工改写，答案正确且证据足够支撑结论的题数 ÷ 总题数。 |
       | **引用有效率**     | 引用能在提供的 Context 中找到、句号正确的比例。           |
       | **两跳证据覆盖率** | 两个关键文档的证据都找对的题数 ÷ 总题数。                 |
       | 人工返工率与耗时   | 人需要纠正答案或补证据的比例，以及平均处理时间。          |
       | 端到端成功率       | 请求成功返回、格式可解析且未截断的比例。                  |
       | 时延与成本         | 单题时延 P50/P95、生成 token 数、每千题推理成本。         |

4. 运行测试脚本：
   ``` bash
   python 06_qwen3_sft_eval_hotpotqa.py --self-test
   
   python 06_qwen3_sft_eval_hotpotqa.py \
     --model-dir /root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master \
     --adapter-dir /root/hotpot_sft_probe/final_adapter \
     --dataset-dir /root/data/hotpotqa_distractor \
     --limit 100 --output-dir /root/sft_eval_100
   ```

   * 当前 06 脚本的一次命令会先用 `--model-dir` 评估**无 LoRA 的底座**，完成后才加载 `--adapter-dir` 指定的 LoRA 再评估。`--adapter-dir` 是这条对比命令必需的；删掉它，脚本无法完成前后对比。
   * 模型上测试的日志文件：在当前文件夹下。



##### LoRA-GRPO

1. 为什么要先SFT再RLHF？
   * 核心原因是：预训练模型只会续写文本，不会按指令作答；而强化学习需要一个"已经能稳定产出合理回答"的策略作为起点，否则奖励信号全是负的，模型根本学不到东西。
   * 通俗理解：强化学习需要能稳定产生策略的模型。

2. GRPO指向。TRL会对每条生成结果调用奖励函数，并负责后续的GRPO更新。

   * 需要在GRPOTrainer里面指定对应的训练策略。

   * 这段是新规则示例，不要在已经进行的训练中图替换。



#### Q&A

损失→数据格式→

1. SFT和RLHF等后训练用的损失是什么？

   * SFT和RLHF的差别在于：数据格式和损失函数不同，本质上都是对于大模型输出过程的纠偏。
   * 从优化对象、损失函数、计算位置来分析。
   * 优化对象
     * SFT：token级监督
     * RLHF：sequence级偏好优化，策略优化。
   * 损失函数：
     * SFT：交叉熵
     * RLHF：PPO损失、GRPO损失
   * 计算位置：
     * SFT：
       * 在**每个 answer token 位置**计算；
       * 模型输出 logits；
       * 和标准答案的下一个 token 做交叉熵；
       * 梯度沿整个序列反向传播。
     * RLHF: RM模型的损失(PPO) + 策略模型的损失
       * Actor 模型先生成完整回答；
       * RM 对整条回答打分；
       * Critic 模型估计每个 token 位置的价值；
       * 优势信号再分布到每个 token 上；
       * 最后用 PPO 损失更新策略。

2. SFT和RLHF的数据格式分别长什么样？

   * SFT数据格式：让模型按照某种格式进行内容输出，保证模型的结构化，也方便Agent开发取出内容进行渲染等。

     * 简单指令-响应格式：
       ``` python
       {
         "instruction": "解释什么是强化学习",
         "input": "",
         "output": "强化学习是一种通过奖励和惩罚来优化策略的学习方法。"
       }
       ```

     * 对话格式：
       ``` python
       {
         "messages": [
           {"role": "system", "content": "你是一个有帮助的助手。"},
           {"role": "user", "content": "解释什么是强化学习"},
           {"role": "assistant", "content": "强化学习是一种通过奖励和惩罚来优化策略的学习方法。"}
         ]
       }
       ```

     * prompt-Response格式：
       ``` python
       {
         "prompt": "解释什么是强化学习",
         "response": "强化学习是一种通过奖励和惩罚来优化策略的学习方法。"
       }
       ```

   * RLHF格式：通常是SFT之后，模型已经有一定的格式输出的能力，进一步提升模型的思考能力（例如：Agent的工具调用能力），因为模型既要学习为什么选择，也要学习为什么拒绝，所以模型对于业务有一定思考边界的能力，具有一定选择能力。

     * 偏好对话格式：
       ``` python
       {
         "prompt": "如何快速提高编程能力？",
         "chosen": "多写代码，遇到问题查文档，参加开源项目，并定期复盘。",
         "rejected": "背代码模板，刷几道题就够了。"
       }
       ```

     * 对话模式：
       ``` python
       {
         "prompt": [
           {"role": "user", "content": "如何快速提高编程能力？"}
         ],
         "chosen": [
           {"role": "assistant", "content": "多写代码，遇到问题查文档，参加开源项目，并定期复盘。"}
         ],
         "rejected": [
           {"role": "assistant", "content": "背代码模板，刷几道题就够了。"}
         ]
       }
       ```

3. SFT是不是只有学术指标用来评估？
   * SFT和RLHF都有学术和业务指标来评估。
   * 所以，这里给了一种指标思考的角度，也就是学术和业务指标的。
4. 如果有GRPO的Lora适配器，是不是要加载两个适配器？
   * 不是，只用加载GRPO适配器，其中包括了SFT适配器本身。
