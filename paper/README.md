
# paper

- Mixture-of-Experts Meets Instruction Tuning:
A Winning Combination for Large Language Models，https://arxiv.org/pdf/2305.14705.pdf, 提出FLAN-MoE，普通finetune之后的moe模型打不过dense model，但是加了instruction tuning之后就beat了dense model|
- LongNet: Scaling Transformers to 1,000,000,000 Tokens，https://arxiv.org/abs//2307.02486, 提出dilated attention，基本思想是随着token间距离越远，attention分配应指数级下降，计算代价只会线性增长。使用dilated attention方法，能够将GPT扩展到1B token长度，论文作者甚至说可以把整个互联网当做一个sequence。但paper里实验只用到了32k长度。
- Demystifying GPT Self-Repair for Code Generation，https://arxiv.org/abs/2306.09896, 基本流程是让code model先生成一个code并做执行，然后使用feedback model（可以和code model是同一个也可以不一样），针对执行结果生成反馈，再将反馈输入给code model。在coding任务上，发现gpt3.5基本没有self-repair能力，只有GPT4才有，并且GPT4在人类反馈的结果上提升最明显。
- Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning，https://arxiv.org/abs//2307.02053, 将Vicuna（在ChatGPT的对话数据上finetune之后的LLaMA模型）在FLAN的子集上进行instruction finetuning后得到的模型，命名为Flacuna，在InstructEVAL上有很大的性能提升。
- InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback，https://intercode-benchmark.github.io/，代码是action，将code执行结果作为observation，不断交互的让模型生成新的代码。主要比较了四种方法：single turn，try again, ReAct，plan & solve。

## Data/pretrain

## Engineer
### train
### inference
### attention
### 新硬件
- 墨芯
- AMD

## Model
- long context
- moe

## Finetune
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688), Flan-v2版本，混合了Flan-v1，P3，SNI，包含了zero-shot,few-shot和CoT的数据。paper里发现混合zeroshot和fewshot以及CoT数据，最有助于模型效果的提升，而通过Flan-v2 instruction tuning之后的T5，能够超过几十倍大小模型的OPT效果。

## Alignment
- [Training Socially Aligned Language Models in Simulated Human Society](https://arxiv.org/abs/2305.16960), 提出Stable Alignment方法，具体方法是建了一个由许多个LM agent组成的一个sandbox，然后基于社交规则可以获得模拟的评分，step-by-step的响应以及各种feedback数据。通过这些数据可以进行contrastive SFT（让模型偏好于评分更高的alignment数据上）。Stable Alignment方法不需要强化学习。


## Evaluation
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109), 


## Multimodal

## Code
- [RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349), 提出RLTF (Reinforcement Learning from unit Test Feedback), 利用单元测试的结果作为强化学习的反馈信号，这个idea非常自然。

## Agent
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126), 让GPT4作为tool maker制作一些工具（主要是可调用的Python代码），然后让弱一点的模型作为tool user，这样可以大幅提升小模型的能力。
- [Embodied Task Planning with Large Language Models](https://arxiv.org/abs/2307.01848), 



# blog
- [June 2023, A Stage Review of Instruction Tuning](https://yaofu.notion.site/June-2023-A-Stage-Review-of-Instruction-Tuning-f59dbfc36e2d4e12a33443bd6b2012c2), FLANv2几乎能提升模型所有方面的能力除了human preference，因为Flan本身回复都偏短且主要是NLP任务为主，instruction mixture很重要需要考虑到各个下游能力的平衡（英文能力MMLU，中文C-Eval，推理主要是GSM8k，coding主要是HumanEval等等）。相关Tweet：https://twitter.com/Francis_YAO_/status/1674287552562360329
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), 将大模型的关键能力总结为planning, memory和tool use。针对这些能力详细做了系统的review。
- [复杂推理：大语言模型的北极星能力](https://yaofu.notion.site/6dafe3f8d11445ca9dcf8a2ca1c5b199), 对于LLM而言有两种大致正交的能力：知识+推理。这个blog主要聚焦在推理能力，pretrain阶段的模型建议使用in context chain of thought的方式做评估，经过finetune之后的模型则可能会使得in context能力有下降。此外介绍了一个推理能力的评估集[Chain-of-thought Hub](https://github.com/FranxYao/chain-of-thought-hub). LLM的推理能力在预训练，finetuning，RL和prompt engineer这四个阶段都能获得，包括：
  - 预训练阶段在科学文献/代码上进行训练（推理能力与Code能力有非常强的相关性）
  - 在finetuning阶段可以使用CoT类型的数据
  - 强化学习阶段则可以使用中间信号建模reward，而不是一个最终的答案
  - prompt engineer阶段则可以使用非常多方式提升LLM的推理能力。
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), 系统的介绍。