
# paper

- Mixture-of-Experts Meets Instruction Tuning:
A Winning Combination for Large Language Models，https://arxiv.org/pdf/2305.14705.pdf, 提出FLAN-MoE，普通finetune之后的moe模型打不过dense model，但是加了instruction tuning之后就beat了dense model|
- LongNet: Scaling Transformers to 1,000,000,000 Tokens，https://arxiv.org/abs//2307.02486, 提出dilated attention，基本思想是随着token间距离越远，attention分配应指数级下降，计算代价只会线性增长。使用dilated attention方法，能够将GPT扩展到1B token长度，论文作者甚至说可以把整个互联网当做一个sequence。但paper里实验只用到了32k长度。
- Demystifying GPT Self-Repair for Code Generation，https://arxiv.org/abs/2306.09896, 基本流程是让code model先生成一个code并做执行，然后使用feedback model（可以和code model是同一个也可以不一样），针对执行结果生成反馈，再将反馈输入给code model。在coding任务上，发现gpt3.5基本没有self-repair能力，只有GPT4才有，并且GPT4在人类反馈的结果上提升最明显。
- Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning，https://arxiv.org/abs//2307.02053, 将Vicuna（在ChatGPT的对话数据上finetune之后的LLaMA模型）在FLAN的子集上进行instruction finetuning后得到的模型，命名为Flacuna，在InstructEVAL上有很大的性能提升。
- InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback，https://intercode-benchmark.github.io/，代码是action，将code执行结果作为observation，不断交互的让模型生成新的代码。主要比较了四种方法：single turn，try again, ReAct，plan & solve。

## Data/pretrain
- [Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models](https://arxiv.org/abs/2307.14430), 主要做Data Selection，从人类学习过程得到启发，在学习更复杂更困难得任务之前，先让模型学习相关的更简单的任务，发现在同样的compute budget下模型能够得到更好的效果。基本方法是先发现数据中的skill graph，然后基于graph进行online sampling，原则是尽量先采样更简单的前置任务以及还没有学习的任务。
- [Language acquisition: do children and language models follow similar learning stages?](https://arxiv.org/abs/2306.03586), 发现模型的语言习得能力也是遵循小孩学习语言的顺序，从简单到复杂。

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
- [Instruction Mining: High-Quality Instruction Data Selection for Large Language Models](https://arxiv.org/abs/2307.06290), 提出instruction mining来从instruction tuning数据集中挖掘高质量的tuning数据集。paper主要是提出了一套指标来衡量tuning数据集的质量，为了验证这套指标的有效性，分别构造了一系列不同质量的数据集，然后实际的finetune一系列对应的LLM，然后观察实际训练出来的LLM效果和这套指标是否能够对应起来，如果实际finetune的LLM效果确实能够与指标对齐，那么就说明这套指标的有效性。而实际finetune的LLM效果是用LLM在一个统一的高质量tuning数据集上的loss来进行判定，这个当做一个groun truth。这个paper的核心贡献是提出了一个指标体系来衡量tuning数据集的质量。

## Alignment
- [Training Socially Aligned Language Models in Simulated Human Society](https://arxiv.org/abs/2305.16960), 提出Stable Alignment方法，具体方法是建了一个由许多个LM agent组成的一个sandbox，然后基于社交规则可以获得模拟的评分，step-by-step的响应以及各种feedback数据。通过这些数据可以进行contrastive SFT（让模型偏好于评分更高的alignment数据上）。Stable Alignment方法不需要强化学习。
- [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217), 先讲RLHF当前的主要问题，然后讲可能得解决方案。在问题部分，主要分成三个方面：human feedback，reward model以及policy，human feedback方面，比如human可能给出不准确的feedback，困难任务上feedback可能本来就很难给，并且还存在许多成本和数据质量的折中；reward model方面，human's value很难用一个reward function来表示等；policy model本身很难训，与真实环境的misgeneralization问题等。解决human feedback的可选方法是用AI feedback（OpenAI基本在押注这个方向），更细粒度的feedback信号，process-based信号等；reward model阶段则可以使用multi-objective,或直接给于feedback，保持一定的uncertainty提高output的多样性。policy model阶段则有直接在pretrain阶段做对齐，或者通过监督学习的手段做对齐。

## Prompt Engineer
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837), 在in-context-learning提供给模型的示例中，发现两个比较重要的点：第一，格式最重要，而示例中是否是golden label并没有那么重要，相比golden label用一个随机label替换效果只有轻微降低，但是完全没有label则会带来很大的损失；第二，input和label的分布空间比较重要。比如input如果是从别的语料中随机采样，或者label是随机选择的一个别的英语单词，这种都属于out of distribution space，则会带来很大的性能损失。这些结果表明LLM在ICL中并没有进行学习（否则就会学到错误的label），只是通过ICL的format和space激发LLM已有的能力，此外在zero shot的设定中，哪怕随机组合一些input和label也能有很明显的效果。这些都说明了LLM非常需要一种结构化的信息。
- [Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango](https://arxiv.org/abs/2209.07686), 将CoT few shot中的样例分成三个部分：symbol, pattern和text，symbol是数字或者实体等，pattern则可以是symbol的组合（比如一个公式或等式）也可以是某种句式等。text则是剩下的文本。paper发现，symbol本身的正确性影响不大，而text和pattern之间的关系则非常重要，text帮助生成useful patterns，而pattern则帮助增强理解任务，使模型能够更好的解决任务。最后，paper将CoT prompt剪枝了20%的token，也能够达到同样差不多的效果。


## Evaluation
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109),
- [Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks](https://arxiv.org/abs/2307.02477), 作者将正常的Evaluation任务中的题全部做了一些更改，比如将二位数加法从十进制改为9进制，python代码中list下标改为从1开始而不是0等等，然后把这样生成的数据叫做Counterfactual Task，结果发现所有模型都在这些新得task上显著下降，这说明模型不是真正的在做reason，而只是某种程度记住了训练数据上的一些潜在数据而已。


## Multimodal

## Code
- [RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349), 提出RLTF (Reinforcement Learning from unit Test Feedback), 利用单元测试的结果作为强化学习的反馈信号，这个idea非常自然。

## Agent
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126), 让GPT4作为tool maker制作一些工具（主要是可调用的Python代码），然后让弱一点的模型作为tool user，这样可以大幅提升小模型的能力。
- [Embodied Task Planning with Large Language Models](https://arxiv.org/abs/2307.01848), 
- [Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners](https://arxiv.org/abs/2307.01928), 让模型知道它们不知道从而寻求人类的帮助。具体就是使用Conformal Prediction方法在预测下一个步骤的时候，给每个可能选项带上分数，如果超过阈值的选项个数大于1个，模型则会反问人类并寻求帮助。
- [Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition](https://arxiv.org/abs/2307.14535), 主要是机器人领域，分成两个大的阶段：第一阶段是利用LLM生成planning数据也就是所谓scale up训练数据，第二阶段是利用生成得到的数据distill到policy model上。生成数据阶段会利用LLM首先生成tree plan，然后转化为一个小段代码保证plan数据的准确。在第二阶段主要使用diffusion policy的方法，基本思想是利用offline RL的思路。
- [Leveraging Implicit Feedback from Deployment Data in Dialogue](https://arxiv.org/abs/2307.14117), 主要是对话场景，分成两个阶段：第一个阶段是通过一些规则和训练一些模型来预测机器说话之后人类的可能反应（比如人类回复文字长度，情感极性等等），这个相当于是训练了一个reranker模型；第二阶段在实际对话过程中先生成20个候选，然后经过reranker做重排，选取分数最高的回复作为最终回复。

## RAG
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909), 提出REALM，基于bert的Mask方法，在预测某个被masked的token时候，去检索Wikipedia语料，拿到结果与当前上下文共同预测这个被masked词。


## Application


## Survey
- [A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT](https://arxiv.org/abs/2302.09419), 这个工作主要从NLP,CV,Graph Learning等领域角度，并且按照底层到下游任务这种角度去切分，比如从word representation,model,instruction finetuning到model compression这种纵向切分的角度展开讨论。
- [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf), 只讨论语言模型，从纵向横切面分别展开：前面主要讨论技术基础（比如scaling law，emergent，OpenAI的早期探索等）和一些可用资源等。后面主要从LLM的底层到上层应用展开，分别包括：数据，模型架构，adaption tuning（包括instruction tuning， alignment tuning，parameter efficient tuning和 memory efficient tuning等），utilization方法（in context learning，CoT prompt，planning prompt等等），最后给出一些prompt engineer的建议和具体的应用方向等。
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169), 分成挑战和应用两个部分。
  挑战包括数据集（near duplicates, benchmark污染，隐私信息，预训练数据mixture，finetuning数据mixture，tokenizer，训练成本太高，过度finetune，inference性能，配套软件比如megatron等，受限的context length，脆弱的prompt，幻觉，对齐问题，信息过时，脆弱的evaluation以及evaluation数据是静态且需要人工标注，无法区分模型生成还是人类数据，无法通过scale解决的困难task）；
  应用包括Chatbots，Computational Biology(比如蛋白质预测)，代码生成（比如codex），创意工作，知识类工作（summarization，for science等），法律，医疗，推理，机器人和具身agent，社会科学，合成数据。
- [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://github.com/Mooler0410/LLMsPracticalGuide), 这个survery内容相对较少，最好的梳理是从encoder-only，decoder-only以及encoder-decoder这三个架构上按照时间线梳理出来一个树状的演化图，除此之外，主要从数据以及下游任务等角度展开，本survey侧重在如何使用大模型的角度。

# blog
- [June 2023, A Stage Review of Instruction Tuning](https://yaofu.notion.site/June-2023-A-Stage-Review-of-Instruction-Tuning-f59dbfc36e2d4e12a33443bd6b2012c2), FLANv2几乎能提升模型所有方面的能力除了human preference，因为Flan本身回复都偏短且主要是NLP任务为主，instruction mixture很重要需要考虑到各个下游能力的平衡（英文能力MMLU，中文C-Eval，推理主要是GSM8k，coding主要是HumanEval等等）。相关Tweet：https://twitter.com/Francis_YAO_/status/1674287552562360329
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), 将大模型的关键能力总结为planning, memory和tool use。针对这些能力详细做了系统的review。
- [复杂推理：大语言模型的北极星能力](https://yaofu.notion.site/6dafe3f8d11445ca9dcf8a2ca1c5b199), 对于LLM而言有两种大致正交的能力：知识+推理。这个blog主要聚焦在推理能力，pretrain阶段的模型建议使用in context chain of thought的方式做评估，经过finetune之后的模型则可能会使得in context能力有下降。此外介绍了一个推理能力的评估集[Chain-of-thought Hub](https://github.com/FranxYao/chain-of-thought-hub). LLM的推理能力在预训练，finetuning，RL和prompt engineer这四个阶段都能获得，包括：
  - 预训练阶段在科学文献/代码上进行训练（推理能力与Code能力有非常强的相关性）
  - 在finetuning阶段可以使用CoT类型的数据
  - 强化学习阶段则可以使用中间信号建模reward，而不是一个最终的答案
  - prompt engineer阶段则可以使用非常多方式提升LLM的推理能力。
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), 系统的介绍。
- 

## OpenAI blog
- [Planning for AGI and beyond](https://openai.com/blog/planning-for-agi-and-beyond), 主要是强调了AGI的safety与alignment问题，其中比较有价值的就是透露了OpenAI在alignment上会使用AI本身来做一些工作，先是让AI来辅助人类做feedback，然后更长远会让AI研究更好的alignment技术。
- [Our approach to alignment research](https://openai.com/blog/our-approach-to-alignment-research), 这篇就是具体讲如何做alignment，分成三步走：第一步是RLHF，第二步是利用AI来辅助人类提供feedback，这个的主要出发点是在一些问题上人类要么很难快速高效的提供feedback（比如对一个很长的书做总结人类可能很难高效率的读完每一本书才能给出feedback），要么是已经超出了人类理解边界（比如AlphaGO下出一个关键步但人类无法理解）。第三步是利用AI直接研究更好的alignment技术。第二步中提到的几个技术：recursive reward modeling (RRM), debate, and iterated amplification. 并且blog中强调目前主要方向在RRM。
- [Learning complex goals with iterated amplification](https://openai.com/research/learning-complex-goals-with-iterated-amplification), 早在2018年，OpenAI便在研究复杂问题中如何引入人类的监督信号，一些真实场景的问题很难类似于简单的supervised learning那样给一个简单的label，因此怎么样把人类的反馈信号引入到这样的学习系统中去，提出了一个想法：先让模型学习一些人类能够直接给出监督信号的简单任务，然后在第二个阶段的稍复杂任务中先让人类做任务分解，分解之后的子任务便可以使用第一阶段已训好的模型能够直接解决。这样便得到了这个稍复杂任务的全部解法和监督信号（人类做任务分解+模型解决子任务），然后再训模型让它直接解决这个稍复杂的任务。通过类似的方法，可以逐步迭代解决更为复杂的任务，因此这个方法叫做Iterated amplification。
- [AI safety via debate](https://openai.com/research/debate), 如果一个任务过于复杂以至于人类无法判断，提出让两个agent不断debate，也就是不断提出它们的argument，直到人类能够做出判断为止。作为原型，OpenAI在mnist任务上让两个agent针对5的图片，一个说是5另一个说是6，然后不断提出自己的论点，最后通过综合它们的论点，一个sparse classifier用了很少的信息便能够得到正确的答案。这两个debate的agent有点像是左右互搏，一个想要尽量欺骗最后的分类器，另一个想要尽量让分类器更准确。
- [Scalable agent alignment via reward modeling](https://deepmindsafetyresearch.medium.com/scalable-agent-alignment-via-reward-modeling-bf4ab06dfd84), 这个其实是DeepMind的工作，但OpenAI多次引用，并说他们现在主要在follow这个工作。主要出发点就是游戏场景有足够清晰的reward signal，但是真实场景没有，因此需要引入human feedback来对其人类的喜好和意图，并且定义agent alignment problem as follows: How can we create agents that behave in accordance with the user’s intentions? 简单问题可以通过一个reward model来解决，这也是目前RLHF主要的情形，但是复杂问题需要用到recursive reward modeling，简单说就是分层的reward以及区分不同方面的reward model，文中还说这个所谓recursive reward modeling是OpenAI提出的Iterated amplification的一个实例。但要实现这个方法，还有很多相关问题需要解决，比如hierarchy feedback，online feedback，leveraging existing data等等。


# Dataset
- [Chinese Open Instruction Generalist](https://huggingface.co/datasets/BAAI/COIG), 中文instruction tuning数据集。
