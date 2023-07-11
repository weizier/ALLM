# Paper
- [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858), 提出Minerva模型，在PaLM的基础上用technical content进行continual training得到。最终在MATH, GSM8k和STEM（从MMLU中抽取出来的science,technology,engineering and mathematics）上达到了当时的SOTA。
- [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085), 在大量高度curated的专业文献，学科，教科书，讲义和科学网站等数据训练，数据量为106B tokens，并且重复多个epoch有正向收益。在数据组织上有两个比较有意思的trick，第一是把论文的引用用某个特定标记符标志出来这是citation，第二是对于需要多步推导的过程用<work></work>这种标记符进行区分，让模型能够知道这是memory的工作区域而并非最终答案。
- [Scaling Instruction-Finetuned Language Models](Scaling Instruction-Finetuned Language Models), Flan(Finetuning Language model)，paper发现instruction tuning符合scaling law：增大模型或者增大Flan数据集的task数量能带来收益，在CoT和非CoT数据集上进行训练能够带来很大的reasoning能力。
- [Specializing Smaller Language Models towards Multi-Step Reasoning](https://arxiv.org/abs/2301.12726), 牺牲模型的通用能力聚焦在CoT能力上，通过finetune蒸馏GPT-3.5的多步推理过程，可以让小模型具备显著的CoT能力提升。
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275), DeepMind的一篇工作，主要比较在GSM8K数学题上分别使用outcome-based supervised和process-based supervised两种方法，最后发现：在final-answer error rates上，这两种方法相差不大，process-based方法略低，并且使用强化学习都能够比只使用监督方法效果更好；而在trace error rates上（主要考察过程是否正确），需要process-based feedback，且需要reward model。总之：在最终结果的预测上，process based feedback作用不大，但如果要求trace error rates，则process based feedback作用则非常大。并且，无论是最终结果还是trace error rates，基于outcome或者process的reward model都很重要。
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780), 提出CodeRL框架，将生成代码的LM作为actor network，然后crictic network预测代码通过率并给出一个dense feedback(主要的反馈来源包括编译和运行结果或者单元测试结果等)，在inference阶段，模型也可以通过这个crictic network进行动态生成和实时的调整。模型结构使用基于encoder-decoder的CodeT5模型。但似乎2023年放出的CodeT5并没有使用CodeRL.
- [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406), 提出Decomposed Prompting，核心思想是可以将复杂的prompt分解成sub prompt，太长的prompt可以分解为较短的prompt，并且每一个sub prompt还可以递归的进行分解。
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), 相比普通的decode策略，self-consistency则一次采样多个，然后选择一致性最高的答案，类似于多数投票。
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), LLM产生一个初始output之后，重新输入这个LLM中让它给出feedback，然后再基于这个feedback让LLM重新生成新的output，这个过程可以迭代下去。
- [Learning Performance-Improving Code Edits](https://arxiv.org/abs/2302.07867), 开源了一个代码数据集https://pie4perf.com/，里面的每一个代码都提供了由同一个用户编写的不同版本的代码，由时间排序，排在后面的代码版本更正确或者性能更优，也就是包含了trajectory of program。让LLM在这种代码轨迹上学习如何提升代码的质量。最终效果表明在CODEGEN上能够达到Codex相当的效果，而Codex要比CodeGEN模型大十倍。
- [INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models](https://arxiv.org/abs/2306.04757), 代码开源在https://github.com/declare-lab/instruct-eval，作者说instruct tuning的数据质量对模型影响非常大（而从GPT4等生成的数据上进行蒸馏效果则有限），在T5上instruct tuning的数据甚至比model size更重要。整体将InstructEval的数据集分成三大块：Problem-Solving, Writing和Alignment，Problem-Solving的数据集包含：MMLU,BBH,DROP,CRASS和HumanEval，Writing能力包括四个方面：Informative,Proffessional, Argumentative和Creative，Alignment能力包括三个方面：Harmlessness, Honesty和Helpfulness。
- [Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning](https://arxiv.org/abs/2307.02053), 在Flan-mini上微调Vicuna（用ChatGPT的真实对话数据微调LLaMA后得到的模型），Flan-mini数据集包含Flan2021的一个子集，另外再加上ChatGPT/GPT4生成的code和对话数据(主要是Alpaca和ShareGPT的数据)，里面包含了大量problem-solving数据集，实验结果证实在problem-solving能力有大幅提升。训练阶段使用了LoRA方法(不过paper里说LoRA方法可能效果不算很好，future work里希望能够直接finetune)。这个paper主要是把Vicuna放在各种problem solving的dataset上进行了finetune，其他并没有什么新的实现。这篇文章的作者和InstructEval的作者是同一批人。
- [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707), 提出Explanation Tuning，通过imitation learning的方式让模型能够学习GPT4的输出，包括explanation traces, step-by-step thought processes等。从Flan-v2中采样5百万条数据，先用ChatGPT生成，然后再从中采样1百万条用GPT4生成复杂prompt，在请求GPT4时候用到的指令类似于"explain like I’m five, think step-by-step and justify your response...".
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837), 在in-context-learning提供给模型的示例中，发现两个比较重要的点：第一，格式最重要，而示例中是否是golden label并没有那么重要，相比golden label用一个随机label替换效果只有轻微降低，但是完全没有label则会带来很大的损失；第二，input和label的分布空间比较重要。比如input如果是从别的语料中随机采样，或者label是随机选择的一个别的英语单词，这种都属于out of distribution space，则会带来很大的性能损失。这些结果表明LLM在ICL中并没有进行学习（否则就会学到错误的label），只是通过ICL的format和space激发LLM已有的能力，此外在zero shot的设定中，哪怕随机组合一些input和label也能有很明显的效果。这些都说明了LLM非常需要一种结构化的信息。
- [Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango](https://arxiv.org/abs/2209.07686), 将CoT few shot中的样例分成三个部分：symbol, pattern和text，symbol是数字或者实体等，pattern则可以是symbol的组合（比如一个公式或等式）也可以是某种句式等。text则是剩下的文本。paper发现，symbol本身的正确性影响不大，而text和pattern之间的关系则非常重要，text帮助生成useful patterns，而pattern则帮助增强理解任务，使模型能够更好的解决任务。最后，paper将CoT prompt剪枝了20%的token，也能够达到同样差不多的效果。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 让模型输出reason thought，同时也让模型输出action（比如去搜索网络结果），也就是reason（Re） + action(Act)，合并为ReAct。实际的inference过程大致上分为：reason -> action -> observation
- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091), 提出Plan-and-Solve Prompting，就是简单的将Zeroshot CoT中的“Let’s think step by step”替换成“Let’s first understand the problem and devise a plan to solve the problem. Then, let’s carry out the plan and solve the problem step by step”, 本质上是一个prompt engineer的工作。
- [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898), 直言现在都是static instruction-to-code错误有可能传导。提出InterCode，类比强化学习的设定，将code当做action，execution feedback当做observation，主要在Bash和SQL任务上做了一些实验。
- 
- 
- 
- 
- 
- 
- [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403), 相关论文放在了https://github.com/jeffhj/LM-reasoning。将推理主要分为演绎推理，归纳推理和归因推理，以及区分为formal和informal，paper中主要聚焦在informal deductive reasoning问题，但整个paper主要关注finetune和prompt侧，没有提到pretrain阶段的工作。
- [Natural Language Reasoning, A Survey](https://arxiv.org/abs/2303.14725), paper作者更看好defeasible reasoning，和deductive reasoning相比，这种推理方式只会得出可能的结论，也就是不保证推理结果的严格正确性。

# Dataset
- MMLU, 
- DROP, Discrete Reasoning Over Paragraphs
- CRASS, The Counterfactual Reasoning Assessment
- HumanEval
- BigBench

# other
- https://github.com/jeffhj/LM-reasoning
- https://nl-reasoning-workshop.github.io/， ACL2023 Reasoning topic
- [复杂推理：大语言模型的北极星能力](https://yaofu.notion.site/6dafe3f8d11445ca9dcf8a2ca1c5b199), pretrain/finetune/RL/prompt阶段都有办法提高推理能力。