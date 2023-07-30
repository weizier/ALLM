# Paper
## Pretrain
- [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858), 提出Minerva模型，在PaLM的基础上用technical content进行continual training得到。最终在MATH, GSM8k和STEM（从MMLU中抽取出来的science,technology,engineering and mathematics）上达到了当时的SOTA。
- [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085), 在大量高度curated的专业文献，学科，教科书，讲义和科学网站等数据训练，数据量为106B tokens，并且重复多个epoch有正向收益。在数据组织上有两个比较有意思的trick，第一是把论文的引用用某个特定标记符标志出来这是citation，第二是对于需要多步推导的过程用<work></work>这种标记符进行区分，让模型能够知道这是memory的工作区域而并非最终答案。
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644), 主要是在coding问题上做数据selection，数据总共7B tokens，由三部分组成：一部分是从the stack和stackoverflow上挑选出来的数据（6B，使用LM based classifier做过滤），第二部分是用GPT3.5生成的textbook数据（不到1B），第三部分是由GPT3.5生成的Python练习题（180M，也是由GPT3.5生成。），paper里统一把前两个数据集统称为"CodeTextbooks"（实际上两个都不是真正的textbooks），然后把第三步数据称为"CodeExercises"，先在不到7B语料的CodeTextbooks上做pretrain，然后在180M的CodeExercises语料上做finetune，不过需要注意的是模型在这7B数据上过了8遍，相当于计算量是56B数据大小。结果发现1.3b的model （humaneval@pass1 = 50.6%）能够超过大它十几倍以及数据量几十倍上训出来的模型（基本上只比GPT4差了，GPT4的humaneval@pass1 = 67%）。paper里说code textbook相比web语料会更加的clear, self-contained, instructive, and balanced，类比人类学习coding，人也很难从各种庞杂繁复解释又少的充满噪声的web语料中学习到更好的coding技能。过滤the stack和stackoverflow数据的时候先使用GPT4（prompt是"determine its educational value for a student whose goal is to learn basic coding concepts"）标注了100k个样例，然后使用LM输出的embedding特征训练一个random forest分类器类决定语料中哪些是比较好的学习语料。实验表明在180m的CodeExercises数据上finetune之后效果提升最显著，但个人猜测主要是CodeExercises数据格式和HumanEval格式完全匹配，使得模型的instruction following能力大幅增强了，这可能是一个很重要的原因。不过有趣的是模型在其他能力上也有很大的提升。此外，paper里也说在小模型上也观察到了涌现现象。
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564), 
  
## Finetuning
### instruction tuning
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416), Flan(Finetuning Language model)，paper发现instruction tuning符合scaling law：增大模型或者增大Flan数据集的task数量能带来收益，在CoT和非CoT数据集上进行训练能够带来很大的reasoning能力。
- [INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models](https://arxiv.org/abs/2306.04757), 代码开源在https://github.com/declare-lab/instruct-eval, 作者说instruct tuning的数据质量对模型影响非常大（而从GPT4等生成的数据上进行蒸馏效果则有限），在T5上instruct tuning的数据甚至比model size更重要。整体将InstructEval的数据集分成三大块：Problem-Solving, Writing和Alignment，Problem-Solving的数据集包含：MMLU,BBH,DROP,CRASS和HumanEval，Writing能力包括四个方面：Informative,Proffessional, Argumentative和Creative，Alignment能力包括三个方面：Harmlessness, Honesty和Helpfulness。
- [Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning](https://arxiv.org/abs/2307.02053), 在Flan-mini上微调Vicuna（用ChatGPT的真实对话数据微调LLaMA后得到的模型），Flan-mini数据集包含Flan2021的一个子集，另外再加上ChatGPT/GPT4生成的code和对话数据(主要是Alpaca和ShareGPT的数据)，里面包含了大量problem-solving数据集，实验结果证实在problem-solving能力有大幅提升。训练阶段使用了LoRA方法(不过paper里说LoRA方法可能效果不算很好，future work里希望能够直接finetune)。这个paper主要是把Vicuna放在各种problem solving的dataset上进行了finetune，其他并没有什么新的实现。这篇文章的作者和InstructEval的作者是同一批人。
- [Symbol tuning improves in-context learning in language models](https://arxiv.org/abs/2305.08298), 这个文章提出symbol tuning，也就是在instruction tuning中，将few shot示例中的label用一个无关词代替，能够显著提升效果。比如说把情感分类任务重的positive或者negative用一个无关的foo和bar这种标记作为label。这也从侧面印证了模型更重要的是format。
  
### process tuning
- [Specializing Smaller Language Models towards Multi-Step Reasoning](https://arxiv.org/abs/2301.12726), 牺牲模型的通用能力聚焦在CoT能力上，通过finetune蒸馏GPT-3.5的多步推理过程，可以让小模型具备显著的CoT能力提升。
- [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707), 提出Explanation Tuning，通过imitation learning的方式让模型能够学习GPT4的输出，包括explanation traces, step-by-step thought processes等。从Flan-v2中采样5百万条数据，先用ChatGPT生成，然后再从中采样1百万条用GPT4生成复杂prompt，在请求GPT4时候用到的指令类似于"explain like I’m five, think step-by-step and justify your response...".
- [Embodied Task Planning with Large Language Models](https://arxiv.org/abs/2307.01848), 在embodied agent上提出TAsk Planing Agent (TaPA), 在一个simulator中（AI2-THOR），通过不同点的viewpoint，然后识别这些viewpoint中有哪些物体，另外再加上人类的instruction（比如做一个三明治），输入GPT-3.5，让它在object list和instruction的输入下生成详细的task planning步骤，这样便能够得到一个数据集。实验使用LLaMA在这个生成的数据集上进行finetune，然后使用finetune之后的LLaMA生成不同scene下不同task planning，最后请30个志愿者对task planning做评估。这篇paper还只是完成了机器人的第一步，就是用LLM生成任务plan，但并没有让机器人实际去执行这些动作。不过其实也是一篇用推理和规划的中间过程对LLM做finetune的又一个工作。
- [Distilling Script Knowledge from Large Language Models for Constrained Language Planning](https://arxiv.org/abs/2305.05252), 提出一个constrained language planning问题，也就是在带限制条件的plan问题，比如make a cake是一个普通问题，但是make a cake for diabetics就是带条件的plan问题了。通过over-generate-then-filter的方式用ChatGPT最终得到了55000个样本，叫做CoScripts(Constrained Scripts)，最后用这份数据对T5做finetune，效果提升显著。作者把这种script叫做goal-oriented scripts。
- [PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning](https://arxiv.org/abs/2305.19472), 先是通过ChatGPT（teacher model）构造了一个plan的数据集，叫做CoPlan数据集，每一条数据包含(goal,plan,condition)，然后再训练一个小模型（这一步叫做所谓的 symbolic procedural knowledge distillation），最后在inference阶段通过多步搜索方法挑选出比较好的step路径（每一个step都有一个verifier进行打分）。最后通过人工评估，以及和teacher model生成的plan比较BLEU或ROUGE等指标。
- [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301), 蒸馏大模型的step by step中间过程，用770M的T5模型能够打败540B的PaLM模型。
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465), 通过few shot方式让LM在给定正确答案情况下生成rationale，如果不对就重试，搜集更多rationales之后就在上面把LM finetune一把，然后再继续few shot的过程以搜集更多rationales，重复这个过程。但是实验发现只能在简单的推理问题上有效果提升，在复杂的推理问题上效果比较差。
- [Distilling Reasoning Capabilities into Smaller Language Models](https://arxiv.org/abs/2212.00193), 提出Socratic CoT方法，通过few shot的方式让LLM先做问题分解，然后再分别解决子问题。同时训练两个小模型，一个用来做问题分解，另一个做子问题。
- [Learning Performance-Improving Code Edits](https://arxiv.org/abs/2302.07867), 开源了一个代码数据集https://pie4perf.com/, 里面的每一个代码都提供了由同一个用户编写的不同版本的代码，由时间排序，排在后面的代码版本更正确或者性能更优，也就是包含了trajectory of program。让LLM在这种代码轨迹上学习如何提升代码的质量。最终效果表明在CODEGEN上能够达到Codex相当的效果，而Codex要比CodeGEN模型大十倍。
- [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381), 主要目的是提高小模型解决数学问题的能力。几个主要的发现：数据format非常重要，比如在多位数加法中简单的把输出结果中的数字调换一下顺序就能获得很好的效果提升，另外直接在CoT数据上进行训练效果非常显著，哪怕模型是随机初始化的并未经过pretrain阶段。
- [Meet FreeWilly, Our Large And Mighty Instruction Fine-Tuned Models](https://stability.ai/blog/freewilly-large-instruction-fine-tuned-models?utm_source=twitter&utm_medium=website&utm_campaign=announcement), 借鉴Orca的做法，分别使用一个小模型和一个大模型生成了50万和10万总共是60万的中间过程数据（数据量是Orca的十分之一），在一些指标上（主要是AGIEval相关指标）和GPT3.5能够相当，但在ARC,MMLU等这些核心指标上还是比不过GPT3.5，相比GPT4就差的更远了。说明这种finetune的方法有一点作用，但是作用还是比较有限。不过相比Orca的结果，这里的结果又提升了一波，哪怕只使用了十分之一的数据量做过程监督的finetuning，这也说明无论是pretrain阶段还是sft阶段，数据的选择都至关重要。



  
## RL(or feedback)
### code
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780), 提出CodeRL框架，将生成代码的LM作为actor network，然后crictic network预测代码通过率并给出一个dense feedback(主要的反馈来源包括编译和运行结果或者单元测试结果等)，在inference阶段，模型也可以通过这个crictic network进行动态生成和实时的调整。模型结构使用基于encoder-decoder的CodeT5模型。但似乎2023年放出的CodeT5并没有使用CodeRL.
- [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898), 直言现在都是static instruction-to-code错误有可能传导。提出InterCode，类比强化学习的设定，将code当做action，execution feedback当做observation，主要在Bash和SQL任务上做了一些实验。但这个工作也主要是prompt engineer，并不涉及到模型的训练。
  
### problem-solving
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275), DeepMind的一篇工作，主要比较在GSM8K数学题上分别使用outcome-based supervised和process-based supervised两种方法，最后发现：在final-answer error rates上，这两种方法相差不大，process-based方法略低，并且使用强化学习都能够比只使用监督方法效果更好；而在trace error rates上（主要考察过程是否正确），需要process-based feedback，且需要reward model。总之：在最终结果的预测上，process based feedback作用不大，但如果要求trace error rates，则process based feedback作用则非常大。并且，无论是最终结果还是trace error rates，基于outcome或者process的reward model都很重要。这个结果后来被OpenAI的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 推翻了。
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), 使用process supervission方法解决了78.2%的MATH问题；使用active learning能够大大提升样本效率。公开了PRM800K的数据集。具体方法：在pretrain后的model上用MathMix（1.5B tokens）做finetune，这个模型叫做base model。然后再使用few shot形式在MATH数据集上让模型生成一些solutions，过滤那些能够得到正确答案的solutions，在这个新的带solutions的MATH数据集上对base model进行finetune后得到generator。最后再让这个generator对MATH数据集生成step by step solutions，然后交给人类去进行标注。最后搜集到了800k个 step-level的数据。（针对12k个math问题总共75k个solutions）。为了让数据标注效率更高，主要选择process reward model打分很高但是错误的solutions给到人工标注，并且每隔一段时间用新搜集到的部分数据重新训一遍PRM，再挑选标注数据。具体使用的方法与[Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275) 很相似。
- [REFINER: Reasoning Feedback on Intermediate Representations](https://arxiv.org/abs/2304.01904), 核心由两个模型组成：generator和critic model，首先由generator生成多个方案，然后交给critic model进行评价，给出一个structure feedback，将feedback输入给generator重新生成方案。为了训练critic model，根据不同任务类型将错误情况进行分类，最后通过rule-based方法构造了错误的数据以及对应的feedback，最后拿这些数据训练一个critic model（具体使用的是UnifiedQa-T5-base，训练目标是文字的feedback）。最大的贡献是提供了一个generator和critic model双模型的结构，并且critic model提供的是一个细粒度的反馈，并且能够联合起来进行训练。但是问题是，训练critic model过程中要对数据进行大量的标注和精心的设计。
- [PaD: Program-aided Distillation Specializes Large Models in Reasoning](https://arxiv.org/abs/2305.13888), 一般是用大模型做推理中间过程来蒸馏小模型，但是大模型的中间过程可能是错误的，这会导致小模型的性能损失。这篇paper提出用代码运行结果来保证中间过程的正确性。具体做法就是通过few shot方式给大模型一些用代码表示的中间推理过程，然后让大模型输出新问题的代码版本推理过程，根据这个生成代码的编译结果和运行结果，只保留正确的样本。最后再拿这些搜集到的样本去蒸馏小模型。核心贡献就是用代码的方式保证了中间推理过程的正确性，这里面最关键的是代码天然带有最准确的反馈信号。
  

## Prompt Engineer
- [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406), 提出Decomposed Prompting，核心思想是可以将复杂的prompt分解成sub prompt，太长的prompt可以分解为较短的prompt，并且每一个sub prompt还可以递归的进行分解。
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), 相比普通的decode策略，self-consistency则一次采样多个，然后选择一致性最高的答案，类似于多数投票。
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), LLM产生一个初始output之后，重新输入这个LLM中让它给出feedback，然后再基于这个feedback让LLM重新生成新的output，这个过程可以迭代下去。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 让模型输出reason thought，同时也让模型输出action（比如去搜索网络结果），也就是reason（Re） + action(Act)，合并为ReAct。实际的inference过程大致上分为：reason -> action -> observation, AutoGPT核心参考这篇工作。
- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091), 提出Plan-and-Solve Prompting，就是简单的将Zeroshot CoT中的“Let’s think step by step”替换成“Let’s first understand the problem and devise a plan to solve the problem. Then, let’s carry out the plan and solve the problem step by step”, 本质上是一个prompt engineer的工作。
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625), 提出least-to-most prompting方法，本质上就是分解的思想，整体分成两步：第一步，让模型先分解成子问题；第二步，按顺序依次解决子问题，后一个子问题会依赖于前一个子问题的答案。这个思路和ReAct尤其和Decomposed Prompting方法很类似。
- [Neuro-Symbolic Procedural Planning with Commonsense Prompting](https://arxiv.org/abs/2206.02928), 主要是扩展了Structural Causal Model，并且通过ConceptNet引入常识从而辅助提出更好的prompt，从而可以让LLM从goal生成step。基本方式：针对某个任务，先借助于一个外部的知识库进行检索得到这个任务相关的graph，然后通过检索得到的subgraph构造一个prompt，再把这个prompt输入到两个嵌套的LM中，最后输出可接受的step.
- [Language Models of Code are Few-Shot Commonsense Learners](https://arxiv.org/abs/2210.07128), 提出COCOGEN，主要研究structured commonsense reasoning问题，主要贡献是证明了code-LM比自然语言LM是更好的推理器，核心思路是把一个常识推理问题转化成代码生成问题，将一个推理graph用Python代码来表示，从而可以直接用Code-LM来解决。
- [Self-planning Code Generation with Large Language Model](https://arxiv.org/abs/2303.06689), 将代码生成问题通过两个阶段来实现，阶段一：通过few-shot的方式(示例就是intent-plan对)让LLM先给出plan，阶段二：通过intent和生成的plan，让模型再生成代码。最后发现在HumanEval和MBPP上比CoT效果还要好。不过这篇paper主要是prompt层面的方案，不涉及到模型参数改动。
- [Plan, Eliminate, and Track Language Models are Good Teachers for Embodied Agents](https://arxiv.org/abs/2305.02412), 主要用在emboddied agent上，设计了一个三层结构：plan模块主要使用LLM作为任务分解器将大任务分解多个子任务，eliminate模块则主要负责将agent看到的物品中与当前子任务无关的object mask掉，track模块则主要负责判断当前子任务是否得到了解决。说自己是第一个hierarchical planning with natural language and progress tracking.
- [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://arxiv.org/abs/2201.07207), 分成三步，第一步是让LLM对任务进行分解得到第一步，然后将第一步结果用一个translator将这个步骤转换为可以执行的步骤，第三步则再将这个被转化为更规范步骤的文本输入到LLM中让它生成下一步动作。这种做法相当于是把LLM作为一个action planner，但是LLM输出的plan step和能被接收的action空间还是有些差距的，需要经过中间的一个翻译器进行转换，完全可以做成end-to-end方式。
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366), 不用二分类或标量的反馈信号，而是用一个所谓"semantic gradient signal"，但是又不去更改模型的参数，只是把这些verbal language feedback用一个memory buffer缓存起来。主要由四个部分组成：actor，evaluator，self-reflextion和memory。actor每次产生一个方案trajectory（实际上是一系列action和observation的组合），这里的actor由ReAct来实现，然后交给evaluator进行评价，实际上就是评价当前这个方案是否成功等等，相当于reward作用。然后再把当前的trajectory和evaluator的输出一起输入self-reflection模块（是一个LLM）进行总结得到一个自然语言的描述，将它存到memory中。这个memory存储了每一轮的总结（paper里最多是存储3轮的结果），以供下一轮actor使用。最后的实验在decision making(AlfWorld)，coding（HumanEval）和language resoning（HotPotQA）任务上都有很好提升。本质上是一个prompt engineer工作，不涉及到模型参数更新。
- [Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models](https://arxiv.org/abs/2306.06891), 提出Recursion of Thought，核心思想是让模型先将问题分解成不同的子问题，然后在每一个子问题后面让模型生成一个[Think]标记，这会触发一个新的prompt进行求解得到答案后再回去把原来的[Think]标记替换成答案，所有子问题求解完成后生成[STOP]标记。论文里使用带标记的数据做了一些训练，idea很棒，但是只在数学计算题（加减乘除）这种问题上做了实验，核心原因还是训练数据太难获取了。
- [Measuring Faithfulness in Chain-of-Thought Reasoning](https://www-files.anthropic.com/production/files/measuring-faithfulness-in-chain-of-thought-reasoning.pdf), 主要研究LLM CoT的faithfulness问题，也就是模型最终答案和CoT中间过程的匹配程度，这对大模型的reasoning或planning的可解释与可追踪至关重要。分别使用四种方法进行测试：截取CoT中间过程的前面一部分（early answering），添加一些错误（adding mistakes），将中间过程做一些改写（Paraphrasing）和将中间过程用圆点代替（Filler tokens）。做了这些操作之后来观察和原来CoT给出答案的一致程度，如果不受到影响，说明模型的faithfulness就比较低，反之如果受到影响说明明显的faithfulness比较高。结果发现early answering和adding mistakes这两种设定下模型受到的影响比较大，而Paraphrasing和Filler tokens的实验结果则表名模型的效果不会因为Paraphrasing和Filler tokens带来效果提升。
- [Question Decomposition Improves the Faithfulness of Model-Generated Reasoning](https://www-files.anthropic.com/production/files/question-decomposition-improves-the-faithfulness-of-model-generated-reasoning.pdf), 主要是探索如何提升模型的faithfulness。首先区分了三个类型的CoT,最原始的CoT是直接让模型step-by-step的分析与推理并最终得到答案，而Chain-of-Thought Decomposition则要求模型先分解成子问题，然后针对每个子问题得到答案，最后总结所有子问题的答案得到最终答案；而Factored Decomposition也是把问题分解成子问题，不过是让模型在不同的context下生成每一个子问题的答案，而Chain-of-Thought Decomposition是在同一个context中一次生成所有子问题的答案，也就是Factored Decomposition是分开逐步生成的，每一个子问题是一个新的context，最后普通的CoT则连子问题分解都没有，只是要求模型step-by-step think。最后发现在任务的准确率上普通CoT > Chain-of-Thought Decomposition > Factored Decomposition，而faithfuleness则是相反。这个faithfulness指标就是通过[Measuring Faithfulness in Chain-of-Thought Reasoning](https://www-files.anthropic.com/production/files/measuring-faithfulness-in-chain-of-thought-reasoning.pdf) 这个工作中介绍的方法测量得到的，简单的说就是往CoT中要么只给前面一部分要么加入一些错误信息，观察模型是否会受到影响，如果不受到影响说明这个模型就ignored reasoning，也就是它的faithfulness就低。Question Decomposition这个工作主要是发现了模型效果与faithfulness是一个权衡折中的两方面。


## Tool
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761), 提出Toolformer，主要过程是先从语料中构造一个可以使用API call的语料，然后再用这个语料finetune模型。构造API call语料的过程主要分为三步：第一步是通过in context learning的方式让LLM采样一些潜在有API call的语料，也就是针对语料的一些部位生成一些候选的API call，第二步是实际执行这些call，第三步是做过滤，具体方式是观察给定这个response下是否能够减小LLM预测下一个token时候的loss，如果有助于则保留这个call继续下一个API call的尝试。语料构造完成之后，再使用这个带API call的语料对LLM进行finetune，API call的相关文本是直接插入到原有语料中的。inference阶段则当模型输出一个"->"符号则意味着要进行API call了，此时模型会停止继续预测，而是先进行API call拿到response后输入到模型中继续生成。paper里主要尝试了QA，search，calculator，translator和calender这些tool。这个paper主要还是数据构造的方式可以值得借鉴（但采样效率极低），但整个API call根本不是交互形式而是直接也按照token进行预测，这没有太多plan的思想在这里。
- [ART: Automatic multi-step reasoning and tool-use for large language models](https://arxiv.org/abs/2303.09014), 针对任务准备好一些示例叫做task library，针对一个新的task，从这个task library中找到一些示例形成demonstration，最后让LLM将这个task做分解并分布解决。这个工作的主要问题在于需要构建和维护一个task library.

## Other
- [Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought](https://arxiv.org/abs/2210.01240), 提出PRONTOQA（Proof and Ontology-Generated Question Answering）数据集，主要分成三个步骤：ontology generation, proof generation，然后生成一个自然语言样例，包括question，CoT和label。有了这个PrOntoQA数据集之后，作者拿InstructGPT和ChatGPT来做了测试，发现较大size的模型才能够做推理，但是在需要多跳上的推理数据上表现都很差。paper更多是提出一个数据集，并没有做任何模型训练。
- [Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners](https://arxiv.org/abs/2210.02969), 提出一个Flipped Learning方法，核心思想是给定input和label的情况下，让模型生成instruction，而普通的few shot方式则是instructio+input，让模型给出label。这里刚好是一个逆过程。实验结果表明，在zero shot设定下，flipped learning之后的模型效果是最好的，哪怕参数比GPT3小几十上百倍，但是依然比GPT3和PaLM的zero shot效果好很多，甚至比GPT3的3-shot效果还要好。
- [Integrating Action Knowledge and LLMs for Task Planning and Situation Handling in Open Worlds](https://arxiv.org/abs/2305.17590), 将大模型的能力与PDDL进行结合，主要是将一些常识能力引入到PDDL中。
- [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477), 提出LLM+P，先由大模型将问题转化为PDDL，然后使用domain planner来生成PDDL的解决方案，再由大模型转化为自然语言描述的解决方案。paper里说LLM善于做linguistic competence，但不擅长于做 functional competence。在LLM+P里，LLM只是做为一个翻译器，完成自然语言和PDDL之间的转换，真正完成决策功能的都是由中间的planner，PDDL planner由domain file和problem file两个文件组成，这个文章都假定domain file已经是现成的了，而这才是整个plan问题的核心，需要大量人工的介入。怎么样自动生成domain pddl以及能否让LLM闭环完成，这个问题更为重要。
- [Leveraging Large Language Models to Generate Answer Set Programs](https://arxiv.org/abs/2307.07699), 利用LLM将问题转化为Answer Set Program（ASP）.
- [AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers](https://arxiv.org/abs/2306.06531), 将LLM当作一个翻译器，同样也是把task翻译成TAMP语言，然后再通过TAMP系统做规划。
- [Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2301.13379), 提出Faithful CoT，基本思路是分成两个阶段：translate和solve，translate阶段主要是用LLM把自然语言描述的问题转化为python/datalog或者PDDL，第二阶段solve则使用一个确定性的系统来解决问题。主要研究了四类问题：数学题，Multi-hop QA, planning和Logical inference. 这个工作比LLM+P研究的范围更大一些，但本质是一样的，将LLM当做一个确定性系统的翻译器，真正完成计算或者执行的都是交由一个确定性系统来进行，比如Python/Datalog Interpreter或者PDDL Planner等等。
- [Leveraging Large Language Models (LLMs) for Process Mining](https://arxiv.org/abs/2307.12701), 将LLM应用在Process Mining(过程挖掘)中，比如从event log中挖掘出过程，形成过程数据。这在RPA等这种过程类的场景有很大的作用。


## Survey
- [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403), 相关论文放在了https://github.com/jeffhj/LM-reasoning. 将推理主要分为演绎推理，归纳推理和归因推理，以及区分为formal和informal，paper中主要聚焦在informal deductive reasoning问题，但整个paper主要关注finetune和prompt侧，没有提到pretrain阶段的工作。
- [Natural Language Reasoning, A Survey](https://arxiv.org/abs/2303.14725), paper作者更看好defeasible reasoning，和deductive reasoning相比，这种推理方式只会得出可能的结论，也就是不保证推理结果的严格正确性。


## To be read
- [TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs](https://arxiv.org/abs/2303.16434),
- [ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning](https://arxiv.org/abs/2212.07919),
- [MathPrompter: Mathematical Reasoning using Large Language Models](https://arxiv.org/abs/2303.05398),
- [Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance](https://arxiv.org/abs/2305.17306),
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688), 
- [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1),
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311),
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374), 提出CodeX,
- [Self-Supervised Learning of Pretext-Invariant Representations](https://arxiv.org/abs/1912.01991), 
- [Teaching Small Language Models to Reason](https://arxiv.org/abs/2212.08410), 
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992), 
- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691),
- [SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks](https://arxiv.org/abs/2305.17390),
- [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388), 
- [Iterated Decomposition: Improving Science Q&A by Supervising Reasoning Processes](https://arxiv.org/abs/2301.01751), 
- [Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning](https://arxiv.org/abs/2205.09712), 
- [Causal Reasoning and Large Language Models: Opening a New Frontier for Causality](https://arxiv.org/abs/2305.00050), 
- [Automatic Generation of Socratic Subquestions for Teaching Math Word Problems](https://arxiv.org/abs/2211.12835), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [](), 
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), 发现用简单的文本训练非常小的模型（10m参数）也能够获得较好的效果（包括生成连贯流利的stories，还有推理和instruction following能力）。提出TinyStories数据集，这个是用GPT3.5和GPT4生成的，使用的词汇都是3-4岁阶段能够明白的简单词汇。构造方式就是从1500个基本词汇中随机挑选3个词让LLM生成包含这3个词的story。主要问题是这个工作用GPT4生成训练集，又用GPT4来做评判。并且没有在公开评测集上做过评测。
- [Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://arxiv.org/abs/2108.04106), 
- [MetaICL: Learning to Learn In Context](https://arxiv.org/abs/2110.15943),



# related topics
- graph based learning
- manifold learning
- structural learning

# Evaluation

## World Knowledge
- TriviaQA，阅读理解

## Subject
- MMLU, 包含57个各类教育学科与职业学科的数据集
- CEval
- AGIEval

## Commonsense Reasoning
- PIQA (Bisk et al., 2020)
- SIQA (Sap et al., 2019),
- OpenBookQA (Mihaylov et al., 2018)
- CommonsenseQA
- StrategyQA
- HellaSwag, 常识NLI数据集，让模型补全句子，一般把它归入语言能力
- [WinoGrande](https://huggingface.co/datasets/winogrande), 常识推理数据集，给出一个句子并且挖掉一个空让模型从给定选项里选择正确词，指代消解问题。需要进行推理。
CommonsenseQA / StrategyQA / ARC / BoolQ / HotpotQA / OpenBookQA / PIQA
  
## Symbolic Reasoning
CoinFlip / LastLetterConcatenation / ReverseList

## Logical Reasoning
ReClor / LogiQA / ProofWriter

## Math Reasoning
- GSM-8K，高中数学题
GSM8K / SVAMP / ASDiv / AQuA / MAWPS / AddSub / MultiArith / SingleEq / SingleOp / Lila

## Coding
- [HumanEval](https://github.com/openai/human-eval), 代码数据集，
- MBPP

## Real World
- Alfworkd
- HotPotQA

## other
- AI2 Reasoning Challenge（ARC），包含grade3到grade9年级的科学数据集，包括easy和challenge两档，challenge需要推理。GTP4只用了challenge-set
- DROP, Discrete Reasoning Over Paragraphs
- QuALITY, 长内容上的QA
- RACE-H，阅读理解
- [BigBench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), 
- XCOPA
- CRASS, The Counterfactual Reasoning Assessment  
- COIG, https://huggingface.co/datasets/BAAI/COIG, Chinese Open Instruction Generalist
- OpenOrca, https://huggingface.co/datasets/Open-Orca/OpenOrca
- COPA, https://huggingface.co/datasets/vietgpt/copa_en
ALERT / CONDAQA / SCAN / WikiWhy


# paper list
- [Reasoning in Large Language Models](https://github.com/atfortes/LLM-Reasoning-Papers),
- [Deep-Reasoning-Papers](https://github.com/floodsung/Deep-Reasoning-Papers), 都是4年前的工作了
- [ReasoningNLP](https://github.com/FreedomIntelligence/ReasoningNLP), 对应的paper是[Natural Language Reasoning, A Survey](https://arxiv.org/abs/2303.14725), 除了paper之外，对数据集也做了很多梳理
- [](), 
- [](), 
- [](), 
- [](), 
- [Chain-of-ThoughtsPapers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers), 
- [LLM Tool Use Papers](https://github.com/xlang-ai/llm-tool-use), 
- [EnvInteractive and Decision Making(personnal usage)](https://github.com/Timothyxxx/EnvInteractiveLMPapers), 

# Other
- https://github.com/jeffhj/LM-reasoning
- https://nl-reasoning-workshop.github.io/， ACL2023 Reasoning topic
- [Teach Language Models to Reason](https://dennyzhou.github.io/Teach-LLMs-to-Reason-7-2023.pdf), from Denny Zhou, 
- [复杂推理：大语言模型的北极星能力](https://yaofu.notion.site/6dafe3f8d11445ca9dcf8a2ca1c5b199), pretrain/finetune/RL/prompt阶段都有办法提高推理能力。
- [ACL 2023 Tutorial: Complex Reasoning in Natural Language](https://wenting-zhao.github.io/complex-reasoning-tutorial/), 
- [LLM Tool Use Papers](https://github.com/xlang-ai/llm-tool-use), 有关tool use paper的github repo
- [Planning.Wiki - The AI Planning & PDDL Wiki](https://planning.wiki/), 
- [Scaling, emergence, and reasoning in large language models
](https://docs.google.com/presentation/d/1EUV7W7X_w0BDrscDhPg7lMGzJCkeaPkGCJ3bN8dluXc/edit?pli=1&resourcekey=0-7Nz5A7y8JozyVrnDtcEKJA#slide=id.g1fc34b3ac18_0_27) from Jason Wei


