# Language
- [LAMBADA](https://huggingface.co/datasets/lambada), 预测最后一个词。要求模型只能通过全局信息而非最后一句信息才能预测对。
- [HellaSwag](https://huggingface.co/datasets/hellaswag), [paper](https://arxiv.org/abs/1905.07830), commonsense NLI, 从字幕数据和WikiHow数据中过滤筛选得到，每一个问题后面跟上四个句子，让模型猜测正确的句子。因为使用的是视频字幕和WikiHow数据，因此需要模型具备一定的常识推理能力，知道在上文的背景下，后文正确的场景应该是怎样的。
- [StoryCloze](https://huggingface.co/datasets/story_cloze), 给定长度为四句话的一个story，让模型从两个候选句子中选择一个作为结尾。这也要求模型具备上下文信息的推理能力。

# Understanding
- [clue_wsc](https://github.com/CLUEbenchmark/CLUEWSC2020), 中文指代消解任务，给定一个句子，然后给出一个代词和一个名词，让模型确认他们是否是指代关系。这个任务很容易让大模型一边倒，要么全输出是要么全输出否。
- []

# Reading Comprehension/Question Answering
- [BoolQ](https://github.com/google-research-datasets/boolean-questions), 15942个 (question, passage, answer)三元组，让模型在给定passage下回答question，只用回答yes/no。被纳入SuperGLUE评估中。因为答案的主要信息在给定的passage中包含，所以更类似为阅读理解类的评估集，但因为许多问题需要进行一些常识推理，因此也有分类方法把BoolQ放到Commonsense Reasoning分类中去。比如给定的passage中的信息是某个电影在White House拍摄，问题则问是否在Washington拍摄，这就要求模型必须知道White House属于Washington这样的常识。

# World Knowledge

# Commonsense Reasoning
- [PIQA](https://huggingface.co/datasets/piqa), Physical Interaction Question Answering. 需要一些真实世界的物理常识来回答问题。比如"how to taste something?"，然后给两个候选让模型选择。
- [SIQA](https://huggingface.co/datasets/social_i_qa), Social Interaction Question Answering. 社会场景下的QA，给定一个context和三个候选答案，让模型回答问题。问题比如Sydney身边走过一个向他乞讨的老人，但他身上没有零钱了他赶到很难过。问题是"How do you describe Sydney?"给定的三个候选答案分别是"sympathetic", "like a person who was unable to help"和"incredulous"。
- [Winogrande](https://huggingface.co/datasets/winogrande), 以填空的形式要求模型从两个候选中选择一个正确答案填空。比如context是"John moved the couch from the garage to the backyard to create space. The _ is small."，让模型从"garage"和"backyard"两个中进行选择。这里因为是把沙发从车库中搬到后院里去来腾空间，因此正确答案应该是garage，正是因为garage小才需要腾空间的。因此这里用到了常识推理。是一种升级版本的词义消歧任务。
- [ARC](https://huggingface.co/datasets/ai2_arc), AI2 Reasoning Challenge. 包含Easy和Challenge两个版本。给定context和question: "One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?" 要求模型从["Shady areas increased.", "Food sources increased.", "Oxygen levels increased.", "Available water increased."]这四个选项中选择正确答案，这里面涉及到推理。context里先是提到了橡树结了更多橡果以及对应的花栗鼠也增加了，问为什么花栗鼠会增加。这需要模型能够推理得到是因为花栗鼠的食物增加的缘故。
- [OpenbookQA](https://huggingface.co/datasets/openbookqa), 给定small book中的fact，要求模型基于这个fact以及额外的常识，推理得到答案，然后从四个候选答案中选择正确答案。比如{'id': '7-980', 比如'fact1': 'the sun is the source of energy for physical cycles on Earth', 问题和候选项为'question_stem': 'The sun is responsible for', ['puppies learning new tricks','children growing up and getting old','flowers wilting in a vase','plants sprouting, blooming and wilting']。
- [LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning](https://arxiv.org/abs/2007.08124), 给定一个passage然后让模型回答问题（从四个候选中选择正确答案）。比如passage是"工厂有多个宿舍区和工区，A宿舍区住的工人都不是纺织工人，现在有个结论是有些在B工区工作的工人没有住在A宿舍区，请问下列选项中哪一个是缺失的前提。"这个问题是由结论推导可能缺失的前提，难度比较大，需要很强的推理能力。给定的四个选项是："有些纺织工人没有在B区工作"，"在B区工作的有些工人不是纺织工人"，"有些纺织工人在B区工作"，"在A宿舍区住的有些工人在B区工作"，正确答案应该是第三个。这种推理题难度特别大，感觉人类也不一定做得对。

# Math Reasoning

# Coding
- HumanEval
- MBPP

# Translation

# Generation

# Exam Test

# Proficiency Test
- [hsk](http://yuyanziyuan.blcu.edu.cn/info/1043/1501.htm), 针对母语非汉语者的汉语言水平考试。

# Safety


# Leaderboard
- [SuperGLUE](https://super.gluebenchmark.com/leaderboard), Natural Language Understanding