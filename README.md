# LLM_Learning_note

这是一个关于大语言模型（LLM）和Transformer相关知识的学习笔记仓库，用于记录和总结学习过程中的重点内容。

## 项目结构

```
├── 1、Transformer基础知识/     # Transformer架构学习资料
│   ├── 1、attention及其变体.md    # Attention机制及其变体详解
│   ├── 2、Encoder-Decoder.md      # Encoder-Decoder架构介绍
│   └── 3、预训练语言模型.md        # 预训练语言模型原理与应用
├── 2、LLM/                    # 大语言模型相关学习笔记
│   └── LLM.md                # LLM核心知识总结
├── 3、手动实现LLM2/           # LLaMA2模型手动实现代码
│   ├── Attention.py          # 注意力机制实现
│   ├── LLaMA2_Decoder_layer.py # LLaMA2解码器层实现
│   ├── MLP.py                # 多层感知机实现
│   ├── ModelConfig.py        # 模型配置
│   ├── RMSNorm.py            # RMSNorm归一化实现
│   ├── Transformer.py        # Transformer模型实现
│   ├── repeat_kv.py          # KV重复函数
│   └── rotary_pos_emb.py     # 旋转位置编码实现
├── 4、LLM实战/                # LLM训练实战资料
│   ├── code/                 # 实战代码
│   │   ├── ds_config_zero2.json  # DeepSpeed配置文件
│   │   ├── main.py           # 训练主程序
│   │   ├── main_deepspeed_version.py  # DeepSpeed版本训练程序
│   │   ├── main_deepspeed_version_bash.sh  # DeepSpeed训练脚本
│   │   ├── pretrain_dataset_small.jsonl  # 小型预训练数据集
│   │   └── process_dataset.py  # 数据集处理脚本
│   └── LLM训练实战.md         # LLM训练实战笔记
└── 5、RAG介绍/                # 检索增强生成(RAG)相关知识
    └── RAG.md                # RAG核心知识总结
```

## 内容概述

### Transformer基础知识
1. **Attention及其变体**：详细介绍了注意力机制的基本原理、各种变体（如Self-Attention、Multi-Head Attention等）及其在NLP中的应用。
2. **Encoder-Decoder**：讲解了经典的Encoder-Decoder架构，以及Transformer如何在此基础上进行改进。
3. **预训练语言模型**：探讨了预训练语言模型的发展历程、主要技术（如BERT、GPT等）及其应用场景。

### LLM
- 深入学习大语言模型的核心原理、训练方法、应用场景和最新研究进展。

### 手动实现LLM2
- 基于Python手动实现LLaMA2模型的核心组件，包括：
  - 注意力机制（Attention）
  - 解码器层（Decoder Layer）
  - 多层感知机（MLP）
  - RMSNorm归一化
  - 旋转位置编码（Rotary Position Embedding）
  - 完整的Transformer模型结构

### LLM实战
- LLM训练实战笔记，涵盖：
  - 数据集处理方法
  - 模型训练流程
  - DeepSpeed分布式训练配置
  - 训练脚本编写

### RAG介绍
- 检索增强生成（Retrieval-Augmented Generation）的核心概念、架构和应用场景
- RAG在大语言模型中的作用和优势
- RAG系统的构建和优化方法

## 学习计划

1. 掌握Transformer架构的基本原理和核心组件
2. 深入理解各种Attention机制的工作原理
3. 学习预训练语言模型的发展和应用
4. 研究大语言模型的最新技术和挑战
5. 手动实现LLaMA2模型的核心组件，深入理解模型内部工作机制
6. 实践LLM训练流程，掌握数据集处理、模型训练和分布式训练技术
7. 学习RAG技术，掌握检索增强生成的原理和应用

## 参考资料

学习过程中参考了大量论文、教程和技术博客，主要包括：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer架构的奠基性论文

- [万字长文带你梳理Llama开源家族：从Llama-1到Llama-3](https://mp.weixin.qq.com/s/5_VnzP3JmOB0D5geV5HRFg) - 张帆, 陈安东
- [An Intuition for Attention](https://jaykmody.com/blog/attention-intuition/) - Jay Mody
- [细节拉满，全网最详细的Transformer介绍（含大量插图）](https://zhuanlan.zhihu.com/p/644122223) - 知乎

### 开源项目与平台
- [Hugging Face](https://huggingface.co/) - 提供预训练模型、数据集和NLP工具的开源平台
- [Happy-LLM](https://github.com/Happy-LLM/Happy-LLM) - 一个开源的大语言模型学习和实践项目


## 说明

本仓库仅用于个人学习和知识整理，如有错误或不足之处，欢迎指正。