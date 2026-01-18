# 简介
在实际的开发中，手动实现LLM的Pretrain、SFT全流程是非常繁琐且浪费时间的，同时也无法保证和原文的效果能保持一致。因此，在实际开发中，我们多采用训练框架进行开发，接下来将介绍使用hugging face的Transformers框架，结合deepspeed、高效微调框架preft，进行模型的pretrain、SFT全流程。

# 模型训练
使用Hugging Face的Transformer框架完成LLM的训练过程。

## 初始化LLM
我们可以使用 transformers 的 AutoModel 类来直接初始化已经实现好的模型。对于任意预训练模型，其参数中都包含有模型的配置信息。如果是想要从头训练一个 LLM，可以使用一个已有的模型架构来直接初始化。

```python
# 加载定义好的模型参数-此处以 Qwen-2.5-1.5B 为例
# 使用 transforemrs 的 Config 类进行加载
from transformers import AutoConfig

# 下载参数的本地路径
model_path = "qwen-1.5b"
config = AutoConfig.from_pretrained(model_name_or_path)

```

查看该 model，下图可以看到其架构和定义的配置文件相同:

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767704288439-36d14fdc-4cbf-4c99-be43-b4c3fb7a744b.png)

该 model 就是一个从零初始化的 Qwen-2.5-1.5B 模型了。一般情况下，我们很少从零初始化 LLM 进行预训练，较多的做法是加载一个预训练好的 LLM 权重，在自己的语料上进行后训练。以下是通过加载已经训练好的模型参数进行初始化：

```python
from transformers import AutoModelForCausalLM

# model_name_or_path 即为下载好的参数的本地路径
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)

```

有了模型，我们还需要初始化一个 tokenizer。此处，我们直接使用 Qwen-2.5-1.5B 对应的 tokenizer 参数即可。

```python
# 加载一个预训练好的 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

```

## 预训练数据集处理
预训练数据集是用于pretrain阶段进行casual Language model learning时使用的。此处使用开源数据集进行训练。此处使用，出门问问序列猴子开源数据集作为预训练数据集。

```python
# 加载预训练数据
from datasets import load_dataset

ds = load_dataset('json', data_files='/mobvoi_seq_monkey_general_open_corpus.jsonl')

```

对于预训练而言，模型的目标是学习目标是casual language model，只需要模型预测下一个token即可，因此一次把多个文本拼在一起，然后分块进行训练。该方法并不会影响模型的训练。但是分块的大小会影响模型训练时的显存开销。（也即上下文限制）。

## 使用Trainer训练
首先我们需要配置训练的超参数，使用 TrainingArguments 类来实例化一个参数对象：

```python
from transformers import TrainingArguments
# 配置训练参数

training_args = TrainingArguments(
    output_dir="output", # 训练参数输出路径
    per_device_train_batch_size=4, # 训练的 batch_size
    gradient_accumulation_steps=4, # 梯度累计步数，实际 bs = 设置的 bs * 累计步数
    logging_steps=10, # 打印 loss 的步数间隔
    num_train_epochs=1, # 训练的 epoch 数
    save_steps=100,  # 保存模型参数的步数间隔
    learning_rate=1e-4, # 学习率
    gradient_checkpointing=True # 开启梯度检查点
)

```

然后基于初始化的 model、tokenzier 和 training_args，并传入处理好的训练数据集，实例化一个 trainer 对象:

```python
from transformers import Trainer, default_data_collator
from torchdata.datapipes.iter import IterableWrapper

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    eval_dataset= None,
    tokenizer=tokenizer,
    # 默认为 MLM 的 collator，使用 CLM 的 collater
    data_collator=default_data_collator
)

```

再使用 train 方法，即会按照配置好的训练超参进行训练和保存：

```python
trainer.train()
```

# Deepspeed初体验
安装deepspeed框架，然后使用deepspeed指令进行运行即可。以下是bash文件示例：

```python
# 设置可见显卡
CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name autodl-tmp/qwen-1.5b \
    --tokenizer_name autodl-tmp/qwen-1.5b \
    --train_files autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus_small.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir autodl-tmp/output/pretrain \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir autodl-tmp/output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \

```



## DeepSpeed框架详解
DeepSpeed主要通过一个JSON配置文件传入参数。如下：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### ZoRO配置（最重要）对应zero_optimization下key
#### stage
DeepSpeed拥有4个stage的选择（0/1/2/3）。

| stage | 分片内容 | 适用场景 |
| --- | --- | --- |
| 0 | 无分片（等同于普通训练） | 小模型 |
| 1 | 分片 optimizer states | 中等模型 |
| 2 | 分片 optimizer + gradients | 大模型 |
| 3 | 分片 optimizer + gradients + parameters（并支持 partitioned model） | 超大模型 |


#### stage3_param_persistence_threshold
该参数针对stage3，控制参数是否持久化到GPU内存中。

作用：减少参数在GPU之间移动，降低通信开销。

#### contiguous_gradients
梯度是否连续存储。

作用：减少fragmentation，提高空间利用率。

#### reduce_bucket_size
梯度 reduce 的 bucket 大小。

作用：影响通信与显存峰值

#### overlap_comm
是否 overlap 通信与计算

作用：提高训练吞吐，但对网络/带宽敏感

#### allgather_bucket_size
allgather 的 bucket 大小

作用：影响通信效率与显存峰值

### 训练/优化器相关
#### train_batch_size
就是batch_size大小。

计算：train_batch_size = 每张卡训练的batch_size * gradient_accumulation_steps * world_size

#### gradient_accumulation_steps
梯度累计步数

作用：多次累加得到最终的batch

#### optimizer
优化器

### 精度
#### `fp16` 或 `bf16`
半精度训练，两者之间的差别在于使用同样的bit数，表达的小数范围不同。

### activation_checkpointing
开启activation checkpoint

作用：减少activation显存，但是会增加计算

### gradient_clipping
梯度裁剪

作用：稳定训练

### steps_per_print
日志打印间隔

## 显存计算
我们用参数量 `P` 表示模型参数数（单位：bytes），用 GPU 数量 `N` 表示并行 GPU 数。

> 这里忽略 activation，因为 activation 与 batch/seq_len/hidden 相关，通常是显存最大头。
>

### Stage 0
普通训练（不做任何优化）

| 内容 | 大小 |
| --- | --- |
| model parameters | P |
| gradients | P |
| optimizer states | 2P（AdamW） |
| activation | A |


**总显存**：

```plain
mem_stage0 = P + P + 2P + A = 4P + A
```

### Stage 1
将opitmizer state进行分片，即将优化器参数分片，分别存储到N张卡上。

```plain
optimizer_per_gpu = 2P / N
```

**每卡显存**:

```plain
mem_stage1 = P + P + (2P/N) + A
         = 2P + 2P/N + A
```

### stage 2
分片：optimizer + gradient。分别存储到N张卡上。

```plain
grad_per_gpu = P / N
optimizer_per_gpu = 2P / N
```

**每卡显存**:

```plain
mem_stage2 = P + (P/N) + (2P/N) + A
         = P + 3P/N + A
```

### stage 3
分片：optimizer + gradient + model parameters。分别存储到N张卡上

```plain
grad_per_gpu = P / N
optimizer_per_gpu = 2P / N
param_per_gpu = P / N
```

**每卡显存**:

```plain
mem_stage3 = (P/N) + (P/N) + (2P/N) + A
         = 4P/N + A
```

## Activation Checkpointa
Activation：forward计算的中间激活值。主要用于反向传播时进行计算。

因为在反向传播的时候必须用到：

+ 每一层的输入
+ 每一层的输出（activation）

因此，默认情况下，pytorch会吧所有的activation保存到显存，直到backward计算完成。

显存只与每张卡上的batch_size、seq_len、hidden_dim、模型结构有关。

**每卡显存：**

```plain
activation ≈ batch_size * seq_len * hidden_dim * bytes_per_token * factor
```

其中：

+ hidden_dim 是模型隐藏维度（Qwen 3 0.8B 约 4096）
+ bytes_per_token：bf16/ fp16 = 2 bytes
+ factor：网络结构相关（约 1.5 ~ 2）

当 seq_len 2048、batch 16 时，activation 会非常巨大。

### 开启activation checkpoint原理
Activation checkpointing 的做法是：

+ **不保存所有 activation**
+ 只保存少量 checkpoint（例如每 N 层）
+ backward 时需要时再重新计算 forward（recompute）

因此，显存开销会变小，但是计算量会变大。



# 有监督微调（SFT）
SFT和Pretrain的区别主要是数据集的问题，SFT目标是让模型具有指令遵循的能力，即问什么，答什么的能力，防止她答非所问。



# 高效微调
前文针对模型的训练，均默认我们存在无限的显存，可以将模型完全放入到显存中进行训练。但是，实际上，并非每个人都有如此庞大计算资源，因此如何在有限的资源下对LLM进行微调是至关重要的。

## Adapt Tuning
即在模型中添加 Adapter 层，在微调时冻结原参数，仅更新 Adapter 层。

具体而言，其在预训练模型每层中插入用于下游任务的参数，即 Adapter 模块，在微调时冻结模型主体，仅训练特定于任务的参数：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767709180199-cb5d61d0-b48c-406a-adc1-92dddcaf2e68.png)

Q：如何理解加入adpater进行微调呢？？

A：我们认为attention层用于捕捉token间的关系，这种关系在pretrain阶段学习好就不需要再更改了。Feed-forward才是真正的知识层，因此我们在Feed-forward后面加一个adapter层，让adapter去适应我们真正的任务即可。极大减小了模型的训练参数。

### 缺点
由于增加了参数和计算量，因此导致微调后模型的计算速度较原预训练模型更慢。

## Prefex Tuning
前缀微调。该种方法固定预训练 LM，为 LM 添加可训练，任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小。具体而言，在每一个输入 token 前构造一段与下游任务相关的 virtual tokens 作为 prefix，在微调时只更新 prefix 部分的参数，而其他参数冻结不变。

### 缺点
占用了token数，使得可用序列长度变短。

# LoRA微调
通过矩阵分解的方式，将模型中的权重参数进行拆解从而达到微调的效果。

其他讲解，说的是针对下游任务，构建分解矩阵，从而可以快速适配不同的下游任务。但我的理解就是矩阵分解。。

## 优点
+ 针对不同的下游任务，构建LoRA模块，从而共享pretrain LLM的参数，有效切换下游任务；
+ LoRA使用自适应优化器，不需要计算梯度或者维护大多数参数的优化器状态，训练更高效，门槛更低；
+ 使用的线性设计，在部署的时候可以讲LoRA参数和pretrain LLM的参数合并，不存在推理延迟。
+ 可以组合其他微调参数。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767744811174-1d4e37b1-7ed6-48ec-b3ee-95e76f907224.png)

<font style="color:#262626;">LoRA示意图</font>

## <font style="color:#262626;">应用到LLM</font>
对于Transformer架构，LoRA技术主要用于模块的4个权重矩阵：$ W_q、W_k、W_v、W_o $,而冻结MLP权重矩阵。消融实验发现同时调整$ W_q，W_v $效果最佳。

因此，微调阶段可训练的参数为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767745044916-957f7547-2146-4b9c-9c5d-f8cbe018b963.png)

其中，$ L_{LoRA} $为LoRA的数量，$ d_{m} $为transformer输入输出纬度，$ r $为设定的$ LoRA $的秩。

## 代码实现
peft 库是 huggingface 开发的第三方库，其中封装了包括 LoRA、Adapt Tuning、P-tuning 等多种高效微调方法，可以基于此便捷地实现模型的 LoRA 微调。

### 流程
LoRA微调流程如下：

+ 确定使用LoRA的层。
+ 将其替换为LoRA层。（即给该层原来结果上增加一个旁路，通过矩阵分解，来模拟参数更新）
+ 冻结原来参数，进行微调，更新LoRA参数。

### 确定LoRA层
`target_modules` 一般是一个字符串列表，每一个字符串是需要进行 LoRA 的层名称，例如

```python
target_modules = ["q_proj","v_proj"]
```

在创建 LoRA 模型时，会获取该参数，然后在原模型中找到对应的层，该操作主要通过使用 re 对层名进行正则匹配实现：

```python
# 找到模型的各个组件中，名字里带"q_proj"，"v_proj"的
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# 这里的 key，是模型的组件名
```

### 替换原有的LoRA层
LoRA 层在具体实现上，是定义了一个基于 Lora 基类的 Linear 类，该类同时继承了 nn.Linear 和 LoraLayer。LoraLayer 即是 Lora 基类，其主要构造了 LoRA 的各种超参：

```python
class LoraLayer:
    def __init__(
        self,
        r: int, # LoRA 的秩
        lora_alpha: int, # 归一化参数
        lora_dropout: float, # LoRA 层的 dropout 比例
        merge_weights: bool, # eval 模式中，是否将 LoRA 矩阵的值加到原权重矩阵上
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

```

nn.Linear 就是 Pytorch 的线性层实现。Linear 类就是具体的 LoRA 层，其主要实现如下：

```python
class Linear(nn.Linear, LoraLayer):
    # LoRA 层
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # 继承两个基类的构造函数
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # 参数矩阵 A
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # 参数矩阵 B
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # 归一化系数
            self.scaling = self.lora_alpha / self.r
            # 冻结原参数，仅更新 A 和 B
            self.weight.requires_grad = False
        # 初始化 A 和 B
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

```

替换时，直接将原层的 weight 和 bias 复制给新的 LoRA 层，再将新的 LoRA 层分配到指定设备即可。

### 训练
实现了 LoRA 层的替换后，进行微调训练即可。由于在 LoRA 层中已冻结原参数，在训练中只有 A 和 B 的参数会被更新，从而实现了高效微调。

由于采用了 LoRA 方式，forward 函数也会对应调整：

```python
def forward(self, x: torch.Tensor):
    if self.disable_adapters:
        if self.r > 0 and self.merged:
            self.weight.data -= (
                transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
            )
            self.merged = False

        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    '''主要分支'''
    elif self.r > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result
    else:
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

```

## 通过Peft框架实现
首先加载所需使用库：

```python
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
```

其次加载原模型与原 tokenizer，此处和第二节一致：

```python
# 加载基座模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
```

接着，设定 peft 参数:

```python
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
```

然后获取 LoRA 模型：

```python
model = get_peft_model(model, peft_config)
```

最后使用 transformers 提供的 Trainer 进行训练即可，训练占用的显存就会有大幅度的降低：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)
trainer.train()

```

