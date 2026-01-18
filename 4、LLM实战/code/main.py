"""
该文件用于示范如何使用hugging face完成某个LLM的预训练，但是没有考虑：
1、多卡训练
2、显存开销问题
3、日志记录
4、checkpoint记录和恢复
5、模型保存
"""

import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


from transformers import AutoConfig, AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
from transformers import TrainingArguments
from transformers import Trainer, default_data_collator
# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import IterableDataset
import torch

def down_load_model(model_name: str, local_dir: str):
    print(os.getcwd())
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

    if model_name:
        down_load_instrument = 'huggingface-cli download --resume-download ' + model_name + '--local-dir ' + local_dir
        os.system(down_load_instrument)

def check_llm_config(model_name_or_path:str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    # print(config)
    return config

def qwenTest(model:AutoModel, tokenizer:AutoTokenizer):
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )


    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


# 这里我们取块长为 2048
block_size = 2048

def group_texts(examples):
    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按 block_size 进行切分
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # CLM 任务，labels 和 input 是相同的
    result["labels"] = result["input_ids"].copy()
    return result

def deal_dataset(tokenizer:AutoTokenizer):
    ds = load_dataset('json', data_files='./pretrain_dataset_small.jsonl')
    # 尝试打印一下看看数据结构是什么
    # print(ds['train'][0])
    column_names = list(ds["train"].features)

    def tokenize_function(text_list):
        # 使用预先加载的 tokenizer 进行分词
        output = tokenizer([item for item in text_list["text"]])
        return output

    # 批量处理
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        num_proc=10,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # 批量处理
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=10,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size=40000,
    )

    train_dataset = lm_datasets["train"]
    return train_dataset

class IterableDatasetWrapper(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)

def train_model(model, train_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="output",  # 训练参数输出路径
        per_device_train_batch_size=4,  # 训练的 batch_size
        gradient_accumulation_steps=4,  # 梯度累计步数，实际 bs = 设置的 bs * 累计步数
        logging_steps=10,  # 打印 loss 的步数间隔
        num_train_epochs=1,  # 训练的 epoch 数
        save_steps=100,  # 保存模型参数的步数间隔
        learning_rate=1e-4,  # 学习率
        gradient_checkpointing=True  # 开启梯度检查点
    )

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IterableDatasetWrapper(train_dataset),
        eval_dataset=None,
        tokenizer=tokenizer,
        # 默认为 MLM 的 collator，使用 CLM 的 collater
        data_collator=default_data_collator
    )

    trainer.train()


if __name__ == '__main__':
    #  step 1: 下载对应的模型，具体可以根据hugging face的官网获取对应的模型
    # down_load_model('Qwen/Qwen3-0.6B', 'qwen3')

    #  step 2： 使用命令查看模型的信息（model_path为前面下载模型时使用的local_dir)
    # config = check_llm_config('qwen3')

    # step 3: 加载对应的模型
    model = AutoModelForCausalLM.from_pretrained('qwen3',trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)

    # step 4: 加载一个tokenizer，可以使用qwen3自带的
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

    # 测试是否加载成功
    # qwenTest(model, tokenizer)

    # step 5: 加载预训练数据
    train_dataset = deal_dataset(tokenizer)

    # step 6: 使用trainer进行训练
    train_model(model, train_dataset, tokenizer)