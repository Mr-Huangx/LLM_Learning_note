import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from datasets import load_dataset, load_from_disk

DATA_PATH = "./data/clinc_oos_small"

if os.path.exists(DATA_PATH):
    dataset = load_from_disk(DATA_PATH)
else:
    dataset = load_dataset("clinc_oos", "small")
    dataset.save_to_disk(DATA_PATH)

intent_id2label = dataset["train"].features["intent"].names

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model



model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

def format_example(example):
    intent_str = intent_id2label[example["intent"]]
    return {
        "text": f"""### Instruction:
You are an intent classification system.
Classify the user's intent into one of the predefined intent labels：{intent_id2label}

### User:
{example['text']}

### Response:
{intent_str}{tokenizer.eos_token}
"""


    }

for split, ds in dataset.items():
    print(split, len(ds))

train_dataset = dataset["train"].map(format_example, remove_columns=dataset["train"].column_names)
val_dataset   = dataset["validation"].map(format_example, remove_columns=dataset["validation"].column_names)
test_dataset  = dataset["test"].map(format_example, remove_columns=dataset["test"].column_names)
print("训练集测试")
print(dataset["train"][0]["intent"])
print(dataset["train"][0])
print(train_dataset[0])


def tokenize_function(example):
    text = example["text"]

    # 1. 找到 Response 起始位置
    response_start = text.index("### Response:") + len("### Response:")

    prompt_text = text[:response_start]
    response_text = text[response_start:]

    # 2. 分别 tokenize
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
        padding=False,
    )

    response_tokens = tokenizer(
        response_text,
        add_special_tokens=False,  # 很关键
        truncation=True,
        max_length=2048,
        padding=False,
    )

    # 3. 拼接 input_ids / attention_mask
    input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)

    # 4. 构造 labels：prompt 部分 mask 掉
    labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=["text"],
    batched=False,
)

val_dataset = val_dataset.map(
    tokenize_function,
    remove_columns=["text"],
    batched=False,
)

test_dataset = test_dataset.map(
    tokenize_function,
    remove_columns=["text"],
    batched=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./qwen3-intent-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.model.save_pretrained("./qwen3-intent-lora/final_adapter")
tokenizer.save_pretrained("./qwen3-intent-lora/final_adapter")