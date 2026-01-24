import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# base_model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen3-0.6B",
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )
#
# tokenizer = AutoTokenizer.from_pretrained("./qwen3-intent-lora/final_adapter")
#
# model = PeftModel.from_pretrained(
#     base_model,
#     "./qwen3-intent-lora/final_adapter"
# )
#
# model.eval()
#
# prompt = """### Instruction:
# You are an intent classification system.
# Classify the user's intent into one of the predefined intent labels.
#
# ### User:
# how can i become an aerospace engineer
#
# ### Response:
# """
#
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=10,
#         do_sample=False,
#     )
#
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


from datasets import load_dataset, load_from_disk
# 1) load dataset
DATA_PATH = "./data/clinc_oos_small"

if os.path.exists(DATA_PATH):
    dataset = load_from_disk(DATA_PATH)
else:
    dataset = load_dataset("clinc_oos", "small")
    dataset.save_to_disk(DATA_PATH)

test_dataset = dataset["test"]
# 获取 intent id -> label name 的映射
intent_id2label = dataset["train"].features["intent"].names

dataset = load_dataset("clinc_oos", "small")

intent_id2label = dataset["train"].features["intent"].names
print(intent_id2label)
print(intent_id2label[1])