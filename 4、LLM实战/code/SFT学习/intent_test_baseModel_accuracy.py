import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import torch
import re
import argparse
from datasets import load_dataset, load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

def extract_intent_from_response(text: str) -> str:
    """
    从模型输出中提取 intent 字符串
    规则：
    - 取最后一段 "### Response:" 之后的内容
    - 只取第一行
    - 去除多余空格
    """
    text = text.strip()
    # 如果存在 "### Response:" 关键字，取后面部分
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
    # 取第一行

    intent = text.split("\n")[1].strip()
    return intent




def main():
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

    # 2) load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained("./qwen3-intent-lora/final_adapter")

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = base_model
    model.eval()

    # 3) accuracy metric
    accuracy = evaluate.load("accuracy")

    preds = []
    labels = []

    # 4) iterate test dataset

    # def build_prompt(utterance: str) -> str:
    #     return f"""### Instruction:
    # You are an intent classification system.
    # Classify the user's intent into one of the predefined intent labels.：{intent_id2label}
    #
    # Rules:
    # - Only output the intent label.
    # - Do NOT provide explanations.
    # - If the intent is unclear, choose the closest matching label.
    #
    # ### Examples:
    #
    # User: I want to book a flight from New York to London.
    # Response: book_flight
    #
    # User: What is the weather like tomorrow?
    # Response: weather_query
    #
    # User: Cancel my reservation for tonight.
    # Response: cancel_reservation
    #
    #
    # ### User:
    # {utterance}
    #
    # ### Response:
    # """

    def build_prompt(utterance: str) -> str:
        return f"""
        ### Instruction:
        You are an intent classification system.
        Classify the <USER_INTENT> into one of the predefined intent labels.：{intent_id2label}
        
        Rules:
        - Only output the intent label.
        - Do NOT provide explanations.
        - If the intent is unclear, choose the closest matching label.
        
        Here are some examples:
        ### Examples:
        
        <USER_INTENT>: I want to book a flight from New York to London.
        <Response>: book_flight
        
        <USER_INTENT>: What is the weather like tomorrow?
        <Response>: weather_query
        
        <USER_INTENT>: Cancel my reservation for tonight.
        <Response>: cancel_reservation
        
        
        <USER_INTENT>: {utterance}
        <Response>:
        """


    PRED_CACHE = "./intent_preds_cache_base_model.jsonl"
    preds = []
    labels = []
    seen = set()
    buffer = []

    for idx, example in tqdm(
            enumerate(test_dataset),
            total=len(test_dataset),
            desc="Evaluating intents"
    ):
        if idx in seen:
            continue

        prompt = build_prompt(example["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(pred_text)
        # pred_intent = extract_intent_from_response(pred_text)
        # true_intent = intent_id2label[example["intent"]]
        # preds.append(pred_intent)
        # labels.append(true_intent)
        #
        # buffer.append({
        #     "pred": pred_intent,
        #     "label": true_intent
        # })

    if buffer:
        with open(PRED_CACHE, "a", encoding="utf-8") as f:
            for item in buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6) compute accuracy
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(preds)
    print(f"Intent Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
