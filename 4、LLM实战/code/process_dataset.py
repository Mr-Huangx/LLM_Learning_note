fw = open("pretrain_dataset_small.jsonl", "w")
i = 0

# 此处对目标数据集进行裁切操作，只读取其中1w行的数据
file_name = "/home/huangxin/LLM_learning/mobvoi_seq_monkey_general_open_corpus.jsonl"
with open(file_name, "r") as f:
    while i <= 10000:
        line = f.readline()
        fw.write(line)
        i += 1
fw.close()