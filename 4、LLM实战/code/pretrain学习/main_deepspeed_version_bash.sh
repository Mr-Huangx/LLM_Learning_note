CUDA_VISIBLE_DEVICES=0

deepspeed main_deepspeed_version.py \
    --config_name Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --train_files pretrain_dataset_small.jsonl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --output_dir output/pretrain \
    --eval_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 1024 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ds_config_zero2.json \
    --report_to swanlab\
    --activation_checkpointing True
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \