#!/bin/bash

# 1. 路径配置
BASE_MODEL="/data/home/Yanchu/llm_repo/Qwen3-8B"
DATA_ROOT="data_completed"
MERGED_OUT="data_completed/qwen/graph_writer/classified/long_5k_30k.jsonl"
PRETOKEN_DIR="data_completed/qwen/graph_writer/long_5k_30k"
OUTPUT_DIR="models/qwen3-8b-lora-5k-30k"

# 2. 数据预处理
python train/merge_qwen_sft.py --data_root "$DATA_ROOT" --output "$MERGED_OUT"
python train/pre_tokenize_qwen3.py --input "$MERGED_OUT" --output_dir "$PRETOKEN_DIR" --tokenizer_path "$BASE_MODEL"

# 3. 显存优化环境变量 (这一行非常重要，保持不动)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6,7

# 4. 启动训练
# 关键修改：显式添加了 --use_flash_attention_2 True
torchrun --nproc_per_node=2 --master_port=29505 train/main.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file "$PRETOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --save_steps 200 \
    --bf16 True \
    --gradient_checkpointing True \
    --use_flash_attention_2 True \
    --lora_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False