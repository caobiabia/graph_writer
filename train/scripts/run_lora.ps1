$BASE_MODEL = "E:\Graph_writer\models_link\Qwen3-0.6B"
$DATA_ROOT = "data"
$MERGED_OUT = "data\qwen\graph_writer\merged.jsonl"
$PRETOKEN_DIR = "data\qwen\graph_writer"
$OUTPUT_DIR = "models\qwen3-0.6b-lora-test"

python train/merge_qwen_sft.py --data_root $DATA_ROOT --output $MERGED_OUT
python train/pre_tokenize_qwen3.py --input $MERGED_OUT --output_dir $PRETOKEN_DIR

python train/main.py `
  --model_name_or_path "$BASE_MODEL" `
  --train_file "$PRETOKEN_DIR" `
  --output_dir "$OUTPUT_DIR" `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 2e-4 `
  --num_train_epochs 100 `
  --logging_steps 1 `
  --save_steps 50 `
  --bf16 True `
  --gradient_checkpointing True `
  --lora_enable True `
  --lora_rank 64 `
  --lora_alpha 32 `
  --lora_dropout 0.05