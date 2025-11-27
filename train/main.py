import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import LMDataset, LMSortDataset, LMPackDataset
from trainer import TrainerNoShuffle
from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="THUDM/glm-4-9b")
    pack_loss: bool = field(default=False)
    # 新增：默认开启 Flash Attention 2，针对长文本必须开启
    use_flash_attention_2: bool = field(default=True, metadata={"help": "Whether to use flash attention 2"})

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    batch_method: str = field(default="naive")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    lora_enable: bool = False
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05 

@dataclass
class DataCollatorForLMDataset(object):
    """
    修正后的 Collator：
    适配 Qwen3 的 pad_token_id (151643)。
    原先的 argmin 逻辑在 pad_token_id 很大时会失效。
    """
    pad_token_id: int = 151643 

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取 input_ids 和 labels
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # 堆叠 Tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        # --- 核心修正逻辑 ---
        # 计算非 padding 的部分 (Qwen pad=151643)
        ne_pad_mask = input_ids.ne(self.pad_token_id)
        
        # 获取 batch 中最长的有效长度
        valid_lengths = ne_pad_mask.sum(dim=1)
        max_valid_length = valid_lengths.max().item()
        
        # 兜底：防止空数据，至少保留长度1
        max_valid_length = max(max_valid_length, 1)

        # 动态截断：去掉尾部多余的 Pad，节省显存
        input_ids = input_ids[:, :max_valid_length]
        labels = labels[:, :max_valid_length]
        # --- 修正结束 ---

        return dict(
            input_ids=input_ids,
            labels=labels
        )

@dataclass
class DataCollatorForLMPackDataset(object):
    # Pack 逻辑保持原样，如果有特殊需求需确认 30 这个系数
    def __call__(self, instances):
        input_ids, attention_masks = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ["input_ids", "attention_mask"])
        labels = ([instance["labels"][0].unsqueeze(0) for instance in instances], [instance["labels"][1].unsqueeze(0) for instance in instances])
        input_ids = torch.cat(input_ids, dim=0)
        labels = (torch.cat(labels[0], dim=0), torch.cat(labels[1], dim=0))
        labels = (labels[0], labels[1].sum()/30)
        max_length = input_ids.shape[1]
        attention_mask = attention_masks[0].squeeze()
        acc_length = max_length
        for new_attention_mask in attention_masks[1:]:
            new_attention_mask = new_attention_mask.squeeze()
            attention_mask = torch.cat([attention_mask, new_attention_mask[1:]+acc_length], dim=0)
            acc_length += max_length
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def make_supervised_data_module(data_args, tokenizer) -> Dict:
    print("loading data...")
    # 获取 tokenizer 的 pad_token_id，如果获取不到则使用 Qwen 默认
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643

    if data_args.batch_method == "naive":
        train_dataset = LMDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset(pad_token_id=pad_id)
    elif data_args.batch_method == "pack":
        train_dataset = LMPackDataset(data_args.train_file)
        data_collator = DataCollatorForLMPackDataset()
    elif data_args.batch_method == "sort":
        train_dataset = LMSortDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset(pad_token_id=pad_id)
    print("finish loading data")
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 1. 确定 Attention 实现
    # 强制使用 PyTorch 原生 SDPA 加速，无需安装 flash-attn 库，且支持长文本
    print("Using PyTorch SDPA (Scaled Dot Product Attention)...")
    attn_implementation = "sdpa"
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 如果 eos 也是 None，强制指定 Qwen 的 pad
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 151643

    # 3. 加载模型
    # 统一使用 AutoModel 加载，并注入 attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        attn_implementation=attn_implementation 
    )
        
    # 4. LoRA 初始化
    if training_args.lora_enable:
        print("Initializing LoRA...")
        lora_config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules='all-linear', # peft 新版支持，或者写具体层列表
                lora_dropout=training_args.lora_dropout,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
        
        # 显存优化：LoRA + Gradient Checkpointing 需要开启 input_require_grads
        if training_args.gradient_checkpointing: 
            model.enable_input_require_grads() 
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if model_args.pack_loss:
        model.pack_loss = True
    
    # 将 tokenizer 传入，以便正确设置 pad_token_id
    data_module = make_supervised_data_module(data_args=data_args, tokenizer=tokenizer)

    trainer = TrainerNoShuffle(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型和 Tokenizer
    trainer.save_model()
    if training_args.output_dir:
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()