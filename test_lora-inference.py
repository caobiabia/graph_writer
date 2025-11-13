import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer # 导入 TextStreamer
from peft import PeftModel, PeftConfig

def test_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    messages: list[dict],
    max_new_tokens: int =32768,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    加载原始模型和LoRA适配器，然后进行文本生成测试。

    Args:
        base_model_path (str): 原始基础模型的路径。
        lora_adapter_path (str): 训练后的LoRA适配器的路径。
        test_prompt (str): 用于测试的输入提示。
        max_new_tokens (int): 生成文本的最大新token数量。
        device (str): 运行模型的设备 ('cuda' 或 'cpu')。
    """
    print(f"正在加载基础模型: {base_model_path}")
    # 1. 加载原始的基础模型
    # 确保 dtype 与训练时一致，这里假设是 torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, # 根据您的训练配置调整
        trust_remote_code=True
    )

    print(f"正在加载分词器: {base_model_path}")
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    # 确保分词器有 pad_token，否则生成时可能会有问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # 对于生成任务，通常将填充放在左侧

    print(f"正在加载LoRA适配器: {lora_adapter_path}")
    # 3. 加载 LoRA 适配器并将其附加到基础模型上
    # PeftModel.from_pretrained 会自动将LoRA权重加载到base_model上
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # 4. (可选) 合并 LoRA 权重到基础模型
    # 如果您希望得到一个独立的、包含LoRA更新的完整模型，可以执行此步骤
    # model = model.merge_and_unload()
    # print("LoRA权重已合并到基础模型。")

    # 将模型移动到指定设备
    model.to(device)
    # 5. 设置模型为评估模式
    model.eval()

    print("\n--- 模型加载完成，开始生成文本 ---")
    print(f"输入消息: {messages}")

    # 6. 准备输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # 开启思考模式
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # 7. 创建 TextStreamer 实例
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 8. 生成文本 (使用 streamer)
    with torch.no_grad(): # 在推理时禁用梯度计算
        # 移除 stream=True，并添加 streamer 参数
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=True, # 开启采样，使生成更具多样性
            # temperature=0.01, # 采样温度，控制随机性
            # top_p=0.95, # Top-p 采样，控制生成词汇的范围
            # top_k=20, # Top-k 采样，控制生成词汇的范围
            # min_p=0, # 最小概率，控制生成词汇的范围
            # repetition_penalty=1.1, # 惩罚重复词汇
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer # 将 streamer 传递给 generate 方法
        )

    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

    # 解析思考内容和最终内容
    try:
        # 查找 \u003c/think\u003e 的 token ID (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("\n--- 思考内容 ---")
    print(thinking_content)
    print("\n--- 最终内容 ---")
    print(content)

if __name__ == "__main__":
    # 请根据您的实际路径进行修改
    BASE_MODEL_PATH = "E:\\Graph_writer\\models_link\\Qwen3-0.6B"
    # 假设您的LoRA适配器保存在训练输出目录的checkpoint-1子目录中
    LORA_ADAPTER_PATH = "E:\\Graph_writer\\models\\qwen3-0.6b-lora-test\\checkpoint-100"
    
    test_messages = [
        {"role": "user", "content": f"编写一本关于机器学习入门的指南\n写作风格: 学术手册\n总字数: 5000/no_think"}
    ]

    test_lora_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH, test_messages)