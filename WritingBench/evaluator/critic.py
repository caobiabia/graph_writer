import time
from typing import Callable, List, Dict
from openai import OpenAI

class CriticAgent(object):
    def __init__(self,
                 system_prompt: str = None,
                 model_name: str = "WritingBench-Critic-Model-Qwen-7B",
                 api_base: str = "http://10.98.36.99:21004/v1",
                 api_key: str = "EMPTY"):
        """
        初始化 CriticAgent
        :param system_prompt: 系统提示词
        :param model_name: 启动 vllm serve 时指定的模型名称或路径
        :param api_base: vllm serve 的地址，注意加上 /v1
        :param api_key: vllm 默认不需要 key，可以填 EMPTY
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        
        # 初始化 OpenAI 客户端连接到你的 vLLM 服务
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    def call_critic(self,
            messages: List[Dict[str, str]],
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 30000):

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                # 使用 OpenAI 兼容接口调用
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=int(max_length),
                    # 如果需要关闭流式输出或其他特定参数，可以在这里添加
                    # extra_body={"repetition_penalty": 1.0} 
                )
                # 获取返回的文本内容
                return response.choices[0].message.content
            
            except Exception as e:
                print(f"Attempt {attempt+1}: API call failed due to error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")
    
    def basic_success_check(self, response):
        if not response:
            print("Empty response received.")
            return False
        else:
            return True
    
    def run(self,
            prompt: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 30000,
            max_try: int = 5,
            success_check_fn: Callable = None):
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        success = False
        try_times = 0
        response_text = ""

        while try_times < max_try:
            try:
                response_text = self.call_critic(
                    messages=messages,
                    top_p=top_p,
                    temperature=temperature,
                    max_length=max_length,
                )

                if success_check_fn is None:
                    # 如果没有提供检查函数，默认只要有返回内容就算成功
                    check_func = lambda x: True
                else:
                    check_func = success_check_fn
                
                if check_func(response_text):
                    success = True
                    break
                else:
                    # 如果内容检查不通过（例如格式不对），增加尝试次数
                    print(f"Check failed on attempt {try_times+1}")
                    try_times += 1
            except Exception as e:
                print(f"Run loop exception: {e}")
                try_times += 1
        
        return response_text, success

# # --- 使用示例 ---
# if __name__ == "__main__":
#     # 注意：model_name 需要和你 vllm serve 启动时的模型路径名称一致
#     # 或者是 vllm 自动解析出的名称。通常直接用路径名即可。
#     agent = CriticAgent(
#         system_prompt="你是一个专业的写作评价助手。",
#         model_name="WritingBench-Critic-Model-Qwen-7B", 
#         api_base="http://10.98.36.99:21004/v1"
#     )

#     prompt_text = "请评价一下这段文字：今天天气真好。"
#     result, is_success = agent.run(prompt=prompt_text, temperature=0.7)
    
#     if is_success:
#         print("Model Output:", result)
#     else:
#         print("Failed to get response.")