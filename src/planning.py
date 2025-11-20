import json
import asyncio
from . import config as cfg
import re

with open("prompts/outline.txt", "r") as f:
    outline_prompt = f.read()

with open("prompts/task.txt", "r") as f:
    task_prompt = f.read()


def parse_task_planning(text: str):
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    else:
        i = s.find("[")
        j = s.rfind("]")
        if i != -1 and j != -1:
            s = s[i:j+1]
    return json.loads(s)


async def generate_outline(query, word_count, style):
    prompt = outline_prompt.format(query=query, word_count=word_count, style=style)

    r = await cfg.openai_async_client.chat.completions.create(
        model=cfg.resolve_model_id(),
        messages=[
            {"role": "system","content": "你是一个大纲生成助手"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.5,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.15}
    )

    return r.choices[0].message.content

async def generate_task_planning(query, outline, word_count, style, save_to_jsonl=True):
    prompt = task_prompt.format(query=query, outline=outline, word_count=word_count, style=style)
    r = await cfg.openai_async_client.chat.completions.create(
        model=cfg.resolve_model_id(),
        messages=[
            {"role": "system", "content": "你是一个严格遵循大纲的任务规划助手"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.5,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.15}
    )
    json_content = r.choices[0].message.content
    if save_to_jsonl:
        save_task_planning_to_jsonl(json_content, query, outline, word_count, style)
    return json_content

def save_task_planning_to_jsonl(json_content, query, outline, word_count, style):
    task_planning = parse_task_planning(json_content)
    filepath = cfg.JSONL_FILE
    with open(filepath, "w", encoding="utf-8") as f:
        for i, block in enumerate(task_planning):
            record = {
                "id": i,
                "query": query,
                "outline": outline,
                "word_count": word_count,
                "style": style,
                "article_title": block.get("article_title", ""),
                "title": block.get("title", ""),
                "subtitles": block.get("subtitles", []),
                "length_limit": block.get("length_limit", 0),
                "style": block.get("style", style),
                "notes": block.get("notes", "")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return filepath

async def async_gen_task_planning_main(query, word_count, style):
    outline = await generate_outline(query, word_count, style)
    task_planning = await generate_task_planning(query, outline, word_count, style, save_to_jsonl=True)
    try:
        task_planning_json = parse_task_planning(task_planning)
        for _ in task_planning_json:
            pass
    except json.JSONDecodeError:
        pass
    return outline, task_planning

async def main():
    # ====== 测试输入数据 ======
    query = "人工智能对未来教育的影响"
    word_count = 1500
    style = "学术风格"

    # ====== 调用你的函数 ======
    outline, task_planning = await async_gen_task_planning_main(
        query=query,
        word_count=word_count,
        style=style
    )

    # ====== 输出结果 ======
    print("======= 生成的 Outline =======")
    print(outline)

    print("\n======= 生成的 Task Planning =======")
    print(task_planning)

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())