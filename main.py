import json
import numpy as np
import networkx as nx
import requests
from itertools import combinations
import asyncio
import aiohttp
from make_dag import is_dag, make_dag, save_graph_json, dag_to_levels

# ===================== 配置 =====================
# Embedding 配置（Silicon Flow API）
EMBED_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_TOKEN = "sk-dhtlheweglbnekhitzcnoaigaeuxxlozvihrkbkrimbchtze"

# GPT 配置（远端）
GPT_API_KEY = "sk-dhtlheweglbnekhitzcnoaigaeuxxlozvihrkbkrimbchtze"
GPT_MODEL = "Qwen/Qwen3-32B"
GPT_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
GPT_HEADERS = {
    "Authorization": f"Bearer {GPT_API_KEY}",
    "Content-Type": "application/json"
}

# 文件及参数
import os
import datetime

# 自动生成带时间戳的文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
JSONL_FILE = f"data/task/task_planning_{timestamp}.jsonl"               # 输入章节 JSONL 文件（大纲）
GENERATED_JSONL = f"data/original_content/output_{timestamp}.jsonl"  # 生成阶段输出（保留）
REFINED_JSONL = f"data/refined_content/refined_{timestamp}.jsonl"  # refine 后输出（新文件）

# 确保data目录存在
os.makedirs("data", exist_ok=True)
SIM_THRESHOLD = 0.4
MAX_CONCURRENT_GPT_CALLS = 10

# ===================== 工具函数 =====================
def append_chapter_jsonl(record_id, title, content, jsonl_file=GENERATED_JSONL):
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": record_id, "title": title, "content": content}, ensure_ascii=False) + "\n")

def load_chapters(file_path):
    """读取一级章节（outline.jsonl），返回节点列表"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for node_id, line in enumerate(f):
            if line.strip():
                chapter = json.loads(line)
                title = chapter.get("title", "无标题")
                summary = chapter.get("notes", "") or chapter.get("summary", "") or ""
                # 保留其他字段如 subtitles, length_limit, style
                record = {
                    "id": node_id,
                    "title": title,
                    "summary": summary,
                    "subtitles": chapter.get("subtitles", []),
                    "notes": chapter.get("notes", ""),
                    "length_limit": chapter.get("length_limit", 1800),
                    "style": chapter.get("style", "学术手册")
                }
                records.append(record)
    print(f"读取了 {len(records)} 个章节节点")
    return records

def gen_embed(text: str) -> np.ndarray:
    """生成 embedding（同步）"""
    try:
        payload = {
            "model": EMBED_MODEL,
            "input": text
        }
        headers = {
            "Authorization": f"Bearer {EMBED_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(EMBED_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return np.array(result['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding 失败: {e}")
        raise

def compute_similarities(records, threshold=SIM_THRESHOLD):
    """计算余弦相似度，返回高于阈值的候选对 (sim, i, j)"""
    embeds = [gen_embed(r["title"] + " " + r["summary"]) for r in records]
    embeds = np.array(embeds)
    embeds_norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    similarities = []
    for i, j in combinations(range(len(records)), 2):
        sim = float(embeds_norm[i] @ embeds_norm[j])
        if sim > threshold:
            similarities.append((sim, i, j))
    similarities.sort(reverse=True)
    return similarities

# ===================== GPT 判断方向（异步） =====================
async def async_gpt_judge_direction(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, i, j, records, sim):
    title_i, summary_i = records[i]["title"], records[i]["summary"]
    title_j, summary_j = records[j]["title"], records[j]["summary"]

    async with semaphore:
        prompt = f"""
你将判断两个章节节点之间的逻辑依赖关系：

节点 A:
- 标题: "{title_i}"
- 摘要: "{summary_i}"

节点 B:
- 标题: "{title_j}"
- 摘要: "{summary_j}"

已知它们的相似度: {sim:.4f}（可辅助判断，但非唯一依据）。

请基于逻辑关系判断依赖方向：
1. 如果节点 A 的内容是节点 B 的前提，输出 "i_to_j"
2. 如果节点 B 的内容是节点 A 的前提，输出 "j_to_i"
3. 如果两者互为前提或具有相关性，输出 "both"
4. 如果没有明显逻辑依赖，输出 "none"

要求：
- 输出严格 JSON 格式：
{{"direction": "i_to_j/j_to_i/both/none", "reason": "简短解释逻辑关系"}}
- 解释尽量简洁，突出前提关系或依赖依据
- 不输出多余文本
"""
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.0
        }
        try:
            async with session.post(GPT_API_URL, headers=GPT_HEADERS, json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()
                content = response_json["choices"][0]["message"]["content"]
                try:
                    result = json.loads(content)
                except Exception:
                    # 尝试提取第一个 JSON 对象
                    import re
                    m = re.search(r"\{.*\}", content, re.S)
                    if m:
                        result = json.loads(m.group(0))
                    else:
                        result = {"direction": "none", "reason": ""}
                print(f"{title_i} <=> {title_j} ({sim:.4f}): {result.get('direction', 'none')} - {result.get('reason', '')}")
                return i, j, sim, result.get("direction", "none"), result.get("reason", "")
        except Exception as e:
            print(f"GPT 判断失败 ({title_i} <=> {title_j}): {e}")
            return i, j, sim, "none", str(e)

# ===================== 异步构建原始有向图（保留 GPT 判断） =====================
async def async_build_graph(records, similarities, max_concurrent_calls: int):
    G = nx.DiGraph()
    for r in records:
        G.add_node(r["id"], title=r["title"], summary=r["summary"])

    semaphore = asyncio.Semaphore(max_concurrent_calls)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for sim, i, j in similarities:
            tasks.append(async_gpt_judge_direction(session, semaphore, i, j, records, sim))
        print(f"\n开始并行调用 {len(tasks)} 个 GPT 判断任务 (最大并发: {max_concurrent_calls})...")
        results = await asyncio.gather(*tasks)

    # 根据 GPT 返回结果构建图
    for i, j, sim, direction, reason in results:
        if direction == "i_to_j":
            G.add_edge(i, j, weight=sim, reason=reason)
        elif direction == "j_to_i":
            G.add_edge(j, i, weight=sim, reason=reason)
        elif direction == "both":
            G.add_edge(i, j, weight=sim, reason=reason)
            G.add_edge(j, i, weight=sim, reason=reason)
        # direction == "none" 则不添加边
    return G, results  # 返回 G 和原始判断结果列表（后续用于选择 both 列表）

# ===================== 生成章节正文（异步） =====================
async def async_generate_chapter(session, semaphore, record, previous_contents):
    async with semaphore:
        prompt = f"""
你是一个写作助手，请根据以下要求和章节大纲生成正文内容：

标题：{record['title']}
子标题：{', '.join(record.get('subtitles', []))}
摘要/notes：{record.get('notes', '')}
长度限制：{record.get('length_limit', 10000)} 字以内
写作风格：{record.get('style', '')}

已完成上游章节内容：
{previous_contents}

请直接生成本章节markdown格式的正文内容，概念清晰、逻辑完整，不要返回其他无关内容。
"""
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.7
        }
        async with session.post(GPT_API_URL, headers=GPT_HEADERS, json=payload) as resp:
            resp.raise_for_status()
            r = await resp.json()
            content = r["choices"][0]["message"]["content"]
            return content

# ===================== pairwise refine（只针对 direction == "both" 的 pair） =====================
async def async_refine_pair(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                            i, j, text_i, text_j):
    async with semaphore:
        prompt = f"""
你将对两段章节文本做局部校正以恢复/强化它们之间的双向依赖（互为前提）。

章节 A (id={i}) 内容：
{text_i}

章节 B (id={j}) 内容：
{text_j}

任务：
1) 在不改变原意的前提下，尽可能使两段内容逻辑一致，补充或修改必要的承接句或前提陈述。
2) 仅输出严格 JSON，字段只包含 "A_new" 和 "B_new"：
{{"A_new": "...", "B_new": "..."}}
- 如果只需修改 A 或只需修改 B，可把另一个字段设为 null。
- 不要输出其他字段或注释。
"""
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.0
        }
        try:
            async with session.post(GPT_API_URL, headers=GPT_HEADERS, json=payload) as resp:
                resp.raise_for_status()
                r = await resp.json()
                content = r["choices"][0]["message"]["content"]
                try:
                    res = json.loads(content)
                except Exception:
                    import re
                    m = re.search(r"\{.*\}", content, re.S)
                    if m:
                        res = json.loads(m.group(0))
                    else:
                        res = {"A_new": None, "B_new": None}
                return i, j, res
        except Exception as e:
            print(f"Refine 调用失败: {i} <=> {j} : {e}")
            return i, j, {"A_new": None, "B_new": None}

# ===================== 读取已生成章节并保存函数 =====================
def load_generated_chapters(jsonl_file=GENERATED_JSONL):
    chapters = {}
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    chapters[int(rec["id"])] = {"title": rec["title"], "content": rec["content"]}
    except FileNotFoundError:
        pass
    return chapters

def save_all_chapters(chapters_dict, jsonl_file=REFINED_JSONL):
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for cid in sorted(chapters_dict.keys()):
            rec = {"id": cid, "title": chapters_dict[cid]["title"], "content": chapters_dict[cid]["content"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ===================== 生成markdown格式的大纲 =====================
def generate_outline(query, word_count, style):
    """
    调用GPT模型生成markdown格式的大纲
    
    Args:
        query (str): 用户需求
        word_count (int): 总字数限制
        style (str): 写作风格
    Returns:
        str: 生成的markdown格式大纲
    """
    prompt = f"""你是一位专业的结构策划师与内容架构专家。你的任务是： 
根据用户需求生成一篇逻辑清晰、层次规范、内容充实的大纲。 

--------------------- 
【输入信息】 
用户需求： 
{query} 
总字数限制：
{word_count}
风格：
{style}
--------------------- 

【生成要求】 
1. 输出格式： 
   - 仅输出 Markdown 格式的大纲内容。 
   - 严禁输出任何额外说明、前言、总结、提示词解释等非大纲内容。 
   - 每个标题下方必须紧跟一段内容概要，概要用括号包裹，并换行书写。
   - 文章大标题不计入正文标题层级。 

2. 动态结构约束：
   - 一级标题数量需依据总字数自动规划：以每章可承载 1000~5000 字为参考区间估算所需章节数量。（{word_count} 字合理章节数量≈ {word_count / 2000} ~ {word_count / 6000} 之间）

3. 内容要求：
   - 以“最少层级，可以完整表达逻辑”为目标，不做无必要的细粒度拆分
   - 全文架构必须能在总字数 {word_count} 限额内合理承载
输出格式：
Markdown 格式的大纲，每个标题下方紧跟内容概要（用括号包裹）。
# 一级标题
（内容概要）
## 二级标题
（内容概要）
...
"""
    
    payload = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"生成大纲失败: {e}")
        raise

# ===================== 根据大纲做任务编排 =====================
def generate_task_planning(query, outline, word_count, style, save_to_jsonl=True):
    """
    根据markdown格式的大纲做任务编排，并可选择保存为JSONL格式
    
    Args:
        query (str): 用户需求
        outline (str): markdown格式的大纲
        word_count (int): 总字数限制
        style (str): 写作风格
        save_to_jsonl (bool): 是否将结果保存为JSONL格式到data目录，默认为True
        
    Returns:
        str: JSON格式的任务编排
    """
    prompt = f"""你是一名专业的内容架构师和写作任务规划专家，你的任务是根据用户提供的信息，生成文章正文的任务编排 JSON，用于后续按块生成内容。  

要求： 

1. JSON 必须是一个列表，每个列表元素对应文章的一个一级标题块。  

2. 每个块必须包含以下字段：  
   - "article_title"：文章大标题（每个块内都应相同） 
   - "title"：一级标题名称和序号  
   - "subtitles"：该一级标题下的子标题列表，列表中每个元素为字符串，可包含二级或三级标题，若大纲中无子标题则为空列表[]
   - "length_limit"：该块的字数限制，严格按照总字数要求和写作风格进行合理分配
   - "style"：写作风格，与整体文章风格一致，可在块级别适当微调  
   - "notes"：对该块的额外说明或写作重点，便于正文生成时把握内容重点 
 
3. 输出要求：  
   - 严格按照 JSON 格式输出，只输出 JSON，不要包含任何解释或注释。  
   - JSON 必须可直接解析，不要出现多余字符。  
   - 子标题可包含不同层级的信息，确保生成内容结构清晰。  
   - 如果某个一级标题下没有子标题，subtitles 可以为空列表。  
   - length_limit 总和应尽量接近用户提供的总字数。 
 
输入信息： 
 
- 用户需求：{query}  
- 大纲：{outline}  
- 总字数限制：{word_count}  
- 写作风格：{style}  
 
请根据以上信息生成 JSON，保证每个一级标题块都合理分配字数，并保留内容层次结构。"""
    
    payload = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GPT_API_URL, headers=GPT_HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        json_content = result["choices"][0]["message"]["content"]
        
        # 如果需要，将JSON转换为JSONL并保存到data目录
        if save_to_jsonl:
            save_task_planning_to_jsonl(json_content, query, outline, word_count, style)
            
        return json_content
    except Exception as e:
        print(f"生成任务编排失败: {e}")
        raise

def save_task_planning_to_jsonl(json_content, query, outline, word_count, style):
    """
    将任务编排JSON转换为JSONL格式并保存到data目录
    
    Args:
        json_content (str): JSON格式的任务编排
        query (str): 用户需求
        outline (str): markdown格式的大纲
        word_count (int): 总字数限制
        style (str): 写作风格
    """
    try:
        # 打印原始内容用于调试
        print(f"GPT返回的原始内容:\n{json_content[:1000]}...\n")  # 只打印前1000字符，避免输出过长
        
        # 尝试提取JSON部分 - 处理可能包含非JSON文本的情况
        # 查找第一个 [ 或 { 和最后一个 ] 或 }
        start_idx = min(json_content.find('['), json_content.find('{'))
        if start_idx == -1:
            start_idx = max(json_content.find('['), json_content.find('{'))
        
        end_idx = max(json_content.rfind(']'), json_content.rfind('}'))
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = json_content[start_idx:end_idx+1]
            print(f"提取的JSON长度: {len(json_content)} 字符")
        else:
            print("无法在返回内容中找到有效的JSON格式")
            raise ValueError("返回内容中不包含有效的JSON格式")
        
        # 解析JSON内容
        try:
            task_planning = json.loads(json_content)
            print(f"成功解析JSON，包含 {len(task_planning)} 个任务块")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"尝试修复JSON格式...")
            # 尝试修复常见的JSON问题
            json_content = json_content.strip()
            if not json_content.startswith('[') and json_content.startswith('{'):
                # 如果是单个对象而不是数组，包装成数组
                json_content = f"[{json_content}]"
            
            # 再次尝试解析
            task_planning = json.loads(json_content)
            print(f"修复后成功解析JSON，包含 {len(task_planning)} 个任务块")
        
        # 使用全局变量中的时间戳，确保文件名一致
        filepath = JSONL_FILE
        
        # 将JSON转换为JSONL格式并保存
        with open(filepath, "w", encoding="utf-8") as f:
            for i, block in enumerate(task_planning):
                # 创建适合微调训练的记录格式
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
        
        print(f"任务编排已保存为JSONL格式到: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存任务编排JSONL失败: {e}")
        print(f"原始内容: {json_content}")
        raise

# ===================== 使用大纲生成和任务编排 =====================
def gen_task_planning_main(query, word_count, style):
    """
    使用generate_outline和generate_task_planning函数生成文章大纲和任务编排
    
    Args:
        query (str): 用户需求
        word_count (int): 总字数限制
        style (str): 写作风格
    """

    
    # 1. 生成markdown格式的大纲
    print("正在生成大纲...")
    outline = generate_outline(query, word_count, style)
    print("生成的大纲：")
    print(outline)
    print("\n" + "="*50 + "\n")
    
    # 2. 根据大纲生成任务编排（默认保存为JSONL）
    print("正在生成任务编排...")
    task_planning = generate_task_planning(query, outline, word_count, style, save_to_jsonl=True)
    print("生成的任务编排：")
    print(task_planning)
    
    # 3. 解析任务编排JSON（可选）
    try:
        task_planning_json = json.loads(task_planning)
        print(f"\n任务编排包含 {len(task_planning_json)} 个章节块")
        for i, block in enumerate(task_planning_json):
            print(f"章节 {i+1}: {block['title']} (字数限制: {block['length_limit']})")
    except json.JSONDecodeError:
        print("无法解析任务编排JSON")
    
    return outline, task_planning

# ===================== 自动化流程 =====================
async def auto_process_tasks(input_jsonl_file):
    """
    从输入的JSONL文件中循环读取每个任务，自动执行生成任务和生成正文并优化的过程
    
    Args:
        input_jsonl_file (str): 输入的JSONL文件路径，每行包含"prompt"、"length"和"type"字段
    """
    # 确保目录存在
    os.makedirs("data/task", exist_ok=True)
    os.makedirs("data/original_content", exist_ok=True)
    os.makedirs("data/refined_content", exist_ok=True)
    
    # 读取输入文件中的所有任务
    tasks = []
    with open(input_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks.append(task)
    
    print(f"从 {input_jsonl_file} 中读取了 {len(tasks)} 个任务")
    
    # 处理每个任务
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"处理第 {i+1}/{len(tasks)} 个任务")
        print(f"{'='*60}")
        
        # 获取任务参数
        query = task.get("prompt", "")
        word_count = task.get("length", 5000)  # 默认5000字
        style = task.get("type", "学术手册")  # 默认学术手册风格
        
        task_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        global JSONL_FILE, GENERATED_JSONL, REFINED_JSONL
        original_jsonl_file = JSONL_FILE
        original_generated_jsonl = GENERATED_JSONL
        original_refined_jsonl = REFINED_JSONL

        print(f"任务描述: {query}")
        print(f"字数限制: {word_count}")
        print(f"写作风格: {style}")

        for run_idx in range(1, 6):
            JSONL_FILE = f"data/task/task_planning_{task_timestamp}_{run_idx}.jsonl"
            GENERATED_JSONL = f"data/original_content/output_{task_timestamp}_{run_idx}.jsonl"
            REFINED_JSONL = f"data/refined_content/refined_{task_timestamp}_{run_idx}.jsonl"
            print(f"\n第 {run_idx} 次生成")
            try:
                print("\n步骤1: 生成大纲和任务编排...")
                outline, task_planning = gen_task_planning_main(query, word_count, style)
                print("\n步骤2: 生成正文...")
                records = load_chapters(JSONL_FILE)
                similarities = compute_similarities(records, SIM_THRESHOLD)
                print(f"发现 {len(similarities)} 个相似度候选对（> {SIM_THRESHOLD}）")
                G, judge_results = await async_build_graph(records, similarities, MAX_CONCURRENT_GPT_CALLS)
                print(f"原始有向图 G 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
                DAG = make_dag(G)
                print(f"去环后 DAG 节点数: {DAG.number_of_nodes()}, 边数: {DAG.number_of_edges()}")
                levels = dag_to_levels(DAG)
                generated_chapters = {}
                open(GENERATED_JSONL, "w", encoding="utf-8").close()
                async with aiohttp.ClientSession() as session:
                    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
                    previous_layer_contents = ""
                    for layer_idx, level in enumerate(levels):
                        print(f"\n生成第 {layer_idx+1} 层，共 {len(level)} 个章节并发处理")
                        tasks_list = [
                            async_generate_chapter(session, semaphore, records[node_id], previous_layer_contents)
                            for node_id in level
                        ]
                        layer_results = await asyncio.gather(*tasks_list)
                        previous_layer_contents = ""
                        for node_id, content in zip(level, layer_results):
                            append_chapter_jsonl(node_id, records[node_id]["title"], content, jsonl_file=GENERATED_JSONL)
                            previous_layer_contents += f"\n\n【{records[node_id]['title']}】\n{content}\n"
                            generated_chapters[node_id] = {"title": records[node_id]["title"], "content": content}
                print(f"\n正文生成完成，共生成 {len(generated_chapters)} 个章节，已保存到 {GENERATED_JSONL}")
                both_pairs = set()
                for u_i, u_j, u_sim, u_dir, u_reason in judge_results:
                    if u_dir == "both":
                        pair = (min(u_i, u_j), max(u_i, u_j))
                        both_pairs.add(pair)
                both_pairs = sorted(list(both_pairs))
                print(f"检测到 {len(both_pairs)} 个 direction='both' 的 pair，进入 pairwise refine 阶段")
                if both_pairs:
                    async with aiohttp.ClientSession() as session:
                        sem = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
                        refine_tasks = []
                        for u, v in both_pairs:
                            text_u = generated_chapters.get(u, {}).get("content", "")
                            text_v = generated_chapters.get(v, {}).get("content", "")
                            if not text_u and not text_v:
                                continue
                            refine_tasks.append(async_refine_pair(session, sem, u, v, text_u, text_v))
                        refine_results = await asyncio.gather(*refine_tasks)
                    for r_i, r_j, res in refine_results:
                        A_new = res.get("A_new")
                        B_new = res.get("B_new")
                        if A_new and isinstance(A_new, str) and A_new.strip():
                            generated_chapters[r_i]["content"] = A_new
                        if B_new and isinstance(B_new, str) and B_new.strip():
                            generated_chapters[r_j]["content"] = B_new
                    save_all_chapters(generated_chapters, jsonl_file=REFINED_JSONL)
                    print(f"回环强化完成，已把 refine 后的章节保存到 {REFINED_JSONL}")
                else:
                    print("没有 direction='both' 的 pair，跳过 refine 阶段。")
                    save_all_chapters(generated_chapters, jsonl_file=REFINED_JSONL)
                    print(f"已把当前章节备份为 {REFINED_JSONL}")
                print(f"\n任务 {i+1} 第 {run_idx} 次生成完成")
                print(f"任务编排: {JSONL_FILE}")
                print(f"原始正文: {GENERATED_JSONL}")
                print(f"优化正文: {REFINED_JSONL}")
            except Exception as e:
                print(f"处理任务 {i+1} 第 {run_idx} 次时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                JSONL_FILE = original_jsonl_file
                GENERATED_JSONL = original_generated_jsonl
                REFINED_JSONL = original_refined_jsonl

# ===================== 主流程 =====================
async def async_main(file_path):
    # 1. 读取大纲
    records = load_chapters(file_path)

    # 2. Embedding + 相似度候选对
    similarities = compute_similarities(records, SIM_THRESHOLD)
    print(f"发现 {len(similarities)} 个相似度候选对（> {SIM_THRESHOLD}）")

    # 3. 使用 GPT 判断方向，构建原始有向图 G（带环）
    G, judge_results = await async_build_graph(records, similarities, MAX_CONCURRENT_GPT_CALLS)

    # 保存原始 G 边集（用于后续识别 both 对）
    original_edges = list(G.edges(data=True))
    print(f"原始有向图 G 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 4. 去环（make_dag）
    DAG = make_dag(G)
    print(f"去环后 DAG 节点数: {DAG.number_of_nodes()}, 边数: {DAG.number_of_edges()}")

    # 5. DAG 拆层
    levels = dag_to_levels(DAG)

    # 6. 异步并发生成正文（按层）
    previous_contents = ""
    generated_chapters = {}
    # 清理或新建生成文件（避免重复 append 导致多行）
    open(GENERATED_JSONL, "w", encoding="utf-8").close()

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
        previous_layer_contents = ""  # 只保留上一层的内容
        for layer_idx, level in enumerate(levels):
            print(f"\n生成第 {layer_idx+1} 层，共 {len(level)} 个章节并发处理")
            tasks = [
                async_generate_chapter(session, semaphore, records[node_id], previous_layer_contents)
                for node_id in level
            ]
            layer_results = await asyncio.gather(*tasks)

            # 拼接当前层内容，作为下一层的上下文
            previous_layer_contents = ""
            for node_id, content in zip(level, layer_results):
                append_chapter_jsonl(node_id, records[node_id]["title"], content, jsonl_file=GENERATED_JSONL)
                previous_layer_contents += f"\n\n【{records[node_id]['title']}】\n{content}\n"
                generated_chapters[node_id] = {"title": records[node_id]["title"], "content": content}
    print(f"\n正文生成完成，共生成 {len(generated_chapters)} 个章节，已保存到 {GENERATED_JSONL}")

    # 7. 回环强化（R2）：对所有 GPT 判断为 "both" 的 pair 做 refine
    # 从 judge_results 中筛选 direction == 'both' 的 pair（去重）
    both_pairs = set()
    for i, j, sim, direction, reason in judge_results:
        if direction == "both":
            pair = (min(i, j), max(i, j))
            both_pairs.add(pair)
    both_pairs = sorted(list(both_pairs))
    print(f"检测到 {len(both_pairs)} 个 direction='both' 的 pair，进入 pairwise refine 阶段（并发受限）")

    # 并发执行 refine
    if both_pairs:
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
            refine_tasks = []
            for u, v in both_pairs:
                text_u = generated_chapters.get(u, {}).get("content", "")
                text_v = generated_chapters.get(v, {}).get("content", "")
                if not text_u and not text_v:
                    continue
                refine_tasks.append(async_refine_pair(session, sem, u, v, text_u, text_v))
            refine_results = await asyncio.gather(*refine_tasks)

        # 应用 refine 结果（只使用 A_new / B_new），覆盖原文（如果返回为空则保留原文）
        for i, j, res in refine_results:
            A_new = res.get("A_new")
            B_new = res.get("B_new")
            if A_new and isinstance(A_new, str) and A_new.strip():
                generated_chapters[i]["content"] = A_new
            if B_new and isinstance(B_new, str) and B_new.strip():
                generated_chapters[j]["content"] = B_new

        # 保存 refine 后的全部章节为新文件（不覆盖原始 GENERATED_JSONL）
        save_all_chapters(generated_chapters, jsonl_file=REFINED_JSONL)
        print(f"回环强化完成，已把 refine 后的章节保存为 {REFINED_JSONL}")
    else:
        print("没有 direction='both' 的 pair，跳过 refine 阶段。")
        # 仍然保存一份 identical 的 refined 文件以示完整性（可选）
        save_all_chapters(generated_chapters, jsonl_file=REFINED_JSONL)
        print(f"已把当前章节备份为 {REFINED_JSONL}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动生成内容工具")
    parser.add_argument("--input", type=str, help="输入的JSONL文件路径，每行包含'prompt'、'length'和'type'字段")
    parser.add_argument("--task_planning", type=str, help="生成任务编排的查询内容")
    parser.add_argument("--word_count", type=int, default=5000, help="总字数限制（默认5000）")
    parser.add_argument("--style", type=str, default="学术手册", help="写作风格（默认'学术手册'）")
    parser.add_argument("--example", action="store_true", help="运行示例")
    
    args = parser.parse_args()
    
    if args.example:
        # 运行示例
        query = "编写一本关于机器学习入门的指南"
        word_count = 5000
        style = "学术手册"
        outline, task_planning = gen_task_planning_main(query, word_count, style)
        asyncio.run(async_main(file_path=JSONL_FILE))
    elif args.input:
        # 自动化处理流程
        asyncio.run(auto_process_tasks(args.input))
    elif args.task_planning:
        # 只生成任务编排
        outline, task_planning = gen_task_planning_main(args.task_planning, args.word_count, args.style)
    else:
        # 默认运行原有的主流程
        asyncio.run(async_main(file_path=JSONL_FILE))
