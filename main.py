import json
import numpy as np
import networkx as nx
import requests
from itertools import combinations
import asyncio
import aiohttp
import os
import datetime
import re
from src import config as cfg
import src.planning as planning
import src.io_utils as io_utils
import src.embeddings as embeddings
import src.judger as judger
import src.generator as generator
import src.refiner as refiner
from make_dag import is_dag, make_dag, save_graph_json, dag_to_levels

# ===================== 自动化流程 =====================
def _derive_paths(task_path):
    b = os.path.basename(task_path)
    m = re.match(r"task_planning_(\d{8}_\d{6})(?:_(\d+))?\.jsonl$", b)
    if not m:
        return None, None
    ts, run = m.group(1), m.group(2)
    if run:
        gen = f"data/original_content/output_{ts}_{run}.jsonl"
        ref = f"data/refined_content/refined_{ts}_{run}.jsonl"
    else:
        gen = f"data/original_content/output_{ts}.jsonl"
        ref = f"data/refined_content/refined_{ts}.jsonl"
    return gen, ref

def _find_resume_for_query(query):
    base = "data/task"
    if not os.path.isdir(base):
        return None
    candidates = []
    for fname in os.listdir(base):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(base, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline()
            if not first:
                continue
            rec = json.loads(first)
            if rec.get("query") != query:
                continue
        except Exception:
            continue
        gen, ref = _derive_paths(path)
        if not gen:
            continue
        records = io_utils.load_chapters(path)
        total = len(records)
        generated = io_utils.load_generated_chapters(gen)
        gen_count = len(generated)
        mtime = os.path.getmtime(gen) if os.path.exists(gen) else 0.0
        candidates.append({"task": path, "gen": gen, "ref": ref, "total": total, "gen_count": gen_count, "mtime": mtime})
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["gen_count"], x["mtime"]), reverse=True)
    return candidates[0]


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
    
    tasks = []
    with open(input_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks.append(task)
    print(f"从 {input_jsonl_file} 中读取了 {len(tasks)} 个任务")
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"处理第 {i+1}/{len(tasks)} 个任务")
        print(f"{'='*60}")
        query = task.get("prompt", "")
        word_count = task.get("length", 5000)
        style = task.get("type", "学术手册")
        task_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_jsonl, orig_gen, orig_ref = cfg.get_file_names()
        print(f"任务描述: {query}")
        print(f"字数限制: {word_count}")
        print(f"写作风格: {style}")
        resume = _find_resume_for_query(query)
        if resume:
            cfg.set_file_names(resume["task"], resume["gen"], resume["ref"])
            print("\n检测到断点，继续上次进度...")
            print(f"已完成章节: {resume['gen_count']}/{resume['total']}")
            try:
                print("\n步骤2: 生成正文...")
                records = io_utils.load_chapters(cfg.JSONL_FILE)
                similarities = embeddings.compute_similarities(records, threshold=cfg.SIM_THRESHOLD)
                print(f"发现 {len(similarities)} 个相似度候选对（> {cfg.SIM_THRESHOLD}）")
                G, judge_results = await judger.async_build_graph(records, similarities, cfg.MAX_CONCURRENT_GPT_CALLS)
                print(f"原始有向图 G 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
                DAG = make_dag(G)
                print(f"去环后 DAG 节点数: {DAG.number_of_nodes()}, 边数: {DAG.number_of_edges()}")
                levels = dag_to_levels(DAG)
                generated_chapters = await generator.generate_layers(records, levels)
                print(f"\n正文生成完成，共生成 {len(generated_chapters)} 个章节，已保存到 {cfg.GENERATED_JSONL}")
                both_pairs = set()
                for u_i, u_j, u_sim, u_dir, u_reason in judge_results:
                    if u_dir == "both":
                        pair = (min(u_i, u_j), max(u_i, u_j))
                        both_pairs.add(pair)
                both_pairs = sorted(list(both_pairs))
                print(f"检测到 {len(both_pairs)} 个 direction='both' 的 pair，进入 pairwise refine 阶段")
                generated_chapters = await refiner.refine_pairs(both_pairs, generated_chapters)
                io_utils.save_all_chapters(generated_chapters, jsonl_file=cfg.REFINED_JSONL)
                print(f"回环强化完成，已把 refine 后的章节保存到 {cfg.REFINED_JSONL}")
                print(f"\n任务 {i+1} 断点续跑完成")
                print(f"任务编排: {cfg.JSONL_FILE}")
                print(f"原始正文: {cfg.GENERATED_JSONL}")
                print(f"优化正文: {cfg.REFINED_JSONL}")
            except Exception as e:
                print(f"处理任务 {i+1} 断点续跑时出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                cfg.set_file_names(orig_jsonl, orig_gen, orig_ref)
            continue
        for run_idx in range(1, 6):
            cfg.set_run_file_names(task_timestamp, run_idx)
            try:
                print("\n步骤1: 生成大纲和任务编排...")
                outline, _ = await planning.async_gen_task_planning_main(query, word_count, style)
                print("\n步骤2: 生成正文...")
                records = io_utils.load_chapters(cfg.JSONL_FILE)
                similarities = embeddings.compute_similarities(records, threshold=cfg.SIM_THRESHOLD)
                print(f"发现 {len(similarities)} 个相似度候选对（> {cfg.SIM_THRESHOLD}）")
                G, judge_results = await judger.async_build_graph(records, similarities, cfg.MAX_CONCURRENT_GPT_CALLS)
                print(f"原始有向图 G 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
                DAG = make_dag(G)
                print(f"去环后 DAG 节点数: {DAG.number_of_nodes()}, 边数: {DAG.number_of_edges()}")
                levels = dag_to_levels(DAG)
                open(cfg.GENERATED_JSONL, "w", encoding="utf-8").close()
                generated_chapters = await generator.generate_layers(records, levels)
                print(f"\n正文生成完成，共生成 {len(generated_chapters)} 个章节，已保存到 {cfg.GENERATED_JSONL}")
                both_pairs = set()
                for u_i, u_j, u_sim, u_dir, u_reason in judge_results:
                    if u_dir == "both":
                        pair = (min(u_i, u_j), max(u_i, u_j))
                        both_pairs.add(pair)
                both_pairs = sorted(list(both_pairs))
                print(f"检测到 {len(both_pairs)} 个 direction='both' 的 pair，进入 pairwise refine 阶段")
                generated_chapters = await refiner.refine_pairs(both_pairs, generated_chapters)
                io_utils.save_all_chapters(generated_chapters, jsonl_file=cfg.REFINED_JSONL)
                print(f"回环强化完成，已把 refine 后的章节保存到 {cfg.REFINED_JSONL}")
                print(f"\n任务 {i+1} 第 {run_idx} 次生成完成")
                print(f"任务编排: {cfg.JSONL_FILE}")
                print(f"原始正文: {cfg.GENERATED_JSONL}")
                print(f"优化正文: {cfg.REFINED_JSONL}")
            except Exception as e:
                print(f"处理任务 {i+1} 第 {run_idx} 次时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                cfg.set_file_names(orig_jsonl, orig_gen, orig_ref)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动生成内容工具")
    parser.add_argument("--input", type=str, help="输入的JSONL文件路径，每行包含'prompt'、'length'和'type'字段")
    parser.add_argument("--task_planning", type=str, help="生成任务编排的查询内容")
    parser.add_argument("--word_count", type=int, default=5000, help="总字数限制（默认5000）")
    parser.add_argument("--style", type=str, default="学术手册", help="写作风格（默认'学术手册'）")
    parser.add_argument("--example", action="store_true", help="运行示例")
    
    args = parser.parse_args()
    if args.input:
        asyncio.run(auto_process_tasks(args.input))
