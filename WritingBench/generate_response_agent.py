import json
import os
import argparse
import asyncio
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src import config as cfg
import src.planning as planning
import src.embeddings as embeddings
import src.judger as judger
import src.generator as generator
import src.refiner as refiner
from make_dag import make_dag, dag_to_levels

def _clip(s, n=800):
    try:
        s = str(s)
    except Exception:
        return "<unprintable>"
    return s[:n].replace("\n", " ")

def save_output(output, file_name):
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

async def _agent_generate(query, word_count=3000, style="任意"):
    try:
        print(f"[agent] start query={_clip(query)}")
        print(f"[agent] cfg OPENAI_BASE_URL={getattr(cfg, 'OPENAI_BASE_URL', None)} GPT_MODEL={getattr(cfg, 'GPT_MODEL', None)}")
        print(f"[agent] cfg OPENAI_EMBED_BASE_URL={getattr(cfg, 'OPENAI_EMBED_BASE_URL', None)} EMBED_MODEL={getattr(cfg, 'EMBED_MODEL', None)}")
        print(f"[agent] outline_in query_head={_clip(query)} word_count={word_count} style={style}")
        outline = await planning.generate_outline(query, word_count, style)
        print(f"[agent] outline_out len={len(outline)} head={_clip(outline)}")
        print(f"[agent] task_in outline_head={_clip(outline)} word_count={word_count} style={style}")
        tp = await planning.generate_task_planning(query, outline, word_count, style, save_to_jsonl=False)
        print(f"[agent] task_out len={len(tp)} head={_clip(tp)}")
        print(f"[agent] parse_in head={_clip(tp)}")
        try:
            blocks = planning.parse_task_planning(tp)
            print(f"[agent] parse_out ok blocks_count={len(blocks)} first_titles={[b.get('title') or b.get('article_title') for b in blocks[:3]]}")
        except Exception as e:
            print(f"[agent] parse_out failed err={e}")
            blocks = [{"title": query, "subtitles": [], "length_limit": word_count, "style": style, "notes": outline}]
        print(f"[agent] records_out_count={len(blocks)} titles={[(_clip(b.get('title') or b.get('article_title'))) for b in blocks[:5]]}")
        records = []
        for i, b in enumerate(blocks):
            records.append({
                "id": i,
                "title": b.get("title") or b.get("article_title") or query,
                "summary": b.get("notes", ""),
                "subtitles": b.get("subtitles", []),
                "notes": b.get("notes", ""),
                "length_limit": b.get("length_limit", word_count),
                "style": b.get("style", style),
            })
        if len(records) <= 1:
            levels = [[r["id"]] for r in records]
            judge_results = []
            print(f"[agent] single-record flow")
        else:
            try:
                print(f"[agent] embed_in titles={[r['title'] for r in records[:5]]}")
                similarities = embeddings.compute_similarities(records, threshold=cfg.SIM_THRESHOLD)
                print(f"[agent] embed_out similarities_count={len(similarities)} sample={[ (int(i), int(j), float(sim)) for sim,i,j in similarities[:5] ]}")
            except Exception as e:
                print(f"[agent] embed_out error={e} fallback to sequential levels")
                similarities = []
            if similarities:
                try:
                    print(f"[agent] judge_in pairs_count={len(similarities)} sample={[ (int(i), int(j), float(sim)) for sim,i,j in similarities[:5] ]}")
                    G, judge_results = await judger.async_build_graph(records, similarities, cfg.MAX_CONCURRENT_GPT_CALLS)
                    print(f"[agent] judge_out graph_nodes={G.number_of_nodes()} graph_edges={G.number_of_edges()} sample={[ (ri, rj, rd) for ri, rj, _, rd, _ in judge_results[:5] ]}")
                    DAG = make_dag(G)
                    levels = dag_to_levels(DAG)
                except Exception as e:
                    print(f"[agent] judge_out error={e} fallback to sequential levels")
                    judge_results = []
                    levels = [[r["id"]] for r in records]
            else:
                judge_results = []
                levels = [[r["id"]] for r in records]
        print(f"[agent] dag_out levels_count={len(levels)} levels_sizes={[len(l) for l in levels]} levels={levels[:5]}")
        generated_chapters = {}
        semaphore = asyncio.Semaphore(cfg.MAX_CONCURRENT_GPT_CALLS)
        previous_layer_contents = ""
        for level in levels:
            print(f"[agent] gen_in level_ids={level} prev_head={_clip(previous_layer_contents)}")
            tasks = [generator.async_generate_chapter(semaphore, records[nid], previous_layer_contents) for nid in level]
            if tasks:
                layer_results = await asyncio.gather(*tasks, return_exceptions=True)
                for nid, content in zip(level, layer_results):
                    if isinstance(content, Exception):
                        print(f"[agent] gen_out_error id={nid} title={records[nid]['title']} err={content}")
                        content = "Connection error."
                    else:
                        print(f"[agent] gen_out_ok id={nid} title={records[nid]['title']} content_len={len(content)} head={_clip(content)}")
                    generated_chapters[nid] = {"title": records[nid]["title"], "content": content}
        previous_layer_contents = ""
        for nid in level:
            content = generated_chapters.get(nid, {}).get("content", "")
            title = records[nid]["title"]
            previous_layer_contents += f"\n\n【{title}】\n{content}\n"
        both_pairs = set()
        for u_i, u_j, u_sim, u_dir, u_reason in judge_results:
            if u_dir == "both":
                pair = (min(u_i, u_j), max(u_i, u_j))
                both_pairs.add(pair)
        both_pairs = sorted(list(both_pairs))
        print(f"[agent] refine_in pairs={both_pairs[:10]}")
        if both_pairs:
            try:
                generated_chapters = await refiner.refine_pairs(both_pairs, generated_chapters)
                print(f"[agent] refine_out updated_ids={[u for u,_ in both_pairs][:10]}")
            except Exception as e:
                print(f"[agent] refine_out error={e} skip refine")
        parts = []
        print(f"[agent] assemble_in chapters_count={len(generated_chapters)}")
        for level in levels:
            for nid in level:
                rec = generated_chapters.get(nid, {})
                t = rec.get("title", records[nid]["title"]) 
                c = rec.get("content", "")
                parts.append(f"【{t}】\n{c}")
        final_text = "\n\n".join(parts).strip()
        print(f"[agent] assemble_out final_len={len(final_text)} head={_clip(final_text)}")
        return final_text or outline
    except Exception as e:
        print(f"[agent] error={e}")
        return str(e)

def writer(query):
    return asyncio.run(_agent_generate(query))

def process(id_query_map, out_file, num_workers=None):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)
    if num_workers is None:
        num_workers = 1
    with tqdm(total=input_cnt, initial=existing_count, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            futures = {}
            next_submit = existing_count
            next_write = existing_count
            while next_write < input_cnt:
                while next_submit < input_cnt and len(futures) < int(num_workers):
                    c = contents[next_submit]
                    q = c["query"]
                    print(f"[agent] process_in index={c.get('index')} query_head={_clip(q)}")
                    f = executor.submit(writer, q)
                    futures[next_submit] = (f, c)
                    next_submit += 1
                f_c = futures.get(next_write)
                if f_c is None:
                    break
                f, c = f_c
                r = f.result()
                print(f"[agent] process_out index={c.get('index')} response_len={len(str(r))} head={_clip(r)}")
                data = {"index": c["index"], "response": r}
                save_output([data], out_file)
                cnt += 1
                pbar.update(1)
                futures.pop(next_write)
                next_write += 1
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent-driven generation for benchmark")
    parser.add_argument("--query_file", type=str, help="Path to the query file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of concurrent workers.")
    args = parser.parse_args()
    process(args.query_file, args.output_file, args.num_workers)

