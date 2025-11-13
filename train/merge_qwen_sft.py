import os, json, glob, re, argparse, random

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_user_prompt(task_lines):
    if not task_lines:
        return ""
    head = task_lines[0]
    q = head.get("query") or head.get("article_title", "")
    style = head.get("style")
    word_count = head.get("word_count")
    parts = [str(q).strip()]
    if style:
        parts.append(f"写作风格: {style}")
    if word_count:
        parts.append(f"总字数: {word_count}")
    return "\n".join(parts)

def main(data_root: str, out_path: str):
    refined_dir = os.path.join(data_root, "refined_content")
    task_dir = os.path.join(data_root, "task")
    ts_pattern = re.compile(r"_(\d{8}_\d{6})\.jsonl$")

    merged_records = []
    for refined_fp in glob.glob(os.path.join(refined_dir, "*.jsonl")):
        m = ts_pattern.search(refined_fp)
        if not m:
            continue
        ts = m.group(1)
        task_fp = os.path.join(task_dir, f"task_planning_{ts}.jsonl")
        if not os.path.exists(task_fp):
            print(f"[WARN] task file missing for {ts}")
            continue

        refined_lines = [obj for obj in read_jsonl(refined_fp) if "content" in obj]
        task_lines = list(read_jsonl(task_fp))

        try:
            refined_lines.sort(key=lambda x: int(x.get("id", 0)))
        except Exception:
            refined_lines.sort(key=lambda x: str(x.get("id", "")))

        long_response = "\n".join([r.get("content", "") for r in refined_lines])
        prompt = build_user_prompt(task_lines)
        merged_records.append({"prompt": prompt, "response": long_response})

    random.shuffle(merged_records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in merged_records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Merged {len(merged_records)} pairs -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="root path containing refined_content and task directories")
    parser.add_argument("--output", default="data/qwen/graph_writer/merged.jsonl")
    args = parser.parse_args()
    main(args.data_root, args.output)