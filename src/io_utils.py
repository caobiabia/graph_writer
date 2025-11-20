import json

def append_chapter_jsonl(record_id, title, content, jsonl_file):
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": record_id, "title": title, "content": content}, ensure_ascii=False) + "\n")

def load_chapters(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for node_id, line in enumerate(f):
            if line.strip():
                chapter = json.loads(line)
                title = chapter.get("title", "无标题")
                summary = chapter.get("notes", "") or chapter.get("summary", "") or ""
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
    return records

def load_generated_chapters(jsonl_file):
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

def save_all_chapters(chapters_dict, jsonl_file):
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for cid in sorted(chapters_dict.keys()):
            rec = {"id": cid, "title": chapters_dict[cid]["title"], "content": chapters_dict[cid]["content"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")