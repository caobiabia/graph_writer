import os
import re
import json
import argparse
from collections import defaultdict, Counter


def count_words(text: str) -> int:
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", text or "")
    english_words = re.findall(r"\b[a-zA-Z]+\b", text or "")
    return len(chinese_characters) + len(english_words)


def extract_style(prompt: str) -> str:
    if not prompt:
        return None
    lines = [l.strip() for l in str(prompt).splitlines()]
    for l in lines:
        if l.startswith("写作风格"):
            m = re.search(r"写作风格\s*:\s*(.+)", l)
            if m:
                return m.group(1).strip()
    return None


def extract_target_word_count(prompt: str) -> int:
    if not prompt:
        return None
    m = re.search(r"总字数\s*:\s*(\d+)", str(prompt))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def super_style(style: str) -> str:
    if not style:
        return "Other"
    s = style.lower()
    if "academic" in s:
        return "Academic"
    if "finance" in s or "business" in s:
        return "Business"
    if "functional" in s:
        return "Functional"
    if "news" in s:
        return "News"
    if "literature" in s or "creative" in s:
        return "Creative"
    if "popular science" in s or "science" in s:
        return "PopularScience"
    if "education" in s or "teaching" in s:
        return "Education"
    return "Other"


def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"\b[a-zA-Z]+\b", text))
    total = zh + en
    if total == 0:
        return "unknown"
    ratio = zh / total
    if ratio >= 0.7:
        return "zh"
    if ratio <= 0.3:
        return "en"
    return "mixed"


def length_bin(n: int, bins=(5000, 10000, 15000, 20000, 30000)) -> str:
    if n < bins[0]:
        return "lt_5k"
    if bins[0] <= n < bins[1]:
        return "5k_10k"
    if bins[1] <= n < bins[2]:
        return "10k_15k"
    if bins[2] <= n < bins[3]:
        return "15k_20k"
    if bins[3] <= n <= bins[4]:
        return "20k_30k"
    return "gt_30k"


def stream_jsonl(fp):
    with open(fp, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except Exception:
                continue


def write_jsonl(path, objs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    augmented_path = os.path.join(output_dir, "augmented.jsonl")
    long_path = os.path.join(output_dir, "long_5k_30k.jsonl")
    stats_path = os.path.join(output_dir, "classified_stats.json")

    style_files = {}
    bin_files = {}

    counters = {
        "total": 0,
        "long_total": 0,
        "by_style": Counter(),
        "by_style_long": Counter(),
        "by_bin": Counter(),
    }

    augmented_out = open(augmented_path, "w", encoding="utf-8")
    long_out = open(long_path, "w", encoding="utf-8")

    for _, obj in stream_jsonl(input_path):
        prompt = obj.get("prompt", "")
        response = obj.get("response", "")
        style = extract_style(prompt)
        sstyle = super_style(style)
        target_wc = extract_target_word_count(prompt)
        length = count_words(response)
        lbin = length_bin(length)
        lang = detect_lang(response)

        augmented = {
            "prompt": prompt,
            "response": response,
            "style": style,
            "super_style": sstyle,
            "target_word_count": target_wc,
            "length_count": length,
            "length_bin": lbin,
            "lang": lang,
        }

        augmented_out.write(json.dumps(augmented, ensure_ascii=False) + "\n")
        counters["total"] += 1
        counters["by_style"][sstyle] += 1
        counters["by_bin"][lbin] += 1

        is_long = 5000 <= length <= 30000
        if is_long:
            long_out.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            counters["long_total"] += 1
            counters["by_style_long"][sstyle] += 1

            if sstyle not in style_files:
                fp = os.path.join(output_dir, f"long_{sstyle}.jsonl")
                style_files[sstyle] = open(fp, "w", encoding="utf-8")
            style_files[sstyle].write(json.dumps(augmented, ensure_ascii=False) + "\n")

            sub_bin = length_bin(length)
            if sub_bin in {"5k_10k", "10k_15k", "15k_20k", "20k_30k"}:
                if sub_bin not in bin_files:
                    fp = os.path.join(output_dir, f"long_{sub_bin}.jsonl")
                    bin_files[sub_bin] = open(fp, "w", encoding="utf-8")
                bin_files[sub_bin].write(json.dumps(augmented, ensure_ascii=False) + "\n")

    augmented_out.close()
    long_out.close()
    for f in style_files.values():
        f.close()
    for f in bin_files.values():
        f.close()

    stats = {
        "total": counters["total"],
        "long_total": counters["long_total"],
        "by_style": dict(counters["by_style"]),
        "by_style_long": dict(counters["by_style_long"]),
        "by_bin": dict(counters["by_bin"]),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/qwen/graph_writer/merged.jsonl")
    p.add_argument("--output_dir", default="data/qwen/graph_writer/classified")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output_dir)