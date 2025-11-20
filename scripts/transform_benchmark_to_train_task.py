import json
import re
import argparse

def merge_type(d1: str, d2: str) -> str:
    d1 = (d1 or '').strip()
    d2 = (d2 or '').strip()
    if d1 and d2:
        return f"{d1} - {d2}"
    return d1 or d2

def extract_length(text: str):
    if not text:
        return None
    t = text
    rflags = re.IGNORECASE
    m = re.search(r"(?<!\d)(\d{2,6})\s*(?:-|~|～|—|–|至|到|to)\s*(\d{2,6})\s*(?:字|词|words?|characters?)", t, rflags)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return max(a, b)
    m = re.search(r"(?:约|大约|左右|around|approximately|about)\s*(\d{2,6})\s*(?:字|词|words?|characters?)", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:至少|不?少于|不?低于|不?小于|at\s*least|no\s*less\s*than|not\s*less\s*than)\s*(\d{2,6})\s*(?:字|词|words?|characters?)", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:不超过|至多|最多|不?高于|no\s*more\s*than|at\s*most|not\s*more\s*than)\s*(\d{2,6})\s*(?:字|词|words?|characters?)", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"字数控制在\s*(\d{2,6})\s*字(?:左右)?", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"字数[^\d]*(\d{2,6})\s*字", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<!\d)(\d{2,6})\s*(?:字|词|words?|characters?)(?:左右|以内|以上|之间|上下)?", t, rflags)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<!\d)(\d{2,6})-word(?:s)?\b", t, rflags)
    if m:
        return int(m.group(1))
    return None

def process(input_path: str, output_path: str, default_length: int = 600):
    added = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'a', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get('query', '')
            type_str = merge_type(obj.get('domain1', ''), obj.get('domain2', ''))
            length = extract_length(prompt)
            rec = {"prompt": prompt, "type": type_str, "length": length if length is not None else default_length}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1
    print(f"Appended {added} records to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--default_length', type=int, default=5000)
    args = parser.parse_args()
    process(args.input, args.output, args.default_length)

if __name__ == '__main__':
    main()