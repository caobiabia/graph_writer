import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.planning as planning

class R:
    def __init__(self, text):
        self._text = text
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"text": self._text}]}

def test_generate_outline_and_planning():
    planning.requests.post = lambda url, headers=None, json=None: R("# 一级标题\n(概要)\n## 二级\n(概要)")
    outline = planning.generate_outline("t", 2000, "学术手册")
    if "一级标题" not in outline:
        raise AssertionError("generate_outline failed")
    sample_blocks = [
        {
            "article_title": "AT",
            "title": "第一章",
            "subtitles": ["a", "b"],
            "length_limit": 1000,
            "style": "学术手册",
            "notes": "n"
        }
    ]
    sample_json = json.dumps(sample_blocks, ensure_ascii=False)
    planning.requests.post = lambda url, headers=None, json=None: R(sample_json)
    res = planning.generate_task_planning("t", outline, 2000, "学术手册", save_to_jsonl=False)
    parsed = json.loads(res)
    if not isinstance(parsed, list) or not parsed:
        raise AssertionError("generate_task_planning failed")

def test_sanitize_json_output():
    blocks = json.dumps([{ "title": "t", "subtitles": "a,b", "length_limit": 10, "style": "s", "notes": "n" }], ensure_ascii=False)
    sanitized = planning.sanitize_json_output(blocks, 100, "s")
    arr = json.loads(sanitized)
    if arr[0]["length_limit"] <= 0:
        raise AssertionError("sanitize_json_output scaling failed")

if __name__ == "__main__":
    test_generate_outline_and_planning()
    test_sanitize_json_output()
    print("test_planning passed")