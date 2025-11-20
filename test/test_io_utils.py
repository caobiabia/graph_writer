import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.io_utils as io

def test_io_roundtrip():
    os.makedirs("data/tmp", exist_ok=True)
    fp = "data/tmp/test.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"title":"T","subtitles":["a"],"notes":"n","length_limit":10,"style":"s"}, ensure_ascii=False)+"\n")
        f.write(json.dumps({"title":"U","subtitles":[],"notes":"m","length_limit":5,"style":"s"}, ensure_ascii=False)+"\n")
    recs = io.load_chapters(fp)
    if len(recs) != 2 or recs[0]["title"] != "T":
        raise AssertionError("load_chapters failed")
    out = "data/tmp/out.jsonl"
    io.save_all_chapters({0:{"title":"T","content":"c"},1:{"title":"U","content":"d"}}, out)
    got = io.load_generated_chapters(out)
    if got[1]["content"] != "d":
        raise AssertionError("save/load generated failed")

if __name__ == "__main__":
    test_io_roundtrip()
    print("test_io_utils passed")