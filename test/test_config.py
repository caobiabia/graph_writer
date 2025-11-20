import os
import sys
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as cfg

class R:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self._data

def test_resolve_model_id():
    data = {"data": [{"id": "modelA"}, {"id": "modelB", "root": "rootB"}]}
    cfg.requests.get = lambda url, headers=None: R(data)
    mid = cfg.resolve_model_id()
    if not isinstance(mid, str) or len(mid) == 0:
        raise AssertionError("resolve_model_id failed")

def test_file_names():
    os.makedirs("data/task", exist_ok=True)
    os.makedirs("data/original_content", exist_ok=True)
    os.makedirs("data/refined_content", exist_ok=True)
    j,g,r = cfg.get_file_names()
    ts = "unittest"
    cfg.set_run_file_names(ts, 1)
    if "unittest_1" not in cfg.JSONL_FILE:
        raise AssertionError("set_run_file_names failed for JSONL_FILE")
    if "unittest_1" not in cfg.GENERATED_JSONL:
        raise AssertionError("set_run_file_names failed for GENERATED_JSONL")
    if "unittest_1" not in cfg.REFINED_JSONL:
        raise AssertionError("set_run_file_names failed for REFINED_JSONL")
    cfg.set_file_names(j,g,r)

if __name__ == "__main__":
    test_resolve_model_id()
    test_file_names()
    print("test_config passed")