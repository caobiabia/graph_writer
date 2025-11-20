import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.embeddings as emb

class R:
    def __init__(self, vec):
        self._vec = vec
    def raise_for_status(self):
        pass
    def json(self):
        return {"data": [{"embedding": self._vec}]}

def test_gen_embed_and_similarities():
    emb.requests = type("req", (), {})()
    emb.requests.post = lambda url, json=None, headers=None: R([0.1]*8)
    records = [
        {"id": 0, "title": "A", "summary": "x"},
        {"id": 1, "title": "B", "summary": "y"},
        {"id": 2, "title": "C", "summary": "z"}
    ]
    v = emb.gen_embed("hello")
    if not isinstance(v, np.ndarray):
        raise AssertionError("gen_embed failed")
    sims = emb.compute_similarities(records, threshold=-1.0)
    if len(sims) < 3:
        raise AssertionError("compute_similarities failed")

if __name__ == "__main__":
    test_gen_embed_and_similarities()
    print("test_embeddings passed")