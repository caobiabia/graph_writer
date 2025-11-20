import numpy as np
from itertools import combinations
from .config import EMBED_MODEL, SIM_THRESHOLD, openai_embed_client

def gen_embed(text: str) -> np.ndarray:
    r = openai_embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding, dtype=np.float32)

def compute_similarities(records, threshold=SIM_THRESHOLD):
    embeds = [gen_embed(r["title"] + " " + r.get("summary", "")) for r in records]
    embeds = np.array(embeds)
    embeds_norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    similarities = []
    for i, j in combinations(range(len(records)), 2):
        sim = float(embeds_norm[i] @ embeds_norm[j])
        if sim > threshold:
            similarities.append((sim, i, j))
    similarities.sort(reverse=True)
    return similarities

def _pairwise_cosines(embeds):
    embeds = np.array(embeds)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeds_norm = embeds / norms
    sims = {}
    for i, j in combinations(range(len(embeds_norm)), 2):
        sims[(i, j)] = float(embeds_norm[i] @ embeds_norm[j])
    return sims

def _format_vec(v, k=5):
    v = np.asarray(v)
    head = ", ".join(f"{float(x):.4f}" for x in v[:k])
    return f"[{head}{'' if len(v) <= k else ', ...'}]"

if __name__ == "__main__":
    texts = [
        "机器学习基础",
        "深度学习基础",
        "今天天气很好"
    ]
    print("Embeddings API:", EMBED_API_URL)
    print("Embedding Model:", EMBED_MODEL)
    ok = True
    vectors = []
    for idx, t in enumerate(texts):
        try:
            vec = gen_embed(t)
            vectors.append(vec)
            print(f"Text#{idx}: {t}")
            print("dim:", vec.shape[0], "sample:", _format_vec(vec))
        except Exception as e:
            ok = False
            print(f"Text#{idx} embedding error:", str(e))
    if not ok:
        raise SystemExit(1)
    sims = _pairwise_cosines(vectors)
    for (i, j), s in sims.items():
        print(f"cosine({i},{j}) = {s:.4f}")
    records = [{"title": t, "summary": ""} for t in texts]
    filtered = compute_similarities(records)
    print("above-threshold pairs:")
    for sim, i, j in filtered:
        print(f"({i},{j}) sim={sim:.4f}")