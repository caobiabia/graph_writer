import os
import datetime
import requests
try:
    from openai import OpenAI, AsyncOpenAI
except Exception:
    OpenAI = None
    AsyncOpenAI = None

EMBED_API_URL = "http://127.0.0.1:21002/v1/embeddings"
EMBED_MODEL = "/data/home/Yanchu/llm_repo/Qwen3-Embedding-4B"
EMBED_TOKEN = "sk-dhtlheweglbnekhitzcnoaigaeuxxlozvihrkbkrimbchtze"

GPT_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GPT_MODEL = "/data/home/Yanchu/llm_repo/Qwen2.5-72B-Instruct"
GPT_API_URL = "http://127.0.0.1:21001/v1/completions"
GPT_HEADERS = {"Content-Type": "application/json"}
if GPT_API_KEY:
    GPT_HEADERS["Authorization"] = f"Bearer {GPT_API_KEY}"
_gpt_base = GPT_API_URL.split("/v1/")[0] if "/v1/" in GPT_API_URL else GPT_API_URL
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or (_gpt_base + "/v1" if not _gpt_base.endswith("/v1") else _gpt_base)
_embed_base = EMBED_API_URL.split("/v1/")[0] if "/v1/" in EMBED_API_URL else EMBED_API_URL
OPENAI_EMBED_BASE_URL = os.getenv("OPENAI_EMBED_BASE_URL", "").strip() or (_embed_base + "/v1" if not _embed_base.endswith("/v1") else _embed_base)

def _build_openai_clients():
    if AsyncOpenAI is None or OpenAI is None:
        return None, None
    if OPENAI_BASE_URL:
        return OpenAI(api_key=GPT_API_KEY, base_url=OPENAI_BASE_URL), AsyncOpenAI(api_key=GPT_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=GPT_API_KEY), AsyncOpenAI(api_key=GPT_API_KEY)

openai_client, openai_async_client = _build_openai_clients()

def _build_embed_client():
    if OpenAI is None:
        return None
    if OPENAI_EMBED_BASE_URL:
        return OpenAI(api_key=EMBED_TOKEN, base_url=OPENAI_EMBED_BASE_URL)
    return OpenAI(api_key=EMBED_TOKEN)

openai_embed_client = _build_embed_client()

_RESOLVED_MODEL_ID = None

def resolve_model_id():
    global _RESOLVED_MODEL_ID
    if _RESOLVED_MODEL_ID:
        return _RESOLVED_MODEL_ID
    try:
        base = GPT_API_URL.split("/v1/")[0]
        r = requests.get(f"{base}/v1/models", headers=GPT_HEADERS)
        r.raise_for_status()
        data = r.json()
        target = None
        for m in data.get("data", []):
            mid = m.get("id", "")
            root = m.get("root", "")
            if mid == GPT_MODEL or root == GPT_MODEL or mid.endswith(GPT_MODEL) or root.endswith(GPT_MODEL) or GPT_MODEL in mid or GPT_MODEL in root:
                target = mid
                break
        if not target and data.get("data"):
            target = data["data"][0].get("id", GPT_MODEL)
        _RESOLVED_MODEL_ID = target or GPT_MODEL
        return _RESOLVED_MODEL_ID
    except Exception:
        _RESOLVED_MODEL_ID = GPT_MODEL
        return _RESOLVED_MODEL_ID

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
JSONL_FILE = f"data/task/task_planning_{timestamp}.jsonl"
GENERATED_JSONL = f"data/original_content/output_{timestamp}.jsonl"
REFINED_JSONL = f"data/refined_content/refined_{timestamp}.jsonl"

SIM_THRESHOLD = 0.4
MAX_CONCURRENT_GPT_CALLS = 10

def set_run_file_names(ts: str, run_idx: int):
    global JSONL_FILE, GENERATED_JSONL, REFINED_JSONL
    JSONL_FILE = f"data/task/task_planning_{ts}_{run_idx}.jsonl"
    GENERATED_JSONL = f"data/original_content/output_{ts}_{run_idx}.jsonl"
    REFINED_JSONL = f"data/refined_content/refined_{ts}_{run_idx}.jsonl"

def set_file_names(jsonl_file: str, generated_jsonl: str, refined_jsonl: str):
    global JSONL_FILE, GENERATED_JSONL, REFINED_JSONL
    JSONL_FILE = jsonl_file
    GENERATED_JSONL = generated_jsonl
    REFINED_JSONL = refined_jsonl

def get_file_names():
    return JSONL_FILE, GENERATED_JSONL, REFINED_JSONL