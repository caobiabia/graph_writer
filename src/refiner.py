import json
import asyncio
from .config import resolve_model_id, openai_async_client, MAX_CONCURRENT_GPT_CALLS

with open("prompts/refine.txt", "r") as f:
    refine_prompt = f.read()

async def async_refine_pair(semaphore: asyncio.Semaphore, i, j, text_i, text_j):
    async with semaphore:
        prompt = refine_prompt.format(i=i, j=j, text_i=text_i, text_j=text_j)

        try:
            # ============= 新版 Chat Completions 调用 =============
            r = await openai_async_client.chat.completions.create(
                model=resolve_model_id(),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8192,
                temperature=0.0,
                top_p=0.5,
                extra_body={"repetition_penalty": 1.1}
            )
            content = r.choices[0].message.content
            # ====================================================

            # JSON 解析保持你的旧逻辑
            try:
                res = json.loads(content)
            except Exception:
                import re
                m = re.search(r"\{.*\}", content, re.S)
                res = json.loads(m.group(0)) if m else {"A_new": None, "B_new": None}

            return i, j, res

        except Exception:
            return i, j, {"A_new": None, "B_new": None}

async def refine_pairs(both_pairs, generated_chapters):
    if not both_pairs:
        return generated_chapters
    sem = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
    refine_tasks = []
    for u, v in both_pairs:
        text_u = generated_chapters.get(u, {}).get("content", "")
        text_v = generated_chapters.get(v, {}).get("content", "")
        if not text_u and not text_v:
            continue
        refine_tasks.append(async_refine_pair(sem, u, v, text_u, text_v))
    refine_results = await asyncio.gather(*refine_tasks)
    for r_i, r_j, res in refine_results:
        A_new = res.get("A_new")
        B_new = res.get("B_new")
        if A_new and isinstance(A_new, str) and A_new.strip():
            generated_chapters[r_i]["content"] = A_new
        if B_new and isinstance(B_new, str) and B_new.strip():
            generated_chapters[r_j]["content"] = B_new
    return generated_chapters

