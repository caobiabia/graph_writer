import json
import asyncio
import logging
from .config import resolve_model_id, openai_async_client

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

with open("prompts/judge.txt", "r") as f:
    judge_prompt = f.read()

def _balanced_json(s):
    i = s.find("{")
    while i != -1:
        depth = 0
        in_str = False
        esc = False
        out = []
        for c in s[i:]:
            out.append(c)
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            else:
                if c == '"':
                    in_str = True
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return "".join(out)
        i = s.find("{", i + 1)
    return None

def _parse_direction_result(text):
    s = (text or "").strip()
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        obj = _balanced_json(s)
        if obj:
            try:
                return json.loads(obj)
            except Exception:
                pass
        dm = re.search(r"direction\s*[:=]\s*\"?([A-Za-z_]+)\"?", s)
        rm = re.search(r"reason\s*[:=]\s*\"([\s\S]*?)\"", s)
        return {"direction": dm.group(1) if dm else "none", "reason": rm.group(1).strip() if rm else s[:200]}

async def async_gpt_judge_direction(semaphore: asyncio.Semaphore, i, j, records, sim):
    title_i, summary_i = records[i]["title"], records[i].get("summary", "")
    title_j, summary_j = records[j]["title"], records[j].get("summary", "")

    async with semaphore:
        logger.info(
            "judge start i=%d j=%d sim=%.4f title_i=%s title_j=%s summary_i=%s summary_j=%s",
            i, j, sim, title_i, title_j, summary_i, summary_j
        )

        prompt = judge_prompt.format(
            title_i=title_i,
            summary_i=summary_i,
            title_j=title_j,
            summary_j=summary_j,
            sim=sim
        )

        logger.info("judge request model=%s", resolve_model_id())

        try:
            r = await openai_async_client.chat.completions.create(
                model=resolve_model_id(),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名严格的结构判定助手，任务是判断两个内容之间的逻辑方向关系。"
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1,
                top_p=0.5,
                extra_body={"repetition_penalty": 1.1}
            )

            content = r.choices[0].message.content
            logger.info(
                "judge response content=%s",
                (content or "").replace("\n", " ")[:200]
            )

            result = _parse_direction_result(content)
            direction = result.get("direction", "none")
            reason = result.get("reason", "")

            logger.info(
                "judge end i=%d j=%d dir=%s reason=%s",
                i, j, direction, reason
            )
            return i, j, sim, direction, reason

        except Exception as e:
            logger.warning("judge error i=%d j=%d err=%s", i, j, str(e))
            return i, j, sim, "none", str(e)

async def async_build_graph(records, similarities, max_concurrent_calls: int):
    import networkx as nx
    G = nx.DiGraph()
    for r in records:
        G.add_node(r["id"], title=r["title"], summary=r.get("summary", ""))
    semaphore = asyncio.Semaphore(max_concurrent_calls)
    logger.info("graph build start nodes=%d edges_to_judge=%d", len(records), len(similarities))
    tasks = [async_gpt_judge_direction(semaphore, i, j, records, sim) for sim, i, j in similarities]
    results = await asyncio.gather(*tasks)
    for i, j, sim, direction, reason in results:
        if direction == "i_to_j":
            G.add_edge(i, j, weight=sim, reason=reason)
        elif direction == "j_to_i":
            G.add_edge(j, i, weight=sim, reason=reason)
        elif direction == "both":
            G.add_edge(i, j, weight=sim, reason=reason)
            G.add_edge(j, i, weight=sim, reason=reason)
    logger.info("graph build end nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G, results