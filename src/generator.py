import asyncio
import logging
from . import config as cfg
from .io_utils import append_chapter_jsonl, load_generated_chapters

with open("prompts/chapter.txt", "r") as f:
    chapter_prompt = f.read()
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
async def async_generate_chapter(semaphore, record, previous_contents):
    async with semaphore:
        logger.info(
            "start generate_chapter title=%s prev_len=%d",
            record['title'], len(previous_contents)
        )

        # 构造 prompt 文本
        prompt = chapter_prompt.format(
            title=record['title'],
            subtitles=', '.join(record.get('subtitles', [])),
            notes=record.get('notes', ''),
            length_limit=record.get('length_limit', 10000),
            style=record.get('style', ''),
            previous_contents=previous_contents
        )

        logger.info("request model=%s", cfg.resolve_model_id())

        r = await cfg.openai_async_client.chat.completions.create(
            model=cfg.resolve_model_id(),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的文章生成助手，擅长创作符合要求的文章。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4096,
            temperature=0.6,
            top_p=0.9,
            # extra_body={"repetition_penalty": 1.2},
        )

        content = r.choices[0].message.content

        logger.info(
            "end generate_chapter title=%s content_len=%d",
            record['title'], len(content)
        )
        return content

async def generate_layers(records, levels):
    generated_chapters = load_generated_chapters(cfg.GENERATED_JSONL)
    semaphore = asyncio.Semaphore(cfg.MAX_CONCURRENT_GPT_CALLS)
    previous_layer_contents = ""
    for layer_idx, level in enumerate(levels):
        logger.info("start layer idx=%d size=%d", layer_idx, len(level))
        missing_ids = [nid for nid in level if nid not in generated_chapters]
        tasks = [
            async_generate_chapter(semaphore, records[nid], previous_layer_contents)
            for nid in missing_ids
        ]
        if tasks:
            layer_results = await asyncio.gather(*tasks)
            for nid, content in zip(missing_ids, layer_results):
                append_chapter_jsonl(nid, records[nid]["title"], content, jsonl_file=cfg.GENERATED_JSONL)
                generated_chapters[nid] = {"title": records[nid]["title"], "content": content}
        previous_layer_contents = ""
        for nid in level:
            content = generated_chapters.get(nid, {}).get("content", "")
            title = records[nid]["title"]
            previous_layer_contents += f"\n\n【{title}】\n{content}\n"
        logger.info("end layer idx=%d written=%d file=%s", layer_idx, len(level), cfg.GENERATED_JSONL)
    return generated_chapters