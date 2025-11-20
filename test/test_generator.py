import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.generator as gen

class R:
    def __init__(self, text):
        self._text = text
    def raise_for_status(self):
        pass
    async def json(self):
        return {"choices": [{"text": self._text}]}

class Session:
    def __init__(self, text):
        self._text = text
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    def post(self, url, headers=None, json=None):
        async def gen_ctx():
            return R(self._text)
        class Ctx:
            async def __aenter__(self_inner):
                return await gen_ctx()
            async def __aexit__(self_inner, exc_type, exc, tb):
                pass
        return Ctx()

def prepare_fs():
    import os
    os.makedirs("data/original_content", exist_ok=True)

async def test_generate_layers():
    prepare_fs()
    gen.aiohttp = type("aio", (), {})()
    gen.aiohttp.ClientSession = lambda: Session("content")
    records = [
        {"id": 0, "title": "A", "subtitles": [], "notes": "", "length_limit": 10, "style": "s"},
        {"id": 1, "title": "B", "subtitles": [], "notes": "", "length_limit": 10, "style": "s"}
    ]
    levels = [[0], [1]]
    res = await gen.generate_layers(records, levels)
    if set(res.keys()) != {0,1}:
        raise AssertionError("generate_layers failed")

def main():
    asyncio.run(test_generate_layers())
    print("test_generator passed")

if __name__ == "__main__":
    main()