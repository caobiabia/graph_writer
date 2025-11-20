import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.refiner as ref

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
        async def gen():
            return R('{"A_new":"A1","B_new":"B1"}')
        class Ctx:
            async def __aenter__(self_inner):
                return await gen()
            async def __aexit__(self_inner, exc_type, exc, tb):
                pass
        return Ctx()

async def test_refine_pairs():
    ref.aiohttp = type("aio", (), {})()
    ref.aiohttp.ClientSession = lambda: Session("ok")
    both_pairs = [(0,1)]
    gen_chapters = {0:{"title":"A","content":"a"},1:{"title":"B","content":"b"}}
    res = await ref.refine_pairs(both_pairs, gen_chapters)
    if res[0]["content"] != "A1" or res[1]["content"] != "B1":
        raise AssertionError("refine_pairs failed")

def main():
    asyncio.run(test_refine_pairs())
    print("test_refiner passed")

if __name__ == "__main__":
    main()