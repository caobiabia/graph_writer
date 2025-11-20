import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.judger as judger

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
            return R(self._text)
        class Ctx:
            async def __aenter__(self_inner):
                return await gen()
            async def __aexit__(self_inner, exc_type, exc, tb):
                pass
        return Ctx()

async def test_async_build_graph():
    records = [
        {"id": 0, "title": "A", "summary": "x"},
        {"id": 1, "title": "B", "summary": "y"}
    ]
    sims = [(0.9, 0, 1)]
    judger.aiohttp = type("aio", (), {})()
    judger.aiohttp.ClientSession = lambda: Session('{"direction":"i_to_j","reason":"r"}')
    G, results = await judger.async_build_graph(records, sims, 2)
    if G.number_of_edges() != 1:
        raise AssertionError("async_build_graph failed")

def main():
    asyncio.run(test_async_build_graph())
    print("test_judger passed")

if __name__ == "__main__":
    main()