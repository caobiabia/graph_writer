
import asyncio
import sys
sys.path.insert(0, '.')
from src import planning

async def run():
    try:
        outline, task_planning = await planning.async_gen_task_planning_main(
            query="测试主题,两个章节",
            word_count=1000,
            style="学术风格",
        )
        print('OK outline :', outline)
        print('OK task :', task_planning)
    except Exception as e:
        print('ERR:', type(e).__name__, str(e))

asyncio.run(run())