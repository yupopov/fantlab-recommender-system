import argparse
import asyncio
import gzip
import json

# from tqdm.auto import tqdm
from tqdm.asyncio import tqdm
import aiohttp

# async def main():
#     print("Sleep now.")
#     await asyncio.sleep(1.5)
#     print("OK, wake up!")
#     return 1

parser = argparse.ArgumentParser(
  description='Download work infos with ids contained in the file')
parser.add_argument(
  'work_ids', type=str, nargs=1, help='path to work ids file'
  )
parser.add_argument(
  'work_infos', type=str, nargs=1, help='path to target file'
)

args = parser.parse_args()
with open(args.work_ids[0], 'r') as f:
    work_ids = f.read().split('\n')


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main():
    urls = [f'https://api.fantlab.ru/work/{work_id}/extended' for work_id in work_ids]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
          *tqdm([fetch(session, url) for url in urls])
        )
        with gzip.open(args.work_infos[0], 'wt') as f:
            json.dump(results, f)

asyncio.run(main())

# for _ in tqdm(range(1000000000)):
#     pass