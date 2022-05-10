import argparse
import asyncio
import gzip
import json
from random import uniform

# from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
from aiohttp import ClientSession

parser = argparse.ArgumentParser(
  description='Download work infos with ids contained in the file')
parser.add_argument(
    'query_template',
    type=str,
    nargs='?',
    default='https://api.fantlab.ru/work/{work_id}/extended',
    help='a template query'
)
parser.add_argument(
    'work_ids',
    type=str,
    nargs='?',
    default='data/raw/work_ids.txt',
    help='path to work ids file', 
  )
parser.add_argument(
    'work_infos',
    type=str,
    nargs='?',
    default='data/raw/work_infos.json.gz',
    help='path to target file'
)

args = parser.parse_args()
with open(args.work_ids, 'r') as f:
    work_ids = f.read().split('\n')


async def fetch(session, url):
    sleep_time = uniform(2, 4)
    await asyncio.sleep(sleep_time)
    async with session.get(url) as response:
        return await response.json()


async def fetch_with_sem(session, url, sem):
    async with sem:
        return await fetch(session, url)

async def main():
    urls = [args.query_template.format(work_id=work_id) for work_id in work_ids]
    print('Downloading data from fantlab api...')
    # limiting number of concurrent connections
    # to be gentle with fantlab api
    sem = asyncio.Semaphore(400) 
    async with ClientSession() as session:
        # results = await asyncio.gather(
        #   *tqdm([fetch(session, url) for url in urls])
        # )
        results = await tqdm_asyncio.gather(
          *[fetch_with_sem(session, url, sem) for url in urls]
        )
    print('Saving results to file...')
    with gzip.open(args.work_infos, 'wt') as f:
        json.dump(results, f)
    print('Done.')


asyncio.run(main())