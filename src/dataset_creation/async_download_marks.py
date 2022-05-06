import argparse
import asyncio
import gzip
import json
import re

# from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
import aiohttp

parser = argparse.ArgumentParser(
  description='Download work infos with ids contained in the file')
parser.add_argument(
    'query_template',
    type=str,
    nargs='?',
    default='https://fantlab.ru/work{work_id}/details',
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
    'work_marks',
    type=str,
    nargs='?',
    default='data/raw/work_marks.json.gz',
    help='path to target file'
)

args = parser.parse_args()
with open(args.work_ids, 'r') as f:
    work_ids = f.read().split('\n')



async def fetch(session, work_id):
    await asyncio.sleep(5)
    url = args.query_template.format(work_id=work_id)
    async with session.get(url) as response:
        result = await response.text('utf-8')
        print(result[:110])
        marks_strs = result[result.find('marks = []'): result.find('</script><br><br><b>') - 1].split('\n')
        marks_dicts = [{'user_id': int(re.search(pattern=r'\.userid=\d+', string=marks_str).group(0).split('=')[-1]),
                        'mark': int(re.search(pattern=r'\.mark=\d{1,2}', string=marks_str).group(0).split('=')[-1]),
                        'date': re.search(pattern=r'\.date=".*?"', string=marks_str).group(0)[7:-1],
                        'work_id': work_id
                        } for marks_str in marks_strs]

    async with asyncio.Semaphore(10):
        return await marks_dicts


async def main():
    marks = []
    async with aiohttp.ClientSession() as session:
        # results = await asyncio.gather(
        #   *tqdm([fetch(session, url) for url in urls])
        # )
        results = await tqdm_asyncio.gather(
          *[fetch(session, work_id) for work_id in work_ids]
        )
        marks.extend(results)
        with gzip.open(args.work_marks, 'wt') as f:
            json.dump(marks, f)


asyncio.run(main())