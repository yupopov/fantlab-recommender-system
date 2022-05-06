import argparse
import asyncio
import gzip
import json
import re
import csv
from itertools import chain
from zipfile import ZipFile

# from tqdm.auto import tqdm
import numpy as np
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
    default='data/raw/work_ids_cut.txt',
    help='path to work ids file',
  )
parser.add_argument(
    'work_marks',
    type=str,
    nargs='?',
    default='data/raw/work_marks.csv.gz',
    help='path to target file'
)

args = parser.parse_args()
with open(args.work_ids, 'r') as f:
    work_ids = f.read().split('\n')


def get_marks_from_html(html, work_id) -> list:
    marks_strs = html[html.find('marks = []'): html.find('</script><br><br><b>') - 1].split('\n')
    # mark_dicts = [
    #   {
    #     'user_id': int(re.search(pattern=r'\.userid=\d+', string=marks_str).group(0).split('=')[-1]),
    #     'mark': int(re.search(pattern=r'\.mark=\d{1,2}', string=marks_str).group(0).split('=')[-1]),
    #     'date': re.search(pattern=r'\.date=".*?"', string=marks_str).group(0)[7:-1],
    #     'work_id': work_id
    #   } for marks_str in marks_strs]
    marks = tuple((
        int(re.search(pattern=r'\.userid=\d+', string=marks_str).group(0).split('=')[-1]),
        int(re.search(pattern=r'\.mark=\d{1,2}', string=marks_str).group(0).split('=')[-1]),
        re.search(pattern=r'\.date=".*?"', string=marks_str).group(0)[7:-10],
    ) for marks_str in marks_strs)

    # return mark_dicts
    return marks

async def fetch(session, work_id):
    sleep_time = np.random.uniform(low=1, high=1.2)
    await asyncio.sleep(sleep_time)
    url = args.query_template.format(work_id=work_id)
    async with session.get(url) as response:
        result = await response.text('utf-8')

        mark_dicts = get_marks_from_html(result, work_id)
        # print(mark_dicts)
        return mark_dicts

    # async with asyncio.Semaphore(10):
    #     return await marks_dicts


async def fetch_with_sem(session, work_id, sem):
    async with sem:
        return await fetch(session, work_id)


async def main():
    sem = asyncio.Semaphore(4)
    async with aiohttp.ClientSession() as session:
        # results = await asyncio.gather(
        #   *tqdm([fetch(session, url) for url in urls])
        # )
        results = await tqdm_asyncio.gather(
          *[fetch_with_sem(session, work_id, sem) for work_id in work_ids]
        )

        results = list(chain.from_iterable(results)) # flattening the list
        print(type(results))
        print(len(results))
        
        # with gzip.open(args.work_marks, 'wt') as f:
        #       (results, f)
        with gzip.open(args.work_marks, 'wt') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['user_id', 'mark', 'date'])
            for row in results:
                csv_out.writerow(row)

asyncio.run(main())