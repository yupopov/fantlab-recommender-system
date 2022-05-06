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
from aiohttp.web import HTTPException
from asyncio import CancelledError

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
parser.add_argument(
    'failed_work_ids',
    type=str,
    nargs='?',
    default='data/raw/failed_work_ids.txt',
    help='path to work ids file for which marks downloading failed'
)

args = parser.parse_args()
with open(args.work_ids, 'r') as f:
  work_ids = f.read().split('\n')
failed_work_ids = []

USER_AGENT = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36 OPR/72.0.3815.465 (Edition Yx GX)',
}

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
        work_id,
        int(re.search(pattern=r'\.mark=\d{1,2}', string=marks_str).group(0).split('=')[-1]),
        re.search(pattern=r'\.date=".*?"', string=marks_str).group(0)[7:-10],
    ) for marks_str in marks_strs)

    # return mark_dicts
    return marks

async def fetch(session, work_id):
    sleep_time = np.random.uniform(low=1, high=1.2)
    await asyncio.sleep(sleep_time)
    url = args.query_template.format(work_id=work_id)
    async with session.get(url, headers=USER_AGENT) as response:
        if response.status != 200:
            failed_work_ids.append(work_id)
            return
        result = await response.text('utf-8')

        mark_dicts = get_marks_from_html(result, work_id)
        # print(mark_dicts)
        return mark_dicts


async def fetch_with_sem(session, work_id, sem):
    async with sem:
        return await fetch(session, work_id)


async def main():
    attempt_num = 1 # counting attempts to download the htmls
    sem = asyncio.Semaphore(50)
    results = []
    global work_ids # ??? without this everything breaks gracefully
    global failed_work_ids # ???? understand why this happens
    async with aiohttp.ClientSession() as session:
        while work_ids and attempt_num < 10:
            print(f'Starting attempt {attempt_num}...')
            attempt_results = await tqdm_asyncio.gather(
                *[fetch_with_sem(session, work_id, sem) for work_id in work_ids],
                )

            # deleting failed responses
            attempt_results = [result for result in attempt_results if result is not None]
            print(f'Attempt {attempt_num}: Downloaded marks for {len(attempt_results)} works, Failed for {len(failed_work_ids)} works')
            
            attempt_results = list(chain.from_iterable(attempt_results)) # flattening the list
            results.extend(attempt_results)

            # trying to download failed responses again
            work_ids = failed_work_ids
            failed_work_ids = []
            attempt_num += 1
            if work_ids:
                await asyncio.sleep(5)

        # saving results to file
        with gzip.open(args.work_marks, 'wt') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['user_id', 'work_id', 'mark', 'date'])
            for row in results:
                csv_out.writerow(row)
        
        # if marks for some works still weren't downloaded,
        # save their work ids to a separate file
        if work_ids:
            work_ids = [str(work_id) for work_id in work_ids]
            with open(args.failed_work_ids, 'w') as f:
                f.write('\n'.join(work_ids))

# MWE for weird stuff we've seen above
# This works just fine
# async def hui():
#     print(work_ids)

# And this function can't see global variables
# Because it modifies them
# async def hui():
#     print(work_ids)
#     work_ids = []


asyncio.run(main())