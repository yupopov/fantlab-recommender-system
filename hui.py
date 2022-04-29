import asyncio
import aiohttp

# async def main():
#     print("Sleep now.")
#     await asyncio.sleep(1.5)
#     print("OK, wake up!")
#     return 1

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main(urls=['https://api.fantlab.ru/work/1']):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
          *[fetch(session, url) for url in urls]
        )
        with open('hui.txt', 'w') as f:
            f.write(str(results))

asyncio.run(main())