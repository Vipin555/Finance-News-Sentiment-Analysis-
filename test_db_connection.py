import asyncio
import os

import asyncpg
from dotenv import load_dotenv


async def main() -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    print("DATABASE_URL =", url)

    try:
        conn = await asyncpg.connect(url)
        await conn.close()
        print("connect: OK")
    except Exception as exc:
        print("connect: FAIL", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
