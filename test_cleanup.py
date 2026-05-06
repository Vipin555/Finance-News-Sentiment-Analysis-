import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv('.env')

async def check():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])
    
    # Check foreign key cascade
    res = await conn.fetch("SELECT confdeltype FROM pg_constraint WHERE conname = 'ff_sentiment_results_calendar_event_id_fkey'")
    print(f"Delete type for fk: {res}")
    
    await conn.close()

asyncio.run(check())
