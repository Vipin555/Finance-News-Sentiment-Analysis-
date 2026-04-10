import asyncio
import asyncpg
import os

from dotenv import load_dotenv

load_dotenv(r"c:\Users\Lenovo\Downloads\news_sentiment_analysis\.env")

async def setup_db():
    print("Connecting to Supabase...")
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    print("Running base schema...")
    with open(r"c:\Users\Lenovo\Downloads\scraped_news_schema.sql", "r", encoding="utf-8") as f:
        await conn.execute(f.read())
        
    print("Running 001_sentiment.sql...")
    with open(r"c:\Users\Lenovo\Downloads\news_sentiment_analysis\001_sentiment.sql", "r", encoding="utf-8") as f:
        await conn.execute(f.read())
        
    print("Running 002_calendar_date.sql...")
    with open(r"c:\Users\Lenovo\Downloads\news_sentiment_analysis\002_calendar_date.sql", "r", encoding="utf-8") as f:
        await conn.execute(f.read())
        
    print("Setup complete!")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_db())
