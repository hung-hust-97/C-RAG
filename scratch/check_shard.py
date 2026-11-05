import asyncio
from lightrag.kg.postgres_impl import ClientManager

async def main():
    db = await ClientManager.get_client()
    res = await db.query("""
        SELECT 
            COUNT(DISTINCT run_id) as total_distinct_runs,
            COUNT(*) FILTER (WHERE shard_index = 0) as shard_0_count
        FROM lightrag_pipeline_stage_runs
    """, [])
    print(dict(res))

if __name__ == "__main__":
    asyncio.run(main())

