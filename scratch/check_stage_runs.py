import asyncio
from lightrag.kg.postgres_impl import ClientManager

async def main():
    db = await ClientManager.get_client()
    res = await db.query("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(*) FILTER (WHERE event_json->>'doc_type' IS NOT NULL) as has_event_doc_type,
            COUNT(*) FILTER (WHERE event_json->'metadata'->>'doc_type' IS NOT NULL) as has_meta_doc_type
        FROM lightrag_pipeline_stage_runs
    """, [])
    print(dict(res))

if __name__ == "__main__":
    asyncio.run(main())

