import asyncio
from lightrag.kg.postgres_impl import ClientManager

async def main():
    db = await ClientManager.get_client()
    res = await db.query("""
        SELECT 
            COUNT(*) FILTER (WHERE (event_json->>'doc_type' IS NULL OR event_json->>'doc_type' != 'legal') 
                               AND (event_json->'metadata'->>'doc_type' IS NULL OR event_json->'metadata'->>'doc_type' != 'legal')
                               AND doc_id ~ '^[0-9]+$') as missing_doc_type_numeric
        FROM lightrag_pipeline_stage_runs
    """, [])
    print(dict(res))

if __name__ == "__main__":
    asyncio.run(main())

