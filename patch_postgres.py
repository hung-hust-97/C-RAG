import re

with open("lightrag/kg/postgres_impl.py", "r", encoding="utf-8") as f:
    content = f.read()

pattern = r'(\s*)for k, v in data\.items\(\):\n\s*# Remove timezone information, store utc time in db\n\s*created_at = parse_datetime\(v\.get\("created_at"\)\)'
replacement = r'''\1existing_rows = await self.get_by_ids(list(data.keys()))
\1existing_map = {row["id"]: row for row in existing_rows} if existing_rows else {}

\1for k, update_data in data.items():
\1    v = {
\1        "content_summary": "",
\1        "content_length": 0,
\1        "status": "UPLOADING",
\1        "file_path": "",
\1        "chunks_count": -1,
\1        "chunks_list": [],
\1        "content_hash": "",
\1        "metadata": {},
\1        "error_msg": ""
\1    }
\1    if k in existing_map:
\1        v.update(existing_map[k])
\1    v.update(update_data)

\1    # Remove timezone information, store utc time in db
\1    created_at = parse_datetime(v.get("created_at"))'''

content = re.sub(pattern, replacement, content)

with open("lightrag/kg/postgres_impl.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch done!")

