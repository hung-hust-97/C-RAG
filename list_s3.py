import asyncio
import os
from dotenv import load_dotenv

# Load from .env
load_dotenv()
import aioboto3

async def list_files():
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")
    print(f"Connecting to {endpoint_url} bucket {bucket}")
    
    session = aioboto3.Session()
    try:
        async with session.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3:
            paginator = s3.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=bucket):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        print(obj['Key'])
                else:
                    print("No contents")
    except Exception as e:
        print("Error:", e)

asyncio.run(list_files())
