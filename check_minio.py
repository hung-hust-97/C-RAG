import os
from dotenv import load_dotenv
load_dotenv()

import pipmaster as pm
if not pm.is_installed("boto3"):
    pm.install("boto3")
import boto3

endpoint_url = os.environ.get("S3_ENDPOINT_URL")
access_key = os.environ.get("S3_ACCESS_KEY")
secret_key = os.environ.get("S3_SECRET_KEY")
bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")

print(f"Connecting to {endpoint_url} bucket {bucket}")

s3 = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)
try:
    response = s3.list_objects_v2(Bucket=bucket)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("No contents")
except Exception as e:
    print("Error:", e)
