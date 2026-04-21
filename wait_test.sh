echo "Waiting for build and container..."
while true; do
  if docker ps | grep "c-rag-lightrag-1" | grep -q "healthy"; then
    echo "API is healthy!"
    break
  fi
  sleep 5
done
echo "Running test..."
.venv/bin/python test_upload_api.py
