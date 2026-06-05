# 1. Build local
cd lightrag_webui && npm run build

cd ..

# 2. Copy vào container nginx
docker cp lightrag/api/webui/. crag-dev-ui:/usr/share/nginx/html/webui/