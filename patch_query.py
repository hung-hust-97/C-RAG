import sys
import re

with open("lightrag/api/routers/query_routes.py", "r") as f:
    code = f.read()

fallback_logic = """
            lang = current_rag.addon_params.get("language", "English") if hasattr(current_rag, "addon_params") else "English"
            mapping = {
                "Vietnamese": "Không tìm thấy dữ liệu ngữ cảnh nào phù hợp cho câu hỏi của bạn.",
                "French": "Aucun contexte pertinent trouvé pour la requête.",
                "Chinese": "未找到与查询相关的上下文。",
                "Japanese": "関連するコンテキストが見つかりませんでした。"
            }
            default_msg = mapping.get(lang, "No relevant context found for the query.")
            response_content = default_msg
"""

# We need to replace `response_content = "No relevant context found for the query."` with the logic.
# Notice the indentation!
# First instance:
#             if not response_content:
#                 response_content = "No relevant context found for the query."
code = re.sub(
    r'(?P<indent>[ \t]+)response_content = "No relevant context found for the query\."',
    lambda m: '\n'.join(m.group('indent') + line for line in fallback_logic.strip().split('\n')),
    code
)

with open("lightrag/api/routers/query_routes.py", "w") as f:
    f.write(code)

print("Patched successfully.")
