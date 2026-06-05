# Hướng Dẫn Luồng Gọi vLLM Trong Quá Trình Trích Xuất Thực Thể (Entity Extraction)

Tài liệu này giải thích chi tiết luồng xử lý và cách hệ thống quản lý Token khi gửi request trích xuất thực thể (Entity Extraction) đến các server LLM tương thích chuẩn OpenAI (đặc biệt là **vLLM**).

Việc hiểu rõ luồng này sẽ giúp bạn cấu hình chính xác các tham số Token trong `.env` và khắc phục hiệu quả lỗi vượt quá ngữ cảnh (Context Length Exceeded).

---

## 1. Luồng Gọi API Trích Xuất Tiêu Chuẩn

Mỗi đoạn văn bản (chunk) sau khi được phân chia sẽ đi qua luồng trích xuất sau:

```mermaid
graph TD
    A[Bắt đầu Trích xuất] --> B[Tạo System & User Prompt]
    B --> C[Kiểm tra Ngữ cảnh (Pre-flight Check)]
    C -->|Vượt quá Context| D[Cắt nhỏ Chunk & Lặp lại]
    C -->|Hợp lệ| E[Gửi Request tới vLLM]
    E --> F[Nhận Response & Xử lý (Parse)]
    F --> G[Gleaning - Trích xuất bổ sung nếu cần]
```

---

## 2. Các Bước Xử Lý Chi Tiết

### Bước 1: Tạo Prompt
LightRAG sẽ gom các thông tin sau để tạo thành một Request gửi đi:
- **System Prompt**: Chứa định dạng bắt buộc, các loại thực thể mục tiêu (Entity Types) và các ví dụ (Few-shot examples).
- **User Prompt**: Nội dung thực tế của đoạn văn bản (chunk).

*Lưu ý:* Do System Prompt mang theo rất nhiều ví dụ mẫu, lượng Token thực tế gửi đi sẽ **đội lên rất nhiều** so với bản thân nội dung của file text.

### Bước 2: Kiểm Tra Ngữ Cảnh (Pre-flight Check)
Trước khi thực sự gửi Request sang vLLM, LightRAG sẽ nhẩm tính tổng số Token dự kiến bằng công thức:

> **Tổng Token Dự Kiến** = Token Đầu Vào + Token Dự Trữ Cho Output + Biên Độ An Toàn

Hệ thống sẽ đối chiếu `Tổng Token Dự Kiến` với giới hạn tối đa của Model được định nghĩa qua biến `LLM_EXTRACT_MODEL_CONTEXT_TOKENS`.
- Nếu **vượt quá**: LightRAG sẽ chủ động chặt đôi đoạn văn bản đó ra và quay lại Bước 1.
- Nếu **hợp lệ**: Request sẽ được gửi đi.

**Các biến môi trường liên quan:**
```env
LLM_EXTRACT_MODEL_CONTEXT_TOKENS=16384  # Tổng số token tối đa mà Model của bạn chịu được
LLM_EXTRACT_CONTEXT_SAFETY_MARGIN=512   # Biên độ an toàn để phòng hờ vLLM đếm lệch
ENTITY_EXTRACTION_MIN_OUTPUT_TOKENS=512 # Mức Output tối thiểu được giữ lại
OPENAI_LLM_MAX_TOKENS_EXTRACT=512       # Giới hạn Output gửi tới API
OPENAI_LLM_MAX_COMPLETION_TOKENS_EXTRACT=512
```

### Bước 3: Gleaning (Trích xuất bổ sung)
Nếu thuật toán đánh giá lần trích xuất đầu tiên bị sót dữ liệu, nó sẽ gọi vLLM thêm một lần nữa. 
Ở lần gọi này, Request phải cõng theo **toàn bộ lịch sử của lần gọi trước**. Do đó, nó cực kỳ tốn Token. Nếu hệ thống tính toán thấy Request này vượt quá giới hạn Context, nó sẽ báo cảnh báo (Warning) và tự động bỏ qua bước Gleaning này.

---

## 3. Luồng Xử Lý Riêng Cho Tài Liệu Pháp Lý (Legal Documents)

Tài liệu pháp lý (Luật, Nghị định) có một luồng quét bổ sung gọi là **Cross-Chunk Relation** để tìm các liên kết chồng chéo giữa các Điều/Khoản.

Luồng này **bỏ qua bước tính biên độ an toàn** và sử dụng một biến kiểm tra độc lập:

```env
MAX_EXTRACT_INPUT_TOKENS=15500
```

Hệ thống đã thiết lập sẵn mức an toàn cho Model 16K là `15500` để nhường đủ không gian ngữ cảnh cho Token đầu ra và độ lệch tokenizer. Mức thiết lập này là cố định và không được vượt quá để tránh lỗi Context Length.

---

## 4. Cách Khắc Phục Lỗi Context Của vLLM

Lỗi đặc trưng của vLLM thường có dạng:
> *This model's maximum context length is 16384 tokens. However, you requested 1024 output tokens and your prompt contains at least 15361 input tokens...*

Lỗi này xảy ra do bộ đếm Token nội bộ của LightRAG (thường dùng Tiktoken) đếm **ít hơn** bộ đếm thực tế của vLLM (do vLLM áp dụng Chat Template và Tokenizer riêng của model như Llama, Mistral, Qwen).

### Các bước khắc phục:
1. Luôn giữ 2 biến `OPENAI_LLM_MAX_TOKENS_EXTRACT` và `OPENAI_LLM_MAX_COMPLETION_TOKENS_EXTRACT` bằng nhau (Mức chuẩn hệ thống là `512`).
2. Tăng `LLM_EXTRACT_CONTEXT_SAFETY_MARGIN` (Mức chuẩn an toàn hiện tại là `512`) để LightRAG chừa lại nhiều khoảng trống dự phòng hơn trước khi gửi.
3. Nếu LLM vẫn báo lỗi do bộ đếm của nó quá lệch, bạn phải ép nhỏ lại kích thước phân mảnh bằng cách giảm `CHUNK_SIZE`.
4. Luôn nhớ **Restart lại Worker** sau khi sửa file `.env`.

