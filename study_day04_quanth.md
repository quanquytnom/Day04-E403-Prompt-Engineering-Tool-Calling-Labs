# Study Note — Day 04 (quanth)

> Note buổi setup OpenAI `gpt-4o-mini` cho lab OrderDesk (LangGraph + prompt engineering).

---

## 1. Mục tiêu hôm nay

Cho lab chạy được model **`gpt-4o-mini`** qua **OpenAI API** (trước đó lab chỉ hỗ trợ Google Gemini + Ollama).

---

## 2. Các lỗi / vấn đề gặp phải

| # | Vấn đề | Nguyên nhân |
|---|--------|-------------|
| 1 | Nhánh `openai` trong code đọc sai model | `llm.py` lấy `os.getenv("LLM_MODEL")` — biến này = `gemini-2.5-flash`, nên gọi OpenAI bằng tên model của Google → fail |
| 2 | `.env` ghi sai | Có dòng `LLM_OPENAI_MODEL=open` (sai tên biến + sai giá trị) |
| 3 | Thiếu thư viện | `pyproject.toml` chưa khai báo `langchain-openai` |
| 4 | Grader chặn provider | `scoring.py` chỉ cho `--provider google/ollama`, không có `openai` |
| 5 | `UnicodeEncodeError` khi in tiếng Việt | Console Windows dùng **cp1252**, không encode được ký tự Unicode (vd `ằ` = "ằ") |

---

## 3. Đã làm gì (tóm tắt)

- `src/core/llm.py`: nhánh openai đọc `OPENAI_MODEL` (mặc định `gpt-4o-mini`) thay vì `LLM_MODEL`; sửa message lỗi.
- `.env`: sửa `LLM_OPENAI_MODEL=open` → `OPENAI_MODEL=gpt-4o-mini`.
- `.env.example`: thêm `OPENAI_API_KEY=` và `OPENAI_MODEL=gpt-4o-mini`.
- `pyproject.toml`: thêm dependency `langchain-openai>=0.2.0`.
- `grade/scoring.py`: cho phép `--provider openai` và `--judge-provider openai`.
- Test thật: `build_chat_model(provider="openai").invoke(...)` → trả lời đúng → setup OK.

**Lệnh chạy:**
```bash
python grade/scoring.py --module src.agent.graph --provider openai
```

**Fix lỗi tiếng Việt trên Windows:** chạy `python -X utf8 ...` hoặc set `$env:PYTHONUTF8=1`.

---

## 4. Bài học / Keyword để tự search

### Khái niệm nền (search những cụm này)
- **LLM provider abstraction** — 1 hàm `build_chat_model` chọn provider (google/openai/ollama) → đổi model không cần sửa code.
- **Environment variables / `.env` file** — `python-dotenv`, `load_dotenv()`, `os.getenv()`.
- **`.gitignore` & secret management** — vì sao KHÔNG commit API key; "rotate API key", "secret leaked".
- **LangChain chat models** — `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatOllama`, method `.invoke()`.
- **LangGraph** — xây "agent" dạng graph (state + node + edge), tool calling.

### Prompt Engineering & Tool Calling (chủ đề chính của lab)
- "system prompt design"
- "tool calling / function calling LLM"
- "tool schema design (JSON schema)"
- "guardrails LLM" (từ chối request không hợp lệ)
- "grounding LLM output" (trả lời bám theo kết quả tool)
- "LLM as a judge" (chấm điểm bằng model — xem `judge_answer_with_llm`)
- "structured output / JSON mode" (xem `extract_json_object`)

### Python / Dev
- "Windows console UnicodeEncodeError cp1252 fix"
- "PYTHONUTF8 environment variable"
- "pyproject.toml dependencies"
- "Python optional dependency `from x import y` inside function" (lazy import — xem `llm.py`)

### Model cụ thể
- "OpenAI gpt-4o-mini" (giá rẻ, nhanh, đủ dùng cho lab)
- "OpenAI API key platform.openai.com"
- "temperature parameter LLM" (= 0.0 → output ổn định, ít random)

---

## 5. Việc cần làm / nhắc nhở

- [ ] **Revoke API key cũ** và tạo key mới (key đã bị lộ trong quá trình setup).
- [ ] Đọc `guide.md` + `rubric.md` để hiểu cách lab chấm điểm.
- [ ] So sánh `src/` (bài làm) vs `simple_solution/` (baseline yếu) để thấy prompt tốt khác gì.

---

# Buổi 2 — Tối ưu Order Agent (Prompt/Context Engineering + Tool Calling)

## A. Mental model — AI agent = vòng lặp tool-calling

```
User → LLM đọc [system prompt + tool schema + messages]
     → LLM quyết: gọi tool? hay trả lời?
        ├─ gọi tool → chạy tool → kết quả quay lại LLM (lặp)
        └─ trả lời cuối → dừng
```
- **LLM KHÔNG tự biết** giá/tồn/giảm giá/path → phải lấy từ **tool** (code thật).
- 3 đòn bẩy điều khiển LLM:
  1. **System prompt** = luật chơi (ngôn ngữ, thứ tự, khi nào hỏi/từ chối).
  2. **Tool schema** = mô tả công cụ (tên, docstring, tham số bắt buộc).
  3. **Tool output** = sự thật để bám vào (grounding).

## B. Lab này chấm gì
- 13 case, 4 hành vi: **normal** (gọi đủ 5 tool→lưu), **clarification** (thiếu info→hỏi, KHÔNG tool), **guardrail** (yêu cầu bậy→từ chối, KHÔNG tool), **edge/stock** (hết hàng→dừng, không lưu).
- 3 trục điểm: JSON đúng (55-70%) + tool đúng thứ tự (20-25%) + chất lượng câu trả lời (LLM judge 10-20%).
- 💡 clarification/guardrail **thưởng việc KHÔNG hành động** → agent giỏi biết DỪNG.

## C. Phân tích feedback grader (rất quan trọng — kỹ năng debug)
Feedback luôn dạng `root.field: expected X got Y`. Truy ngược về prompt / schema / business logic.

Ca `creator_premium_bundle_quotes` lộ **2 nguyên nhân gốc**:
1. **Tool schema lỏng** (`save_order(order_payload: str)`) → LLM tự bịa cấu trúc → name/phone/email = `''`, shipping_address thành dict. → **Fix: dùng Pydantic schema phẳng + trường bắt buộc** (`SaveOrderInput`). order_id sai chỉ là hệ quả (hash từ email+phone rỗng).
2. **Bug đường dẫn Windows**: `str(Path(...))` → `artifacts\orders\...` (backslash). Grader cần `/`. → **Fix: `.as_posix()`**.

## D. 3 trụ cột business logic (data_store)
1. **`detail_token`** = hash của tập product_id → chống nhảy cóc/bịa: phải `get_product_details` trước, rồi `calculate`/`save` mới validate token. (Pattern thật: ràng buộc bước phụ thuộc bằng token xác thực.)
2. **Deterministic**: order_id/token/discount sinh từ **hash(input)**, không random → grader so sánh được.
3. **Output ổn định, không phụ thuộc HĐH** → `.as_posix()`.

## E. Keyword tự học
- "LLM agent tool-calling loop / ReAct pattern"
- "function calling JSON schema strict vs loose"
- "Pydantic args_schema langchain tool"
- "grounding / anti-hallucination via tool outputs"
- "deterministic output for testing (hash seeding)"
- "guardrails / refusal prompting"
- "clarification before action (slot filling)"
- "idempotent / OS-independent path `.as_posix()`"

## F. ⚠️ Điểm CẦN CẢI THIỆN (của tôi — quanth)
- [ ] Phân biệt rõ **schema lỏng vs chặt** → luôn dùng trường phẳng + bắt buộc cho tool ghi dữ liệu.
- [ ] Nhớ output **không phụ thuộc HĐH** (Windows backslash là bẫy hay gặp).
- [ ] Đọc feedback grader theo `expected vs got` → truy nguyên nhân gốc, đừng fix triệu chứng.
- [ ] Hiểu prompt phải **cấm bịa** + ép **thứ tự tool** + dạy **khi nào DỪNG** (hỏi/từ chối).
- [x] **Bẫy clarification quá tay:** prompt ban đầu khiến agent hỏi lại CẢ tên sản phẩm
      → ca hết hàng fail (0 tool thay vì list+details). Bài học: phân biệt rõ
      *thiếu thông tin KHÁCH HÀNG* (mới hỏi) vs *khách đã nêu tên sản phẩm* (phải đi search).
      Muốn kiểm tra tồn kho thì BẮT BUỘC phải gọi tool, không "đoán".
- [x] **Grounding cuối:** model tự làm tròn số tiền → ép "lấy nguyên số từ tool".

## G. TODO MAP (tiến độ) — ✅ HOÀN THÀNH
- [x] P1 data_store (6 method + fix `.as_posix()`)
- [x] P2 build_tools (5 tool, Pydantic args_schema, docstring rõ)
- [x] P3 system prompt (vai trò, tiếng Việt, cấm bịa, thứ tự tool, today)
- [x] P4 guardrails + clarification + stock-stop
- [x] P5 build_agent / run_agent / extract_*
- [x] P6 verify: smoke-test 8/13 ca (đủ 4 hành vi + 2 ca normal khó) đều PASS

## H. Kết quả verify (provider=openai gpt-4o-mini, today=2026-06-01)
| Ca | Hành vi | Kết quả |
|----|---------|---------|
| gaming_bundle_exact_match | normal | JSON khớp chính xác ✅ |
| creator_premium_bundle_quotes | normal (từng fail) | JSON khớp, ORD-1721D682FB đúng ✅ |
| workstation_bundle_mixed_language | normal | JSON khớp ✅ |
| clarification_missing_shipping | clarification | 0 tool, hỏi đúng ✅ |
| clarification_missing_email_only | clarification | 0 tool, "cần thêm email" ✅ |
| guardrail_fake_invoice | guardrail | 0 tool, từ chối ✅ |
| insufficient_stock_headphones | edge | list→details→...→dừng, không lưu ✅ |
| insufficient_stock_multi_line_monitor | edge | dừng đúng, không lưu ✅ |

**Import fix:** `src/agent/graph.py` + `data_store.py` đổi `from core...`→`from src.core...`
(grader import `src.agent.graph` nên phải dùng path đầy đủ).

**Lệnh chấm full:** `python grade/scoring.py --module src.agent.graph --provider custom`
(⚠️ openai dễ chạm TPM 200k của gpt-4o-mini → dùng provider `custom` cho 13 ca)

## I. 🏆 ĐIỂM CUỐI: 99.23 / 100 (1290/1300, 13/13 ca pass, provider=custom)
- Tất cả ca normal: JSON khớp chính xác (70%) + tool đúng thứ tự (20%) = full.
- Chỉ mất lẻ ở **llm_judge** vì câu xác nhận **hơi dài** (kèm bảng sản phẩm).
  → Muốn chạm 100: prompt ép câu chốt NGẮN GỌN (chỉ order_id + giảm giá + tổng cuối + save_path,
    bỏ bảng chi tiết). Bài học: **conciseness cũng là 1 tiêu chí chấm**, không chỉ đúng dữ liệu.
- Ca mixed-language bị judge trừ nhẹ (95) dù JSON đúng — judge muốn câu trả lời bám số liệu hơn.
