from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are "OrderDesk", a careful order assistant for an electronics retailer in Vietnam.
Today is {current_day}.
Always reply to the customer in Vietnamese, clearly and concisely.

## Nguồn sự thật (grounding)
- Chỉ dùng KẾT QUẢ TỪ TOOL cho: product_id, giá, tồn kho, mức giảm giá, tổng tiền, đường dẫn lưu.
- TUYỆT ĐỐI không tự bịa thông tin sản phẩm, giá, giảm giá, tổng tiền hay file path.

## Thông tin bắt buộc trước khi gọi BẤT KỲ tool nào
Một đơn hợp lệ cần đủ 5 thứ:
  1) tên khách hàng
  2) số điện thoại
  3) email
  4) địa chỉ giao hàng
  5) ít nhất 1 sản phẩm kèm số lượng
Nếu THIẾU bất kỳ mục nào: KHÔNG gọi tool nào cả. Trả lời ngắn bằng tiếng Việt,
bắt đầu bằng "Mình cần thêm" và liệt kê đúng (các) mục còn thiếu
(ví dụ: email, số điện thoại, địa chỉ giao hàng). Sau đó dừng lại.

LƯU Ý: nếu khách ĐÃ nêu tên sản phẩm (kể cả tên thương mại như "Sony WH-1000XM5")
kèm số lượng, thì coi như đã có sản phẩm — hãy dùng list_products để tra cứu,
TUYỆT ĐỐI không hỏi lại tên/mô tả sản phẩm. Chỉ hỏi lại khi thiếu thông tin
KHÁCH HÀNG (tên, số điện thoại, email, địa chỉ) hoặc khách chưa nêu sản phẩm nào.

## Từ chối (guardrail)
Nếu khách yêu cầu tạo hóa đơn giả, tự ép/đặt mức giảm giá, bỏ qua tồn kho,
hoặc bỏ qua catalog/policy → KHÔNG gọi tool nào. Trả lời lịch sự bằng tiếng Việt,
nói rõ "Mình không thể" làm việc đó và rằng giá cùng khuyến mãi chỉ áp dụng theo
hệ thống/policy thật. Sau đó dừng lại.

## Quy trình cho đơn hợp lệ (đúng thứ tự, không bỏ bước)
1. list_products: tìm từng sản phẩm khách nêu để lấy product_id chính xác.
2. get_product_details: gọi MỘT LẦN với TẤT CẢ product_id đã chọn để lấy
   giá/tồn kho và nhận detail_token.
3. get_discount: seed_hint = email của khách; customer_tier = "standard"
   (chỉ dùng "vip" nếu khách nói rõ). Ghi lại discount_rate và campaign_code.
4. calculate_order_totals: truyền items (product_id + quantity), detail_token,
   và discount_rate vừa nhận.
   - Nếu kết quả trả về status = "error" (ví dụ không đủ tồn kho): KHÔNG lưu đơn.
     Báo khách ngắn gọn lý do (vd hết hàng) rồi dừng.
5. save_order: chỉ gọi khi bước 4 thành công. Truyền đầy đủ tên, số điện thoại,
   email, địa chỉ giao hàng (đều là chuỗi), items, detail_token, discount_rate,
   campaign_code. Dùng lại CÙNG detail_token và CÙNG danh sách items.

## Câu trả lời cuối
- Sau khi lưu thành công: xác nhận ngắn gọn bằng tiếng Việt, nêu rõ mã đơn (order_id),
  mức giảm giá, tổng tiền cuối cùng, và nơi đã lưu (save_path).
- Mọi con số phải lấy nguyên từ kết quả tool, không tự làm tròn hay thay đổi.
""".strip()


def _coerce_items(items: Any) -> list[OrderLineInput]:
    """Normalize tool `items` arg into OrderLineInput objects (langchain may pass dicts)."""
    normalized: list[OrderLineInput] = []
    for item in items or []:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
        elif isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            if product_id:
                normalized.append(OrderLineInput(product_id=product_id, quantity=int(item.get("quantity", 1))))
    return normalized


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return compact matches.

        Use this FIRST to resolve each product the customer mentions into an exact
        product_id. Returns a JSON list of candidates (product_id, name, brand, category).
        """
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact price, stock, and warranty for the given product IDs.

        Call this ONCE with every chosen product_id. Returns a JSON object that
        includes a `detail_token`; later pricing and save steps require that token.
        """
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the deterministic campaign discount for this customer.

        Pass the customer email as `seed_hint`. Returns `discount_rate` (0.1 or 0.2)
        and a `campaign_code`.
        """
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock against the detail_token and compute the discounted total.

        Returns status "error" with messages when stock is insufficient or the token
        is invalid; in that case do not save the order.
        """
        payload = store.calculate_order_totals(
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final, validated order to a local JSON file.

        Only call after calculate_order_totals succeeds. Returns the saved order
        payload and its file path.
        """
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
