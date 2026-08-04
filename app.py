"""Streamlit UI cho E-commerce Support RAG Chatbot.

Chạy: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Shopee Policy Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --brand: #ee4d2d;
        --brand-dark: #b22204;
        --slate: #263238;
        --teal: #008471;
        --canvas: #f9f9f9;
        --surface: #ffffff;
        --line: #e3beb6;
    }
    .stApp {
        background: var(--canvas);
        color: #1a1c1c;
        font-family: "Be Vietnam Pro", Inter, "Segoe UI", sans-serif;
    }
    .block-container {max-width: 1080px; padding-top: 1.35rem; padding-bottom: 6rem;}
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid #eeeeee;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 12px; border: 1px solid #e8e8e8; text-align: left;
        background: #ffffff; color: var(--slate); min-height: 42px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--brand); color: var(--brand-dark); background: #fff7f4;
    }
    .hero {
        padding: 1.15rem 1.35rem; border-radius: 16px;
        background: #ffffff; border: 1px solid #eeeeee;
        box-shadow: 0 4px 20px rgba(38, 50, 56, 0.045);
        margin-bottom: 1.25rem; position: relative; overflow: hidden;
    }
    .hero:before {content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; background: var(--brand);}
    .hero h1 {margin: 0 0 .25rem; font-size: 1.65rem; color: var(--slate); letter-spacing: -.02em;}
    .hero p {margin: 0; color: #667279; line-height: 1.5; font-size: .92rem;}
    .status-ok {color: var(--teal); font-weight: 600;}
    .status-warn {color: #a15c00; font-weight: 600;}
    [data-testid="stChatMessage"] {
        background: #ffffff; border: 1px solid #eceff1; border-radius: 16px;
        padding: .55rem .8rem; margin: 0 0 .85rem;
        box-shadow: 0 3px 14px rgba(38, 50, 56, 0.035);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #fff3ef; border-color: #ffd6cc;
        max-width: 78%; margin-left: auto;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid var(--teal);
    }
    [data-testid="stExpander"] {
        border-radius: 14px; border-color: #e8e8e8; background: #ffffff;
    }
    [data-testid="stChatInput"] {border-color: #d8d8d8; border-radius: 16px;}
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--brand); box-shadow: 0 0 0 2px rgba(238,77,45,.12);
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {background: var(--brand);}
    .welcome {
        padding: 2.4rem 1rem 1.4rem; text-align: center; color: #546067;
    }
    .welcome .mark {
        width: 54px; height: 54px; line-height: 54px; margin: 0 auto 14px;
        border-radius: 16px; background: #fff0ec; color: var(--brand);
        font-size: 26px;
    }
    .welcome h2 {font-size: 1.35rem; color: var(--slate); margin: 0 0 .4rem;}
    .welcome p {margin: 0 auto; max-width: 560px; font-size: .92rem; line-height: 1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)


def source_name(chunk: dict[str, Any], index: int) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        metadata.get("source")
        or metadata.get("document")
        or metadata.get("title")
        or metadata.get("url")
        or f"Nguồn {index}"
    )


def render_sources(sources: list[dict[str, Any]], retrieval_source: str = "unknown") -> None:
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo · {len(sources)} đoạn", expanded=False):
        st.caption(f"Phương thức truy xuất: `{retrieval_source}`")
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            name = source_name(source, index)
            doc_type = metadata.get("type") or metadata.get("doc_type") or "document"
            score = source.get("score")
            score_text = f" · score `{float(score):.4f}`" if isinstance(score, (int, float)) else ""
            st.markdown(f"**{index}. {name}** · `{doc_type}`{score_text}")
            section = metadata.get("section")
            page = metadata.get("page_index")
            if section or page:
                details = []
                if section:
                    details.append(f"Mục: {section}")
                if page:
                    details.append(f"Trang: {page}")
                st.caption(" · ".join(details))
            preview = str(source.get("content", "")).strip()
            if preview:
                st.text(preview[:420] + ("…" if len(preview) > 420 else ""))
            if index < len(sources):
                st.divider()


def friendly_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if isinstance(error, NotImplementedError):
        return "Pipeline retrieval chưa hoàn thiện. Hãy merge Task 9 rồi thử lại."
    if "api key" in lowered or "401" in lowered or "authentication" in lowered:
        return "API key LLM chưa hợp lệ. Hãy cập nhật OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong `.env`."
    if "collection" in lowered or "chroma" in lowered:
        return "Chưa tìm thấy ChromaDB. Hãy chạy `python -m src.task4_chunking_indexing`."
    if "pageindex" in lowered or "insufficientcredits" in lowered:
        return "PageIndex hiện không khả dụng; hãy kiểm tra API key/credit hoặc giảm ngưỡng fallback."
    return f"Pipeline chưa thể xử lý câu hỏi: {text}"


@st.cache_resource
def chroma_status() -> tuple[bool, int]:
    """Kiểm tra collection thật thay vì chỉ kiểm tra sự tồn tại của thư mục."""
    try:
        import chromadb
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            COLLECTION_NAME
        )
        count = collection.count()
        return count > 0, count
    except Exception:
        return False, 0


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

suggestions = [
    "Shopee hỗ trợ những phương thức thanh toán nào?",
    "Thời gian nhận tiền hoàn là bao lâu?",
    "Làm sao đổi phương thức thanh toán cho đơn hàng?",
    "Tôi có thể mua hàng trên Shopee quốc gia khác không?",
]

with st.sidebar:
    st.title("🛍️ Policy Assistant")
    st.caption("RAG chatbot cho chính sách và hỗ trợ khách hàng thương mại điện tử")
    st.divider()

    st.subheader("Trạng thái")
    chroma_ready, chroma_count = chroma_status()
    llm_ready = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.markdown(
        f'<div class="{"status-ok" if chroma_ready else "status-warn"}">'
        f'{"●" if chroma_ready else "○"} ChromaDB '
        f'{f"{chroma_count} chunks" if chroma_ready else "chưa index"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="{"status-ok" if llm_ready else "status-warn"}">'
        f'{"●" if llm_ready else "○"} LLM key {"đã cấu hình" if llm_ready else "chưa cấu hình"}</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    top_k = st.slider("Số đoạn tài liệu", min_value=3, max_value=10, value=5)

    st.subheader("Câu hỏi gợi ý")
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion

    st.divider()
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()
    st.caption("Semantic + BM25 → RRF → PageIndex fallback → Citation generation")

st.markdown(
    """
    <div class="hero">
      <h1>Hỏi đáp chính sách Shopee</h1>
      <p>Câu trả lời được tổng hợp từ kho tài liệu của nhóm và kèm nguồn kiểm chứng.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages and not st.session_state.pending_query:
    st.markdown(
        """
        <div class="welcome">
          <div class="mark">✦</div>
          <h2>Bạn cần tra cứu chính sách nào?</h2>
          <p>Hỏi về thanh toán, trả hàng, hoàn tiền hoặc quy định người bán. Hệ thống sẽ tìm bằng chứng và ghi rõ nguồn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2, gap="small")
    for index, suggestion in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(
                suggestion,
                key=f"main_suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_query = suggestion
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "unknown"),
            )

typed_query = st.chat_input("Nhập câu hỏi về thanh toán, hoàn tiền, mua hàng…")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tổng hợp câu trả lời…"):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = str(response.get("answer") or "Không có câu trả lời.")
                sources = response.get("sources") or []
                retrieval_source = str(response.get("retrieval_source") or "unknown")
            except Exception as error:
                # UI vẫn hữu ích khi LLM key lỗi: thử hiển thị bằng chứng retrieval
                # thay vì biến toàn bộ trải nghiệm thành một error box.
                try:
                    from src.task9_retrieval_pipeline import retrieve

                    sources = retrieve(query, top_k=top_k)
                except Exception:
                    try:
                        from src.task5_semantic_search import semantic_search

                        sources = semantic_search(query, top_k=top_k)
                        for source in sources:
                            source["source"] = "semantic"
                    except Exception:
                        sources = []
                if sources:
                    answer = (
                        f"**Chế độ chỉ tra cứu:** {friendly_error(error)}\n\n"
                        "Mình đã tìm được các đoạn tài liệu liên quan bên dưới; "
                        "hãy mở **Nguồn tham khảo** để kiểm tra bằng chứng."
                    )
                    retrieval_source = str(sources[0].get("source") or "retrieval-only")
                else:
                    answer = f"⚠️ {friendly_error(error)}"
                    retrieval_source = "error"

        st.markdown(answer)
        render_sources(sources, retrieval_source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
