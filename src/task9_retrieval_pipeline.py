"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
# Ngưỡng được so với cosine similarity gốc từ semantic_search.
# Không so với RRF score.
SCORE_THRESHOLD = 0.48

# Số kết quả mặc định trả về cho generation.
DEFAULT_TOP_K = 5

# Semantic và BM25 được hợp nhất bằng RRF.
RERANK_METHOD = "rrf"

def make_result_key(item: dict) -> str:
    """
    Nhận diện cùng một chunk giữa hai retriever.
    """
    metadata = item.get("metadata") or {}

    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")

    if source is not None and chunk_index is not None:
        return f"{source}::chunk::{chunk_index}"

    content = str(item.get("content", ""))
    return " ".join(content.lower().split())


def merge_without_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge đơn giản để phục vụ A/B test RRF vs không RRF.

    Dense được ưu tiên trước, sau đó bổ sung các sparse result
    chưa xuất hiện.
    """
    merged = []
    seen = set()

    for item in dense_results + sparse_results:
        key = make_result_key(item)

        if not key or key in seen:
            continue

        seen.add(key)
        merged.append(item.copy())

        if len(merged) >= top_k:
            break

    return merged

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh.

    Quy trình:
        1. Chạy Semantic Search.
        2. Chạy BM25 Search.
        3. Giữ lại cosine score gốc.
        4. Gộp hai danh sách bằng RRF.
        5. Nếu cosine tốt nhất dưới threshold, gọi PageIndex.
        6. Nếu PageIndex không khả dụng, trả kết quả hybrid.

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict,
            "source": "hybrid" hoặc "pageindex"
        }
    """
    query = query.strip()

    if not query:
        return []

    if top_k <= 0:
        return []

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(
            "score_threshold phải nằm trong khoảng [0, 1]"
        )

    # Lấy nhiều candidates hơn kết quả cuối để RRF có đủ lựa chọn.
    candidate_top_k = max(top_k * 2, top_k)

    # ---------------------------------------------------------
    # Step 1: Dense retrieval bằng semantic search
    # ---------------------------------------------------------
    dense_results = semantic_search(
        query=query,
        top_k=candidate_top_k,
    )
    dense_results = clean_results(dense_results)

    # Semantic Search phải trả score giảm dần.
    # Sort lại để bảo đảm phần tử đầu có cosine tốt nhất.
    dense_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Giữ riêng cosine score gốc để quyết định fallback.
    best_dense_score = (
        dense_results[0]["score"]
        if dense_results
        else 0.0
    )

    # ---------------------------------------------------------
    # Step 2: Sparse retrieval bằng BM25
    # ---------------------------------------------------------
    sparse_results = lexical_search(
        query=query,
        top_k=candidate_top_k,
    )
    sparse_results = clean_results(sparse_results)

    # RRF sử dụng thứ hạng, nên cần bảo đảm BM25 đã sort giảm dần.
    sparse_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Step 3: Gộp Semantic + BM25 bằng RRF
    # ---------------------------------------------------------
    if use_reranking:
        merged_results = rerank_rrf(
            ranked_lists=[
                dense_results,
                sparse_results,
            ],
            top_k=candidate_top_k,
            k=60,
        )
    else:
        # Khi tắt RRF để A/B testing, ưu tiên dense trước,
        # sau đó thêm sparse và loại chunk trùng.
        merged_results = merge_without_rrf(
            dense_results,
            sparse_results,
            top_k=candidate_top_k,
        )

    hybrid_results = [
        normalize_result(item, source="hybrid")
        for item in merged_results
        if item.get("content")
    ]

    # ---------------------------------------------------------
    # Step 4: Kiểm tra fallback bằng COSINE GỐC
    # ---------------------------------------------------------
    should_fallback = best_dense_score < score_threshold

    if should_fallback:
        print(
            "  ⚠ Semantic best score "
            f"({best_dense_score:.3f}) "
            f"< threshold ({score_threshold:.3f})"
        )
        print("  → Thử PageIndex fallback")

        try:
            fallback_results = pageindex_search(
                query=query,
                top_k=top_k,
            )

            fallback_results = [
                normalize_result(item, source="pageindex")
                for item in clean_results(fallback_results)
            ]

            if fallback_results:
                return fallback_results[:top_k]

            print(
                "  ⚠ PageIndex không trả về kết quả, "
                "sử dụng hybrid results"
            )

        except Exception as exc:
            # PageIndex là dịch vụ ngoài; thiếu API key, hết quota
            # hoặc lỗi mạng không được làm hỏng toàn bộ pipeline.
            print(
                "  ⚠ PageIndex fallback không khả dụng: "
                f"{exc}"
            )
            print("  → Sử dụng hybrid results")

    return hybrid_results[:top_k]

def normalize_result(
    item: dict,
    source: str,
) -> dict:
    """
    Chuẩn hóa kết quả từ các retriever về cùng schema.

    Args:
        item: Kết quả retrieval gốc.
        source: "hybrid" hoặc "pageindex".
    """
    metadata = item.get("metadata") or {}

    try:
        score = float(item.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    return {
        **item,
        "content": str(item.get("content", "")).strip(),
        "score": score,
        "metadata": metadata,
        "source": source,
    }
def clean_results(results: list[dict]) -> list[dict]:
    """
    Loại kết quả không hợp lệ và chuẩn hóa score.
    """
    cleaned = []

    for item in results or []:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()

        if not content:
            continue

        normalized = item.copy()
        normalized["content"] = content
        normalized["metadata"] = (
            item.get("metadata") or {}
        ).copy()

        try:
            normalized["score"] = float(
                item.get("score", 0.0)
            )
        except (TypeError, ValueError):
            normalized["score"] = 0.0

        cleaned.append(normalized)

    return cleaned

if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
