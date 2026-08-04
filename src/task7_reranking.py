"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")

def make_document_key(item: dict) -> str:
    """
    Tạo khóa ổn định để nhận biết cùng một chunk xuất hiện
    trong Semantic Search và BM25.

    Ưu tiên source + chunk_index vì hai chunk khác nhau trong cùng
    một tài liệu có thể có nội dung gần giống nhau.

    Nếu metadata không đầy đủ, dùng content làm fallback.
    """
    metadata = item.get("metadata") or {}

    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")

    if source is not None and chunk_index is not None:
        return f"{source}::chunk::{chunk_index}"

    content = str(item.get("content", "")).strip()

    # Chuẩn hóa khoảng trắng để hai nội dung giống nhau nhưng khác xuống dòng
    # vẫn được nhận diện là cùng một chunk.
    normalized_content = " ".join(content.split())

    return normalized_content

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Gộp nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion.

    Công thức:
        RRF(d) = sum(1 / (k + rank_r(d)))

    Trong đó:
        - d là một document/chunk.
        - r là từng ranker, ví dụ Semantic hoặc BM25.
        - rank bắt đầu từ 1.
        - k=60 giúp giảm chênh lệch quá lớn giữa các vị trí đầu.

    Args:
        ranked_lists:
            Danh sách các kết quả đã được xếp hạng.
            Ví dụ: [semantic_results, lexical_results].
        top_k:
            Số kết quả tối đa cần trả về.
        k:
            Hằng số làm mượt của RRF.

    Returns:
        Danh sách kết quả được sắp xếp giảm dần theo RRF score.
    """
    if top_k <= 0:
        return []

    if k < 0:
        raise ValueError("k phải >= 0")

    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}
    first_seen_order: dict[str, int] = {}

    order_counter = 0

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue

        # Tránh cùng một document bị tính hai lần trong cùng một ranker.
        seen_in_current_list = set()

        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict):
                continue

            content = str(item.get("content", "")).strip()

            if not content:
                continue

            key = make_document_key(item)

            if key in seen_in_current_list:
                continue

            seen_in_current_list.add(key)

            if key not in first_seen_order:
                first_seen_order[key] = order_counter
                order_counter += 1

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (k + rank)
            )

            # Giữ một bản sao, tránh thay đổi object đầu vào.
            if key not in item_map:
                item_map[key] = item.copy()
                item_map[key]["metadata"] = item.get(
                    "metadata", {}
                ).copy()

    sorted_keys = sorted(
        rrf_scores,
        key=lambda key: (
            -rrf_scores[key],
            first_seen_order[key],
        ),
    )

    results = []

    for key in sorted_keys[:top_k]:
        result = item_map[key].copy()
        result["score"] = round(rrf_scores[key], 6)
        result["rrf_score"] = round(rrf_scores[key], 6)
        results.append(result)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates,
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    Với RRF:
        - Nếu candidates là list[dict], coi đó là một ranked list.
        - Nếu candidates là list[list[dict]], gộp nhiều ranked lists.

    Query không được RRF sử dụng vì RRF chỉ dựa trên thứ hạng.
    Query vẫn được giữ trong interface để đồng nhất với cross-encoder.
    """
    if top_k <= 0:
        return []

    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(
            query,
            candidates,
            top_k,
        )

    if method == "mmr":
        raise NotImplementedError(
            "MMR cần query_embedding và embedding của candidates"
        )

    if method == "rrf":
        # Trường hợp Task 9 truyền:
        # [dense_results, sparse_results]
        if isinstance(candidates[0], list):
            ranked_lists = candidates

        # Trường hợp test hoặc code khác truyền:
        # candidates = [{...}, {...}]
        else:
            ranked_lists = [candidates]

        return rerank_rrf(
            ranked_lists=ranked_lists,
            top_k=top_k,
        )

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    semantic_results = [
        {
            "content": "Chính sách trả hàng và hoàn tiền",
            "score": 0.85,
            "metadata": {
                "source": "returns-refund.md",
                "chunk_index": 0,
            },
        },
        {
            "content": "Phương thức thanh toán",
            "score": 0.70,
            "metadata": {
                "source": "payment-methods.md",
                "chunk_index": 0,
            },
        },
        {
            "content": "Quy định đăng bán",
            "score": 0.60,
            "metadata": {
                "source": "seller-listing.md",
                "chunk_index": 0,
            },
        },
    ]

    lexical_results = [
        {
            "content": "Chính sách trả hàng và hoàn tiền",
            "score": 12.4,
            "metadata": {
                "source": "returns-refund.md",
                "chunk_index": 0,
            },
        },
        {
            "content": "Quy định đăng bán",
            "score": 8.2,
            "metadata": {
                "source": "seller-listing.md",
                "chunk_index": 0,
            },
        },
        {
            "content": "Phương thức thanh toán",
            "score": 5.1,
            "metadata": {
                "source": "payment-methods.md",
                "chunk_index": 0,
            },
        },
    ]

    results = rerank_rrf(
        [semantic_results, lexical_results],
        top_k=3,
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. RRF={result['score']:.6f} "
            f"{result['content']}"
        )
