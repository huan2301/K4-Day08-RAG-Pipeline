"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import re


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> list[str]:
    """Normalize text so punctuation does not prevent keyword matches."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    """Load standardized Markdown files as the lexical-search corpus."""
    if not STANDARDIZED_DIR.exists():
        return []

    corpus = []
    for markdown_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = markdown_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = markdown_file.relative_to(STANDARDIZED_DIR)
        corpus.append(
            {
                "content": content,
                "metadata": {
                    "source": str(relative_path),
                    "type": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
                },
            }
        )
    return corpus


# List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = _load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        return None

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip() or not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    query_tokens = _tokenize(query)
    if not query_tokens or bm25 is None:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(CORPUS)), key=lambda index: (-scores[index], index))

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[index]["content"],
                "score": score,
                "metadata": CORPUS[index].get("metadata", {}),
            }
        )
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
