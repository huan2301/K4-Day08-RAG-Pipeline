"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25.
"""

from pathlib import Path
import re


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> list[str]:
    """
    Normalize text for BM25 matching.

    - lowercase
    - convert common English keywords to Vietnamese equivalents
    - remove punctuation
    """
    text = text.casefold()

    # Mapping để query tiếng Anh match dữ liệu tiếng Việt
    synonyms = {
        "refund": "hoan tien",
        "return": "tra hang",
        "payment": "thanh toan",
        "shipping": "van chuyen",
        "delivery": "giao hang",
        "order": "don hang",
        "purchase": "mua hang",
    }

    for eng, vie in synonyms.items():
        text = text.replace(eng, vie)

    return re.findall(r"\w+", text, flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    """
    Load standardized Markdown files.
    """
    if not STANDARDIZED_DIR.exists():
        return []

    corpus = []

    for markdown_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = markdown_file.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            continue

        relative_path = markdown_file.relative_to(STANDARDIZED_DIR)

        corpus.append(
            {
                "content": content,
                "metadata": {
                    "source": str(relative_path),
                    "type": (
                        relative_path.parts[0]
                        if len(relative_path.parts) > 1
                        else "unknown"
                    ),
                },
            }
        )

    return corpus


# Corpus dùng cho lexical search
CORPUS: list[dict] = _load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Build BM25 index.
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        return None

    tokenized_corpus = [
        _tokenize(doc["content"])
        for doc in corpus
    ]

    return BM25Okapi(tokenized_corpus)


def lexical_search(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    BM25 keyword search.

    Returns:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict
            }
        ]
    """

    if (
        top_k <= 0
        or not query.strip()
        or not CORPUS
    ):
        return []

    bm25 = build_bm25_index(CORPUS)

    if bm25 is None:
        return []

    query_tokens = _tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(CORPUS)),
        key=lambda i: (-scores[i], i)
    )

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Không filter score <= 0
        # Vì BM25 corpus nhỏ có thể trả score âm
        results.append(
            {
                "content": CORPUS[index]["content"],
                "score": score,
                "metadata": CORPUS[index].get(
                    "metadata",
                    {}
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    results = lexical_search(
        "phương thức thanh toán shopee",
        top_k=5
    )

    for result in results:
        print(
            f"[{result['score']:.3f}] "
            f"{result['content'][:100]}..."
        )