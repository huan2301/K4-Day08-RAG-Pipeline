"""Task 5 - Dense retrieval bằng cosine similarity và HyDE tùy chọn.

Module dùng đúng embedding model, đường dẫn ChromaDB và collection được cấu hình
ở Task 4. Các dependency nặng được import lazy để việc import module không tải model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import import_module
from typing import Mapping, Protocol, TypedDict, cast

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class EmbeddingVector(Protocol):
    """Giá trị vector do SentenceTransformer trả về."""

    def tolist(self) -> list[float]: ...


class EmbeddingModel(Protocol):
    def encode(self, text: str, **kwargs: object) -> EmbeddingVector: ...


class VectorCollection(Protocol):
    def count(self) -> int: ...

    def query(self, **kwargs: object) -> Mapping[str, object]: ...


class SearchResult(TypedDict):
    content: str
    score: float
    metadata: dict[str, object]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Tải một lần embedding model đã chọn ở Task 4."""
    module = import_module("sentence_transformers")
    model_class = getattr(module, "SentenceTransformer")
    return cast(EmbeddingModel, model_class(EMBEDDING_MODEL))


@lru_cache(maxsize=1)
def get_collection() -> VectorCollection:
    """Mở collection ChromaDB persistent do Task 4 tạo ra."""
    chromadb = import_module("chromadb")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # get_collection giúp phát hiện rõ trường hợp Task 4 chưa index, thay vì âm thầm
    # tạo một collection rỗng do viết sai tên.
    return cast(VectorCollection, client.get_collection(name=COLLECTION_NAME))


def generate_hypothetical_document(query: str) -> str:
    """Sinh tài liệu giả định (HyDE) bằng OpenAI/OpenRouter compatible API.

    Cấu hình bằng ``OPENAI_API_KEY`` hoặc ``OPENROUTER_API_KEY``. Khi dùng
    OpenRouter có thể đặt ``HYDE_MODEL`` và ``OPENROUTER_BASE_URL`` trong .env.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HyDE cần OPENAI_API_KEY hoặc OPENROUTER_API_KEY. "
            "Dùng use_hyde=False nếu chỉ cần semantic search thông thường."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL")
    openai = import_module("openai")
    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client_class = getattr(openai, "OpenAI")
    client = client_class(**client_options)
    model = os.getenv("HYDE_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=250,
        messages=[
            {
                "role": "system",
                "content": (
                    "Viết một đoạn tài liệu hỗ trợ khách hàng thương mại điện tử "
                    "ngắn, chính xác và có khả năng trả lời câu hỏi. Không mở đầu, "
                    "không trích dẫn nguồn và không nói rằng đây là câu trả lời giả định."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    document = (response.choices[0].message.content or "").strip()
    if not document:
        raise RuntimeError("LLM không sinh được tài liệu HyDE.")
    return document


def _first_list(result: Mapping[str, object], key: str) -> list[object]:
    """Đọc danh sách đầu tiên từ format batch response của ChromaDB."""
    value = result.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        return []
    first_batch = value[0]
    return list(first_batch) if isinstance(first_batch, (list, tuple)) else []


def semantic_search(
    query: str,
    top_k: int = 10,
    *,
    use_hyde: bool = False,
) -> list[SearchResult]:
    """Tìm các chunk gần query nhất trong không gian embedding.

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số kết quả tối đa; giá trị <= 0 trả về danh sách rỗng.
        use_hyde: Sinh một tài liệu giả định rồi embed tài liệu đó thay cho query.

    Returns:
        Danh sách ``content``, ``score`` và ``metadata``, giảm dần theo cosine
        similarity. Score được chuẩn hóa về [0, 1].
    """
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    retrieval_text = generate_hypothetical_document(query) if use_hyde else query
    model = get_embedding_model()
    query_vector = model.encode(
        retrieval_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(int(top_k), collection_size),
        include=["documents", "metadatas", "distances"],
    )
    documents = _first_list(result, "documents")
    metadatas = _first_list(result, "metadatas")
    distances = _first_list(result, "distances")

    output: list[SearchResult] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if not isinstance(distance, (int, float)):
            continue
        # Collection Task 4 dùng hnsw:space=cosine, do đó similarity = 1-distance.
        score = max(0.0, min(1.0, 1.0 - distance))
        output.append(
            {
                "content": str(document),
                "score": round(score, 6),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("quy định trả hàng hoàn tiền Shopee", top_k=5)
    for item in results:
        print(f"[{item['score']:.3f}] {item['content'][:100]}...")
