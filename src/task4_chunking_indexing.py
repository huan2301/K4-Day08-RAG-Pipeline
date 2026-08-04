"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # Vì sao chọn 800? ... 
CHUNK_OVERLAP = 100      # Vì sao chọn 100? ... (Khoảng overlap giữa các chunk để giữ tính liên tục)
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
# BGE-M3 hỗ trợ đa ngôn ngữ, phù hợp tài liệu và câu hỏi tiếng Việt.
# Mỗi văn bản được biểu diễn bằng vector 1024 chiều.
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
# ChromaDB chạy local, lưu persistent và hỗ trợ cosine distance.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def infer_customer_role(filename: str, content: str) -> str:
    """
    Suy luận nhóm khách hàng từ tên và nội dung tài liệu.

    Returns:
        buyer, seller hoặc both
    """
    text = f"{filename} {content[:2000]}".lower()

    seller_keywords = [
        "seller",
        "người bán",
        "đăng bán",
        "sản phẩm bị cấm",
        "hạn chế sản phẩm",
        "quy định bán hàng",
    ]

    buyer_keywords = [
        "buyer",
        "người mua",
        "phương thức thanh toán",
        "theo dõi đơn hàng",
        "đặt hàng",
    ]

    has_seller = any(keyword in text for keyword in seller_keywords)
    has_buyer = any(keyword in text for keyword in buyer_keywords)

    if has_seller and not has_buyer:
        return "seller"

    if has_buyer and not has_seller:
        return "buyer"

    return "both"


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chia documents thành các chunk có metadata.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

    chunks = []

    for doc in documents:
        content = doc.get("content", "").strip()

        if not content:
            continue

        splits = splitter.split_text(content)

        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": chunk_index,
                },
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Tạo embedding cho toàn bộ chunks.
    """
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        vector = embedding.tolist()

        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension không đúng: "
                f"expected={EMBEDDING_DIM}, actual={len(vector)}"
            )

        chunk["embedding"] = vector

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks, embeddings và metadata vào ChromaDB.
    """
    if not chunks:
        print("Không có chunk để index")
        return

    collection = get_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        metadata = chunk["metadata"]

        source_path = metadata["source_path"].replace("/", "_")
        chunk_index = metadata["chunk_index"]

        ids.append(f"{source_path}_chunk_{chunk_index}")
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])
        metadatas.append(metadata)

    # Chia batch để tránh truyền một payload quá lớn
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(
        f"Đã lưu {len(chunks)} chunks vào collection "
        f"'{COLLECTION_NAME}'. Tổng hiện tại: {collection.count()}"
    )

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ Markdown trong data/standardized/.

    Returns:
        List of {
            "content": str,
            "metadata": {
                "source": str,
                "type": str,
                "customer_role": str
            }
        }
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        print(f"Không tìm thấy thư mục: {STANDARDIZED_DIR}")
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()

        # Không index file rỗng
        if not content:
            print(f"Bỏ qua file rỗng: {md_file.name}")
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        path_parts = relative_path.parts

        doc_type = "legal" if "legal" in path_parts else "news"
        customer_role = infer_customer_role(md_file.name, content)

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": relative_path.as_posix(),
                "type": doc_type,
                "customer_role": customer_role,
            },
        })

    return documents

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")

def get_embedding_model():
    """Load embedding model một lần và tái sử dụng."""
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model

def get_collection():
    """Mở collection ChromaDB persistent."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

if __name__ == "__main__":
    run_pipeline()
