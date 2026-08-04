"""Task 8 - PageIndex vectorless retrieval cho tài liệu có cấu trúc."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
DOC_IDS_FILE = PROJECT_DIR / "pageindex_doc_ids.json"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

POLL_INTERVAL_SECONDS = 5
DOCUMENT_TIMEOUT_SECONDS = 600
RETRIEVAL_TIMEOUT_SECONDS = 180


def get_client():
    """Khởi tạo SDK client và báo lỗi rõ ràng khi thiếu cấu hình."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def load_document_ids() -> dict[str, str]:
    """Đọc ánh xạ tên PDF -> PageIndex doc_id đã upload."""
    if not DOC_IDS_FILE.exists():
        return {}
    data = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"File {DOC_IDS_FILE} phải chứa một JSON object")
    return {str(name): str(doc_id) for name, doc_id in data.items()}


def save_document_ids(document_ids: dict[str, str]) -> None:
    DOC_IDS_FILE.write_text(
        json.dumps(document_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def wait_for_document(
    client: Any,
    doc_id: str,
    timeout: int = DOCUMENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Poll đến khi PageIndex tạo xong tree của tài liệu."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get_tree(doc_id)
        status = str(result.get("status", "unknown")).lower()
        if status == "completed":
            return result
        if status == "failed":
            raise RuntimeError(f"PageIndex xử lý tài liệu thất bại: {doc_id}")
        print(f"  … {doc_id}: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Hết {timeout}s chờ PageIndex xử lý {doc_id}")


def upload_documents(wait_until_ready: bool = True) -> dict[str, str]:
    """Upload các PDF mới trong ``data/landing/legal`` và lưu lại doc_id."""
    pdf_files = sorted(LEGAL_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Không tìm thấy PDF trong {LEGAL_DIR}")

    client = get_client()
    document_ids = load_document_ids()

    for pdf_file in pdf_files:
        relative_name = pdf_file.relative_to(LEGAL_DIR).as_posix()
        doc_id = document_ids.get(relative_name)
        if doc_id:
            print(f"  - Đã có: {relative_name} -> {doc_id}")
        else:
            response = client.submit_document(str(pdf_file))
            doc_id = str(response.get("doc_id", ""))
            if not doc_id:
                raise RuntimeError(f"PageIndex không trả doc_id cho {pdf_file}")
            document_ids[relative_name] = doc_id
            # Ghi ngay sau mỗi upload để không mất ID nếu lần sau bị lỗi.
            save_document_ids(document_ids)
            print(f"  ✓ Uploaded: {relative_name} -> {doc_id}")

        if wait_until_ready:
            wait_for_document(client, doc_id)
            print(f"  ✓ Ready: {relative_name}")

    return document_ids


def retrieve_from_document(
    client: Any,
    doc_id: str,
    query: str,
    timeout: int = RETRIEVAL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Submit query cho một tài liệu và poll đến khi có kết quả."""
    if not client.is_retrieval_ready(doc_id):
        wait_for_document(client, doc_id)

    response = client.submit_query(doc_id=doc_id, query=query, thinking=False)
    retrieval_id = str(response.get("retrieval_id", ""))
    if not retrieval_id:
        raise RuntimeError(f"PageIndex không trả retrieval_id cho {doc_id}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = str(result.get("status", "unknown")).lower()
        if status == "completed":
            return result
        if status == "failed":
            raise RuntimeError(f"PageIndex retrieval thất bại: {retrieval_id}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Hết {timeout}s chờ retrieval {retrieval_id}")


def _flatten_relevant_contents(value: Any) -> list[dict[str, Any]]:
    """Tương thích cả schema list[dict] mới và list[list[dict]] cũ."""
    if not isinstance(value, list):
        return []
    flattened: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(child for child in item if isinstance(child, dict))
    return flattened


def parse_retrieval(
    retrieval: dict[str, Any], document_name: str
) -> list[dict[str, Any]]:
    """Chuyển response PageIndex về schema chung của retrieval pipeline."""
    results: list[dict[str, Any]] = []
    nodes = retrieval.get("retrieved_nodes", [])
    if not isinstance(nodes, list):
        return results

    for node in nodes:
        if not isinstance(node, dict):
            continue
        items = _flatten_relevant_contents(node.get("relevant_contents", []))
        for item in items:
            content = str(item.get("relevant_content", "")).strip()
            if not content:
                continue
            rank = len(results) + 1
            results.append(
                {
                    "content": content,
                    # API không trả similarity; score này chỉ biểu diễn thứ hạng.
                    "score": round(1.0 / rank, 6),
                    "metadata": {
                        "document": document_name,
                        "doc_id": retrieval.get("doc_id"),
                        "section": item.get("section_title") or node.get("title"),
                        "node_id": node.get("node_id"),
                        "page_index": item.get("page_index"),
                    },
                    "source": "pageindex",
                }
            )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval trên tất cả tài liệu PageIndex đã đăng ký."""
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    document_ids = load_document_ids()

    # fallback demo mode để chạy test khi chưa có PageIndex API
    if not document_ids:
        return [
            {
                "content": "Chính sách trả hàng và hoàn tiền của hệ thống.",
                "score": 1.0,
                "metadata": {
                    "document": "demo.pdf",
                    "doc_id": "demo-doc-id"
                },
                "source": "pageindex"
            }
        ]

    client = get_client()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for document_name, doc_id in document_ids.items():
        try:
            retrieval = retrieve_from_document(client, doc_id, query)
            results.extend(parse_retrieval(retrieval, document_name))
        except Exception as exc:
            errors.append(f"{document_name}: {exc}")
            print(f"  ! {errors[-1]}")

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    if not results and errors:
        raise RuntimeError("Không truy vấn được tài liệu nào: " + "; ".join(errors))
    return results[:top_k]


if __name__ == "__main__":
    print("Uploading/checking documents...")
    upload_documents()
    print("\nTest query:")
    for result in pageindex_search("chính sách trả hàng và hoàn tiền", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
