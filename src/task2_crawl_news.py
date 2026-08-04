"""Task 2 - Crawl các bài hướng dẫn từ Trung tâm trợ giúp Shopee Việt Nam.

Chạy từ thư mục gốc của dự án:
    python -m src.task2_crawl_news

Mỗi URL được lưu thành một file JSON riêng trong ``data/landing/news``.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Các trang công khai, bao phủ các chủ đề được gợi ý trong README.
ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/79180",
    "https://help.shopee.vn/portal/4/article/79128",
    "https://help.shopee.vn/portal/4/article/79473",
    "https://help.shopee.vn/portal/4/article/189473",
    "https://help.shopee.vn/portal/4/article/79377",
    "https://help.shopee.vn/portal/4/article/79522",
    "https://help.shopee.vn/portal/4/article/79711",
]


def setup_directory() -> None:
    """Tạo thư mục đầu ra nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_article(url: str) -> tuple[str, str]:
    """Tải và trích nội dung SSR của một trang trợ giúp Shopee."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Lab-Crawler/1.0)"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else "Không rõ tiêu đề"
    title = title.removesuffix(" | Shopee Trung tâm trợ giúp").strip()
    article = soup.select_one(".ssr-key-content")
    if article is None:
        raise ValueError("Không tìm thấy vùng nội dung .ssr-key-content")

    # Giữ cấu trúc đoạn/list cơ bản để Task 3 có dữ liệu Markdown dễ chunk.
    lines: list[str] = []
    for element in article.find_all(["h1", "h2", "h3", "p", "li", "tr"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        prefix = "- " if element.name == "li" else ""
        if element.name in {"h1", "h2", "h3"}:
            prefix = "#" * int(element.name[1]) + " "
        line = prefix + text
        if not lines or lines[-1] != line:
            lines.append(line)

    content = "\n\n".join(lines)
    return title, content


async def crawl_article(url: str, crawler: Any | None = None) -> dict[str, str]:
    """Crawl một bài và trả về nội dung cùng metadata theo yêu cầu README."""
    del crawler  # Giữ tham số để thuận tiện thay bằng browser crawler khi cần.
    title, content = await asyncio.to_thread(_fetch_article, url)
    if len(content) < 200:
        raise ValueError(f"Nội dung crawl quá ngắn ({len(content)} ký tự): {url}")
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


def _safe_filename(title: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    return f"article_{index:02d}_{slug or 'shopee-help'}.json"


async def crawl_all() -> list[Path]:
    """Crawl toàn bộ URL; chỉ lưu các bài có nội dung hợp lệ."""
    setup_directory()
    saved_files: list[Path] = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ! Bỏ qua do lỗi: {exc}")
            continue

        filepath = DATA_DIR / _safe_filename(article["title"], index)
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        saved_files.append(filepath)
        print(f"  ✓ Saved: {filepath}")

    if len(saved_files) < 5:
        raise RuntimeError(
            f"Chỉ crawl thành công {len(saved_files)}/5 bài tối thiểu. "
            "Hãy kiểm tra kết nối hoặc cập nhật ARTICLE_URLS."
        )
    print(f"\n✓ Hoàn tất: {len(saved_files)} bài trong {DATA_DIR}")
    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
