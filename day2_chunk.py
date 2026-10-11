"""第 2 天：把 extracted.txt 切成小块，存成 chunks.json。

用法：
    python day2_chunk.py              按默认参数切块
    python day2_chunk.py 800 150      指定块大小和重叠长度
"""

import json
import re
import sys
from pathlib import Path

SOURCE = Path("extracted.txt")
OUTPUT = Path("chunks.json")

CHUNK_SIZE = 600
OVERLAP = 100

PAGE_PATTERN = re.compile(r"^=====\s*第\s*(\d+)\s*页\s*=====\s*$")


def split_pages(text: str) -> list[tuple[int, str]]:
    """按页码标记切分，返回 [(页码, 该页文字), ...]。"""
    pages = []
    current_page = 0
    buffer: list[str] = []

    for line in text.splitlines():
        match = PAGE_PATTERN.match(line.strip())
        if match:
            if buffer:
                pages.append((current_page, "\n".join(buffer)))
                buffer = []
            current_page = int(match.group(1))
        else:
            buffer.append(line)

    if buffer:
        pages.append((current_page, "\n".join(buffer)))

    return pages


def clean_text(text: str) -> str:
    """去掉多余空行和首尾空白。"""
    lines = [line.strip() for line in text.splitlines()]
    kept = [line for line in lines if line]
    return "\n".join(kept)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """把一段文字切成若干块，相邻块之间有 overlap 个字符重叠。"""
    if size <= overlap:
        raise ValueError("块大小必须大于重叠长度")

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step

    return chunks


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else CHUNK_SIZE
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP

    if not SOURCE.exists():
        print(f"找不到 {SOURCE}，请先运行：python day1_read_pdf.py --extract")
        return

    raw = SOURCE.read_text(encoding="utf-8")
    pages = split_pages(raw)

    all_chunks = []
    for page_no, page_text in pages:
        cleaned = clean_text(page_text)
        if not cleaned:
            continue
        for piece in chunk_text(cleaned, size, overlap):
            all_chunks.append({
                "id": len(all_chunks),
                "page": page_no,
                "text": piece,
                "length": len(piece),
            })

    OUTPUT.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lengths = [c["length"] for c in all_chunks]
    print("=" * 50)
    print(f"块大小 {size}，重叠 {overlap}")
    print(f"共 {len(pages)} 页 → 切成 {len(all_chunks)} 块")
    print(f"块长度：最短 {min(lengths)}，最长 {max(lengths)}，平均 {sum(lengths) // len(lengths)}")
    print(f"已保存到：{OUTPUT.resolve()}")
    print("=" * 50)
    print("前 3 块预览：")
    for chunk in all_chunks[:3]:
        print(f"\n--- id={chunk['id']} 第{chunk['page']}页 长度{chunk['length']} ---")
        print(chunk["text"][:200])


if __name__ == "__main__":
    main()
