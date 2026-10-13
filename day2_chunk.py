"""把 extracted.txt 切成小块，存成 chunks.json。

用法：
    python day2_chunk.py              用默认参数
    python day2_chunk.py 800 150      指定块大小和重叠
"""

import json
import re
import sys
from pathlib import Path

SOURCE = Path("extracted.txt")
OUTPUT = Path("chunks.json")

CHUNK_SIZE = 600
OVERLAP = 100
MIN_LENGTH = 50

PAGE_PATTERN = re.compile(r"^=+\s*(.+?)\s*\|\s*第\s*(\d+)\s*页\s*=+\s*$")


def split_pages(text: str) -> list:
    """按页标记切分，返回 [(文件名, 页码, 该页文字), ...]。"""
    pages = []
    current = None
    buffer = []

    for line in text.splitlines():
        match = PAGE_PATTERN.match(line.strip())
        if match:
            if current is not None and buffer:
                pages.append((current[0], current[1], "\n".join(buffer)))
            current = (match.group(1), int(match.group(2)))
            buffer = []
        else:
            buffer.append(line)

    if current is not None and buffer:
        pages.append((current[0], current[1], "\n".join(buffer)))

    return pages


def clean_text(text: str) -> str:
    """去掉空行和多余空白。"""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chop(text: str, size: int, overlap: int) -> list:
    """把一段文字切成若干块，相邻块之间有 overlap 个字符重叠。"""
    if size <= overlap:
        raise ValueError("块大小必须大于重叠长度")

    pieces = []
    start = 0
    step = size - overlap

    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
        start += step

    return pieces


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else CHUNK_SIZE
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP

    if not SOURCE.exists():
        print(f"找不到 {SOURCE}，请先运行：python day1_read_pdf.py --extract")
        return

    raw = SOURCE.read_text(encoding="utf-8-sig")
    pages = split_pages(raw)

    chunks = []
    skipped = 0
    borrowed = 0

    for name, page_no, page_text in pages:
        cleaned = clean_text(page_text)
        if not cleaned:
            continue

        pieces = chop(cleaned, size, overlap)

        # 内容太少的页：借用上一块的内容补足上下文
        if pieces and len(pieces[0]) < MIN_LENGTH and chunks:
            borrowed += 1
            previous = chunks[-1]["text"]
            merged = (previous[-overlap:] + "\n" + cleaned).strip()
            pieces = chop(merged, size, overlap)

        for piece in pieces:
            if len(piece) < MIN_LENGTH:
                skipped += 1
                continue
            chunks.append({
                "id": len(chunks),
                "source": name,
                "page": page_no,
                "text": piece,
                "length": len(piece),
            })

    OUTPUT.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lengths = [c["length"] for c in chunks]
    print("=" * 50)
    print(f"块大小 {size}，重叠 {overlap}，最小长度 {MIN_LENGTH}")
    print(f"共 {len(pages)} 页 → 切成 {len(chunks)} 块")
    print(f"短页借用上文：{borrowed} 次，丢弃过短块：{skipped} 个")
    print(f"块长度：最短 {min(lengths)}，最长 {max(lengths)}，平均 {sum(lengths)//len(lengths)}")
    print(f"已保存到：{OUTPUT.resolve()}")
    print("=" * 50)

    from collections import Counter
    stat = Counter()
    for c in chunks:
        stat[c["source"]] += 1
    print("各文件块数：")
    for name, count in stat.items():
        print(f"  {name}: {count} 块")


if __name__ == "__main__":
    main()
