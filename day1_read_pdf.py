"""第 1 天：读取课件（PDF / PPTX），提取文字。

用法：
    python day1_read_pdf.py --extract   解析 PDF 并缓存为 extracted.txt
    python day1_read_pdf.py             查看缓存内容
"""

import sys
from pathlib import Path

SUPPORTED = (".pdf", ".pptx")
CACHE = Path("extracted.txt")


def extract_pdf(path: str) -> list[str]:
    """用 pdfplumber 解析 PDF，每页返回一段文字。"""
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for index, page in enumerate(pdf.pages, start=1):
            if index % 20 == 0 or index == total:
                print(f"  已解析 {index}/{total} 页...")
            pages.append(page.extract_text() or "")
    return pages


def extract_pptx(path: str) -> list[str]:
    """读取 PPTX，每页幻灯片返回一段文字。"""
    from pptx import Presentation

    prs = Presentation(path)
    slides = []
    for index, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs)
                    if text.strip():
                        lines.append(text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
        slides.append("\n".join(lines))
        if index % 20 == 0:
            print(f"  已解析 {index} 页...")
    return slides


def find_courseware() -> Path | None:
    """在当前文件夹里找第一个课件文件。"""
    for suffix in SUPPORTED:
        found = sorted(Path(".").glob(f"*{suffix}"))
        if found:
            return found[0]
    return None


def do_extract() -> None:
    target = find_courseware()
    if target is None:
        print("当前文件夹里没找到 PDF 或 PPTX 文件。")
        print("请把课件复制到：", Path(".").resolve())
        return

    suffix = target.suffix.lower()
    print(f"找到文件：{target.name}")
    print("开始解析，请稍等...")

    if suffix == ".pdf":
        pages = extract_pdf(str(target))
    elif suffix in (".pptx", ".ppt"):
        pages = extract_pptx(str(target))
    else:
        print(f"暂不支持的文件类型：{suffix}")
        return

    # 存成缓存文件，方便下次直接看
    with CACHE.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(pages, start=1):
            handle.write(f"\n===== 第 {index} 页 =====\n")
            handle.write(text)
            handle.write("\n")

    total = sum(len(p) for p in pages)
    print("=" * 50)
    print(f"文件：{target.name}")
    print(f"共 {len(pages)} 页，总字符数：{total}")
    print(f"已保存到：{CACHE.resolve()}")
    print("=" * 50)


def do_view() -> None:
    if not CACHE.exists():
        print("还没有缓存文件。先运行：python day1_read_pdf.py --extract")
        return

    text = CACHE.read_text(encoding="utf-8")
    print("=" * 50)
    print(f"缓存文件：{CACHE}")
    print(f"总字符数：{len(text)}")
    print("=" * 50)
    print("前 1500 个字符：")
    print("-" * 50)
    print(text[:1500])


def main() -> None:
    if "--extract" in sys.argv:
        do_extract()
    else:
        do_view()


if __name__ == "__main__":
    main()
