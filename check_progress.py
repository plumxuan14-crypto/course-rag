"""看看我的 RAG 项目走到哪一步了。

直接运行：python check_progress.py
"""

import json
from pathlib import Path

STEPS = [
    ("课件文件", ["*.pdf", "*.pptx"], "原始课件"),
    ("纯文本", ["extracted.txt"], "把课件变成文字"),
    ("文本切块", ["chunks.json"], "把文字切成小块"),
    ("向量数据", ["vectors.json", "vectors.npy"], "每块变成一串数字"),
    ("问答程序", ["ask.py", "day4_search.py"], "能提问、能检索"),
]

print()
print("=" * 56)
print("  我的 RAG 项目 · 进度检查")
print("=" * 56)

done = 0

for name, patterns, note in STEPS:
    found = []
    for pattern in patterns:
        found.extend(Path(".").glob(pattern))

    if found:
        done += 1
        mark = "[完成]"
        detail = f"{len(found)} 个文件"
    else:
        mark = "[待办]"
        detail = "还没做"

    print(f"  {mark} {name:<8} {note:<18} {detail}")

print("-" * 56)
print(f"  进度：{done} / {len(STEPS)} 步")
print("=" * 56)

# 看看切块结果
chunks_file = Path("chunks.json")
if chunks_file.exists():
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    print()
    print(f"  你的 chunks.json 里有 {len(chunks)} 块文字")
    print()
    print("  下面是其中一块的真实样子（将来 AI 就靠它回答问题）：")
    print("  " + "-" * 52)

    sample = chunks[len(chunks) // 2]
    print(f"  这块的编号：{sample['id']}")
    print(f"  来自第     ：{sample['page']} 页")
    print(f"  文字长度   ：{sample['length']} 个字")
    print()
    for line in sample["text"][:300].splitlines():
        print(f"    {line}")

    print("  " + "-" * 52)
    print()
    print("  想自己看更多块，可以打开 chunks.json（记事本就能开）")
else:
    print()
    print("  还没有 chunks.json，先完成第 3 步（切块）")

print()
print("  下一步：把这些小块变成【一串数字】，这样就能按意思搜索了。")
print()
