"""检索质量评估：测多个问题，看能不能找到正确的章节。

用法：
    python day4_eval.py
"""

import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

VECTORS = Path("vectors.json")
EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")

# (问题, 期望命中的章节关键词, 期望内容里应该出现的词)
CASES = [
    ("什么是顺序结构", "第2章", ["顺序"]),
    ("顺序结构和分支结构有什么区别", "第2章", ["顺序"]),
    ("分支结构有哪几种", "第3章", ["分支", "if", "switch"]),
    ("if 语句的格式是什么", "第3章", ["if"]),
    ("switch 语句怎么用", "第3章", ["switch", "case"]),
    ("scanf 怎么用", "第2章", ["scanf"]),
    ("什么是变量", "第2章", ["变量"]),
    ("printf 怎么输出", "第2章", ["printf", "输出"]),
    ("计算机技术是什么", "第1章", ["计算机"]),
    ("为什么要学习编程", "第1章", ["编程"]),
]


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def keyword_bonus(question: str, text: str) -> float:
    tokens = [question[i:i + 2] for i in range(len(question) - 1)]
    if not tokens:
        return 0.0
    unique = set(tokens)
    hits = sum(1 for t in unique if t in text)
    return (hits / len(unique)) * 0.15


def main() -> None:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    records = json.loads(VECTORS.read_text(encoding="utf-8"))
    print(f"载入 {len(records)} 块，模型 {records[0].get('model')}")
    print()

    hit_chapter = 0
    hit_keyword = 0

    for question, want_chapter, want_words in CASES:
        qvec = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding

        scored = []
        for record in records:
            score = cosine(qvec, record["vector"]) + keyword_bonus(question, record["text"])
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]

        # 看 top3 里有没有正确的章节
        ok_chapter = any(want_chapter in r["source"] for _, r in top)
        # 看 top3 里有没有关键词
        ok_word = any(
            any(w in r["text"] for w in want_words) for _, r in top
        )

        if ok_chapter:
            hit_chapter += 1
        if ok_word:
            hit_keyword += 1

        mark_c = "✅" if ok_chapter else "❌"
        mark_w = "✅" if ok_word else "❌"

        print(f"Q: {question}")
        print(f"   期望章节 {want_chapter} {mark_c}   期望关键词 {want_words} {mark_w}")
        for rank, (score, record) in enumerate(top, 1):
            chapter = record["source"].split()[0] if record["source"] else "?"
            snippet = record["text"][:40].replace("\n", " ")
            print(f"     [{rank}] {score:.3f}  {chapter}  第{record['page']}页  {snippet}")
        print()

    total = len(CASES)
    print("=" * 62)
    print(f"章节命中率：{hit_chapter}/{total} = {hit_chapter / total * 100:.0f}%")
    print(f"关键词命中率：{hit_keyword}/{total} = {hit_keyword / total * 100:.0f}%")
    print("=" * 62)
    print()
    if hit_keyword / total >= 0.8:
        print("  ✅ 检索质量不错，可以继续下一步")
    elif hit_keyword / total >= 0.6:
        print("  ⚠️ 检索质量一般，可以调优，但也能用")
        print("     调优方向：调阈值、改切块大小、加关键词权重")
    else:
        print("  ❌ 检索质量较差，建议先调优")
        print("     可能原因：切块太碎、文档内容太少、需要重排模型")


if __name__ == "__main__":
    main()
