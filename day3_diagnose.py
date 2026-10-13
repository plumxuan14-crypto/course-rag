"""诊断向量检索效果：对比不同模型 + 分析分数分布。

用法：
    python day3_diagnose.py
"""

import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

VECTORS = Path("vectors.json")

# 测试用例：(问题, 期望能搜到的关键词)
CASES = [
    ("什么是顺序结构？", ["顺序"]),
    ("分支结构有哪几种？", ["分支", "if", "switch"]),
    ("switch 语句怎么用？", ["switch", "case"]),
    ("什么是位运算？", ["位", "移位", "取反"]),
    ("scanf 怎么用？", ["scanf"]),
    # 下面这些课件里应该没有
    ("今天中午吃什么？", []),
    ("量子力学的测不准原理是什么？", []),
    ("如何做红烧肉？", []),
    ("明朝为什么灭亡？", []),
]


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_client():
    load_dotenv()
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )


def evaluate(client, model, records):
    """用给定模型跑所有测试用例，返回每个用例的最高分。"""
    texts = [c["text"] for c in records]
    stored = []
    for start in range(0, len(texts), 10):
        response = client.embeddings.create(model=model, input=texts[start:start + 10])
        ordered = sorted(response.data, key=lambda item: item.index)
        stored.extend([item.embedding for item in ordered])

    print(f"\n{'=' * 62}")
    print(f"模型：{model}")
    print("=" * 62)

    relevant_scores = []
    irrelevant_scores = []

    for question, keywords in CASES:
        response = client.embeddings.create(model=model, input=[question])
        qvec = response.data[0].embedding

        scored = sorted(
            ((cosine(qvec, vec), records[i]) for i, vec in enumerate(stored)),
            key=lambda item: item[0],
            reverse=True,
        )

        top_score, top_record = scored[0]
        expect = "相关" if keywords else "无关"

        # 检查 top3 里有没有命中关键词
        hit = ""
        if keywords:
            for _, record in scored[:3]:
                if any(kw.lower() in record["text"].lower() for kw in keywords):
                    hit = "命中"
                    break
            if not hit:
                hit = "未命中"
        else:
            hit = ""

        print(f"  [{expect}] {question:<22} 最高分 {top_score:.3f}  {hit}")
        print(f"           → {top_record['text'][:52]}")

        if keywords:
            relevant_scores.append(top_score)
        else:
            irrelevant_scores.append(top_score)

    if relevant_scores and irrelevant_scores:
        r_avg = sum(relevant_scores) / len(relevant_scores)
        i_avg = sum(irrelevant_scores) / len(irrelevant_scores)
        gap = r_avg - i_avg
        print("-" * 62)
        print(f"相关问题的平均分：  {r_avg:.3f}")
        print(f"无关问题的平均分：  {i_avg:.3f}")
        print(f"差距：              {gap:+.3f}   {'✅ 区分度好' if gap > 0.06 else '⚠️ 区分度差'}")

    return relevant_scores, irrelevant_scores


def main() -> None:
    records = json.loads(VECTORS.read_text(encoding="utf-8"))
    print(f"载入 {len(records)} 条已存的向量")
    client = get_client()

    for model in ["text-embedding-v3", "text-embedding-v4"]:
        try:
            evaluate(client, model, records)
        except Exception as exc:
            print(f"\n模型 {model} 调用失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
