"""第 4 天自检：找出为什么检索分数异常低。

用法：
    python day4_check.py
"""

import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ============ 先加载 .env，再读任何环境变量 ============
load_dotenv(override=True)

VECTORS = Path("vectors.json")
CHUNKS = Path("chunks.json")

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("=" * 62)
    print("自检报告")
    print("=" * 62)
    print(f"当前 .env 里的模型：{EMBED_MODEL}")

    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))

    stored_model = vectors[0].get("model", "未记录")
    print(f"vectors.json 生成时用的模型：{stored_model}")
    print(f"vectors.json 条数：{len(vectors)}")
    print(f"chunks.json  条数：{len(chunks)}")
    print(f"向量维度         ：{len(vectors[0]['vector'])}")

    if stored_model != "未记录" and stored_model != EMBED_MODEL:
        print()
        print("  ❌ 生成向量的模型和当前配置不一致！")
        print("     → 请重新运行：python day3_embed.py")
        return

    # 检查 1：vectors 和 chunks 是否一一对应
    print()
    print("-" * 62)
    print("检查 1：vectors 和 chunks 的 id / 文字 是否一致")
    print("-" * 62)

    mismatch = 0
    for vec, chunk in zip(vectors, chunks):
        if vec["id"] != chunk["id"] or vec["text"][:30] != chunk["text"][:30]:
            mismatch += 1
    if mismatch == 0:
        print("  ✅ 完全一致")
    else:
        print(f"  ❌ 有 {mismatch} 条对不上！")
        for vec, chunk in list(zip(vectors, chunks))[:3]:
            print(f"     vectors id={vec['id']}  chunks id={chunk['id']}")
            print(f"       vectors: {vec['text'][:40]}")
            print(f"       chunks : {chunk['text'][:40]}")

    # 检查 2：重新算同一段文字的向量，看和存的一致不一致
    print()
    print("-" * 62)
    print("检查 2：重新算一遍第 0 块的向量，和存的比一比")
    print("-" * 62)

    sample_text = vectors[0]["text"]
    print(f"  测试文字：{sample_text[:50]}")

    response = client.embeddings.create(model=EMBED_MODEL, input=[sample_text])
    fresh = response.data[0].embedding
    stored = vectors[0]["vector"]

    print(f"  新算的维度：{len(fresh)}")
    print(f"  存的维度  ：{len(stored)}")

    if len(fresh) != len(stored):
        print("  ❌ 维度不一样！说明向量是【另一个模型】生成的")
        print("     → 请重新运行：python day3_embed.py")
        return

    same = cosine(fresh, stored)
    print(f"  新算 vs 已存 的相似度：{same:.4f}")
    if same > 0.99:
        print("  ✅ 一致，向量没问题")
    else:
        print("  ❌ 不一致！说明存下来的向量不是当前模型算的")
        print("     → 请重新运行：python day3_embed.py")

    # 检查 3：用问题去搜，看最高分
    print()
    print("-" * 62)
    print("检查 3：4 个问题的检索最高分")
    print("-" * 62)

    for question in ["什么是顺序结构", "分支结构", "scanf 怎么用", "今天中午吃什么"]:
        qvec = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
        best = max((cosine(qvec, v["vector"]), v) for v in vectors)
        print(f"  {question:<16} 最高分 {best[0]:.3f}  ← {best[1]['text'][:36]}")

    # 检查 4：向量数值是否正常
    print()
    print("-" * 62)
    print("检查 4：向量数值是否正常")
    print("-" * 62)

    print(f"  前 5 个值：{stored[:5]}")
    print(f"  最大值：{max(stored):.6f}  最小值：{min(stored):.6f}")
    all_zero = all(abs(v) < 1e-9 for v in stored)
    print(f"  是否全为 0：{'❌ 是！' if all_zero else '✅ 否'}")

    # 检查 5：向量之间的平均相似度
    print()
    print("-" * 62)
    print("检查 5：课件向量之间的平均相似度")
    print("-" * 62)

    sample = vectors[:30]
    scores = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            scores.append(cosine(sample[i]["vector"], sample[j]["vector"]))
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  前 30 块两两相似度的平均值：{avg:.3f}")
    if avg > 0.9:
        print("  ❌ 太高了，向量可能退化了")
    elif avg > 0.7:
        print("  ⚠️ 偏高，但可能正常（同类课件文本相似）")
    else:
        print("  ✅ 正常")


if __name__ == "__main__":
    main()
