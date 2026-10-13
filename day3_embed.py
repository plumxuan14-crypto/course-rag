"""第 3 天：把每块文字变成向量，存成 vectors.json。

用法：
    python day3_embed.py            调用 API 算向量
    python day3_embed.py --test     做相似度实验
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

CHUNKS = Path("chunks.json")
VECTORS = Path("vectors.json")
BATCH = 10          # 每次发给 API 的块数（阿里云 embedding 单次最多 10 条）
MIN_SCORE = 0.35    # 相似度低于这个值，认为"课件里没有"


def get_client():
    """读取 .env，创建 API 客户端。"""
    load_dotenv()
    key = os.getenv("DASHSCOPE_API_KEY")
    base = os.getenv("DASHSCOPE_BASE_URL")
    if not key or not base:
        print("读不到 .env 里的配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        sys.exit(1)
    return OpenAI(api_key=key, base_url=base)


def do_embed() -> None:
    model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    client = get_client()

    print(f"载入 {len(chunks)} 块文字")
    print(f"使用模型：{model}")

    vectors = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(model=model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend([item.embedding for item in ordered])

        done = min(start + BATCH, len(chunks))
        print(f"  已处理 {done}/{len(chunks)}")

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "vector": [round(float(v), 6) for v in vector],
        })

    VECTORS.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    size_mb = VECTORS.stat().st_size / 1024 / 1024
    print("=" * 50)
    print(f"共 {len(records)} 条向量，维度 {len(records[0]['vector'])}")
    print(f"已保存到：{VECTORS.resolve()}  ({size_mb:.1f} MB)")
    print("=" * 50)


def cosine(a, b) -> float:
    """余弦相似度：越接近 1 表示意思越像。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def do_test() -> None:
    if not VECTORS.exists():
        print("还没有 vectors.json，先运行：python day3_embed.py")
        return

    model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
    records = json.loads(VECTORS.read_text(encoding="utf-8"))
    client = get_client()

    questions = [
        "什么是顺序结构？",
        "分支结构有哪几种？",
        "循环和分支有什么区别？",
        "今天中午吃什么？",
        "量子力学的测不准原理是什么？",
    ]

    for question in questions:
        response = client.embeddings.create(model=model, input=[question])
        qvec = response.data[0].embedding

        scored = sorted(
            ((cosine(qvec, r["vector"]), r) for r in records),
            key=lambda item: item[0],
            reverse=True,
        )

        print()
        print("=" * 60)
        print(f"问题：{question}")
        print("=" * 60)
        for score, record in scored[:3]:
            text = record["text"][:70].replace("\n", " ")
            print(f"  {score:.3f}  [{record['source'][:8]}.. 第{record['page']}页] {text}")

        top = scored[0][0]
        if top < MIN_SCORE:
            print(f"  → 最高分只有 {top:.3f}，课件里大概没有相关内容")


def main() -> None:
    if "--test" in sys.argv:
        do_test()
    else:
        do_embed()


if __name__ == "__main__":
    main()
