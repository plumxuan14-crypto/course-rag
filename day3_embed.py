"""第 3 天：把每块文字变成向量，存成 vectors.json。

用法：
    python day3_embed.py            调用 API 算向量
    python day3_embed.py --test     做相似度实验

注意：
    .env 必须在【模块最开始】就加载好，否则后面读环境变量会拿到默认值。
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ============ 先加载 .env，再读任何环境变量 ============
# override=True：.env 里的值优先，强行覆盖环境里的同名变量
load_dotenv(override=True)

CHUNKS = Path("chunks.json")
VECTORS = Path("vectors.json")
BATCH = 10
MIN_SCORE = 0.35

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")


def get_client() -> OpenAI:
    """创建 API 客户端。"""
    if not API_KEY or not BASE_URL:
        print("读不到 .env 配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def do_embed() -> None:
    if not CHUNKS.exists():
        print(f"找不到 {CHUNKS}，请先运行：python day2_chunk.py")
        return

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    client = get_client()

    print(f"载入 {len(chunks)} 块文字")
    print(f"使用模型：{EMBED_MODEL}")

    vectors = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
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
            # 记录这条向量是哪个模型生成的，将来可以校验
            "model": EMBED_MODEL,
        })

    VECTORS.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    size_mb = VECTORS.stat().st_size / 1024 / 1024
    print("=" * 50)
    print(f"共 {len(records)} 条向量，维度 {len(records[0]['vector'])}")
    print(f"使用的模型：{EMBED_MODEL}")
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

    records = json.loads(VECTORS.read_text(encoding="utf-8"))
    client = get_client()

    print(f"已存向量用的模型：{records[0].get('model', '未记录')}")
    print(f"当前 .env 模型：{EMBED_MODEL}")

    questions = [
        "什么是顺序结构？",
        "分支结构有哪几种？",
        "scanf 怎么用？",
        "今天中午吃什么？",
        "量子力学的测不准原理是什么？",
    ]

    for question in questions:
        response = client.embeddings.create(model=EMBED_MODEL, input=[question])
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
            print(f"  {score:.3f}  [{record['source'][:10]} 第{record['page']}页] {text}")

        if scored[0][0] < MIN_SCORE:
            print(f"  → 最高分只有 {scored[0][0]:.3f}，课件里大概没有相关内容")


def main() -> None:
    if "--test" in sys.argv:
        do_test()
    else:
        do_embed()


if __name__ == "__main__":
    main()
