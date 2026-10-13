"""第 4 天：从 109 块课件里，找出和问题最相关的几块。

用法：
    python day4_search.py                     进入交互模式，随便问
    python day4_search.py "什么是顺序结构"      直接问一个问题
    python day4_search.py -k 5                改检索块数（默认 3）

说明：
    这一步只做【检索】，不做【回答】。
    它会把最相关的原文找出来给你看，但不会替你说人话。
    第 5 天我们接上模型，让它根据这些原文回答你。
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ============ 先加载 .env，再读任何环境变量 ============
load_dotenv(override=True)

VECTORS = Path("vectors.json")
BATCH = 10
THRESHOLD = 0.40

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")


def get_client() -> OpenAI:
    if not API_KEY or not BASE_URL:
        print("读不到 .env 配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def cosine(a, b) -> float:
    """余弦相似度，越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def keyword_bonus(question: str, text: str) -> float:
    """关键词命中加分：字面上包含问句里的词，就给一点奖励。

    这是"混合检索"的简化版：向量管"意思像"，关键词管"字面准"。
    """
    tokens = [question[i:i + 2] for i in range(len(question) - 1)]
    if not tokens:
        return 0.0
    unique = set(tokens)
    hits = sum(1 for token in unique if token in text)
    return (hits / len(unique)) * 0.15      # 最多加 0.15 分


def search(client: OpenAI, records, question: str, top_k: int):
    """检索：返回 [(最终得分, 向量分, 关键词加分, 记录), ...]。"""
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    qvec = response.data[0].embedding

    scored = []
    for record in records:
        vec_score = cosine(qvec, record["vector"])
        bonus = keyword_bonus(question, record["text"])
        scored.append((vec_score + bonus, vec_score, bonus, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def show(question: str, results, verbose: bool = True) -> None:
    """打印检索结果。"""
    print()
    print("=" * 64)
    print(f"问题：{question}")
    print("=" * 64)

    if not results or results[0][0] < THRESHOLD:
        best = results[0][0] if results else 0.0
        print(f"  ✗ 课件里没有找到相关内容（最高分 {best:.3f}，阈值 {THRESHOLD}）")
        return

    for rank, (total, vec_score, bonus, record) in enumerate(results, start=1):
        if total < THRESHOLD:
            break
        print()
        print(f"  [{rank}] 综合 {total:.3f}  "
              f"(语义 {vec_score:.3f} + 关键词 {bonus:+.3f})")
        print(f"      来源：{record['source']}  第 {record['page']} 页")

        if verbose:
            print("      " + "-" * 56)
            for line in record["text"][:400].splitlines():
                print(f"      {line}")


def parse_args():
    """解析命令行参数。"""
    top_k = 3
    question = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
            continue
        question = args[i]
        i += 1

    return question, top_k


def main() -> None:
    if not VECTORS.exists():
        print("还没有 vectors.json，请先运行：python day3_embed.py")
        return

    question, top_k = parse_args()
    client = get_client()
    records = json.loads(VECTORS.read_text(encoding="utf-8"))

    stored_model = records[0].get("model", "未记录")
    print(f"已载入 {len(records)} 块课件的向量")
    print(f"向量模型：{stored_model}")
    print(f"当前配置：{EMBED_MODEL}")
    if stored_model != "未记录" and stored_model != EMBED_MODEL:
        print()
        print("  ⚠️ 警告：向量是用【另一个模型】生成的，检索结果会不准！")
        print("     请重新运行：python day3_embed.py")
        return
    print(f"检索 {top_k} 块，阈值 {THRESHOLD}")

    # 单次提问模式
    if question:
        show(question, search(client, records, question, top_k))
        return

    # 交互模式
    print()
    print("进入交互模式。输入问题回车即可，输入 q 退出。")
    print("-" * 64)

    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit", "退出"):
            break

        show(question, search(client, records, question, top_k))

    print("\n再见。")


if __name__ == "__main__":
    main()
