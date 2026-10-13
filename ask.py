"""课件问答助手：检索 + 生成，给出带出处的答案。

用法：
    python ask.py                      进入交互模式
    python ask.py "什么是顺序结构"       直接问一个问题
    python ask.py -k 5 "分支结构"        指定检索块数

RAG 的完整流程：
    1. 用问题去 109 块课件里检索最相关的几块     ← 第 3、4 天已完成
    2. 把这几块原文 + 问题，拼成提示词
    3. 交给大模型，让它【只根据原文】回答
    4. 打印答案 + 标注出处
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
TOP_K = 3

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
CHAT_MODEL = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# ============ 提示词：RAG 的灵魂 ============
SYSTEM_PROMPT = """你是一个课程课件问答助手。请严格根据我提供的课件原文回答问题。

规则：
1. 只使用【课件原文】里的信息回答，不要编造。
2. 如果原文里没有答案，就直接说"课件里没有相关内容"，不要猜测。
3. 回答要简洁清楚，条理分明，适合大学生理解。
4. 如果原文里有代码示例，可以引用出来。
5. 不要提到"根据原文""根据上下文"这类话，直接回答问题本身。
"""

USER_TEMPLATE = """【课件原文】
{context}

【问题】
{question}
"""


def get_client() -> OpenAI:
    if not API_KEY or not BASE_URL:
        print("读不到 .env 配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def cosine(a, b) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def keyword_bonus(question: str, text: str) -> float:
    """关键词命中加分（混合检索）。"""
    tokens = [question[i:i + 2] for i in range(len(question) - 1)]
    if not tokens:
        return 0.0
    unique = set(tokens)
    hits = sum(1 for token in unique if token in text)
    return (hits / len(unique)) * 0.15


def retrieve(client: OpenAI, records, question: str, top_k: int):
    """第一步：检索出最相关的几块。"""
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    qvec = response.data[0].embedding

    scored = []
    for record in records:
        score = cosine(qvec, record["vector"]) + keyword_bonus(question, record["text"])
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def build_context(results) -> str:
    """第二步：把检索到的原文拼成上下文。"""
    parts = []
    for index, (score, record) in enumerate(results, start=1):
        header = f"[片段{index}] 来源：{record['source']} 第 {record['page']} 页"
        parts.append(f"{header}\n{record['text']}")
    return "\n\n".join(parts)


def generate(client: OpenAI, question: str, results) -> str:
    """第三步：交给大模型生成答案。"""
    context = build_context(results)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=context, question=question)},
        ],
        temperature=0.3,      # 低温度 = 更稳定、更少发挥
    )
    return response.choices[0].message.content


def answer(client: OpenAI, records, question: str, top_k: int, show_source: bool) -> None:
    """完整流程：检索 → 生成 → 打印。"""
    print()
    print("=" * 66)
    print(f"问题：{question}")
    print("=" * 66)

    results = retrieve(client, records, question, top_k)
    best_score = results[0][0] if results else 0.0

    # 检索分数太低，直接告诉用户课件里没有
    if best_score < THRESHOLD:
        print()
        print(f"  课件里没有找到相关内容（最高相关度 {best_score:.3f}）")
        print("  换个问法试试，或者这个问题超出了这三章的范围。")
        return

    print()
    print(f"  正在根据 {len(results)} 段课件内容生成答案...")

    text = generate(client, question, results)

    print()
    print("-" * 66)
    print(text.strip())
    print("-" * 66)

    if show_source:
        print()
        print("  参考来源：")
        for score, record in results:
            if score < THRESHOLD:
                continue
            snippet = record["text"][:50].replace("\n", " ")
            print(f"    · {record['source']} 第 {record['page']} 页"
                  f"（相关度 {score:.3f}）")
            print(f"      {snippet}...")


def parse_args():
    top_k = TOP_K
    show_source = True
    question = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--no-source":
            show_source = False
            i += 1
            continue
        question = args[i]
        i += 1

    return question, top_k, show_source


def main() -> None:
    if not VECTORS.exists():
        print("还没有 vectors.json，请先运行：python day3_embed.py")
        return

    question, top_k, show_source = parse_args()
    client = get_client()
    records = json.loads(VECTORS.read_text(encoding="utf-8"))

    stored_model = records[0].get("model", "未记录")
    if stored_model != "未记录" and stored_model != EMBED_MODEL:
        print(f"⚠️ 向量是用 {stored_model} 生成的，当前配置是 {EMBED_MODEL}")
        print("   请重新运行：python day3_embed.py")
        return

    print(f"课件问答助手（{len(records)} 块课件，检索 {top_k} 块，回答模型 {CHAT_MODEL}）")

    # 单次提问
    if question:
        answer(client, records, question, top_k, show_source)
        return

    # 交互模式
    print("输入问题回车即可，输入 q 退出。")
    print("-" * 66)

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

        try:
            answer(client, records, question, top_k, show_source)
        except Exception as exc:
            print(f"\n  出错了：{type(exc).__name__}: {exc}")

    print("\n再见。")


if __name__ == "__main__":
    main()
