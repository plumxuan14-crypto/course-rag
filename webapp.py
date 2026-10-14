"""课件问答助手 · 网页版（零依赖）

只用了 Python 标准库，不需要装 streamlit / flask。

运行：
    python webapp.py

然后浏览器打开：http://localhost:8501
停止：按 Ctrl + C
"""

import json
import math
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ============ 先加载 .env，再读任何环境变量 ============
load_dotenv(override=True)

VECTORS = Path("vectors.json")
PORT = 8501
THRESHOLD = 0.40
RETRY = 3

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
CHAT_MODEL = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

SYSTEM_PROMPT = """你是一个课程课件问答助手。请严格根据我提供的课件原文回答问题。

规则：
1. 只使用【课件原文】里的信息回答，不要编造。
2. 如果原文里没有答案，就直接说"课件里没有相关内容"，不要猜测。
3. 回答要简洁清楚，条理分明，适合大学生理解。
4. 如果原文里有代码示例，请用代码块包裹（三个反引号加 c）。
5. 不要提到"根据原文""根据上下文"这类话，直接回答问题本身。
"""

USER_TEMPLATE = """【课件原文】
{context}

【问题】
{question}
"""

# ============ 载入数据（启动时一次） ============
RECORDS = json.loads(VECTORS.read_text(encoding="utf-8"))
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_api(func, *args, **kwargs):
    """调用 API，网络失败时自动重试。"""
    last_error = None
    for attempt in range(RETRY):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            print(f"  [重试] 第 {attempt + 1}/{RETRY} 次失败：{type(exc).__name__}")
            if attempt < RETRY - 1:
                time.sleep(1.5)

    print(f"  [失败] API 调用连续 {RETRY} 次失败")
    raise last_error


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
    hits = sum(1 for token in unique if token in text)
    return (hits / len(unique)) * 0.15


def retrieve(question: str, top_k: int = 3):
    response = call_api(
        CLIENT.embeddings.create,
        model=EMBED_MODEL,
        input=[question],
    )
    qvec = response.data[0].embedding

    scored = []
    for record in RECORDS:
        score = cosine(qvec, record["vector"]) + keyword_bonus(question, record["text"])
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def generate(question: str, results) -> str:
    parts = []
    for index, (score, record) in enumerate(results, start=1):
        parts.append(
            f"[片段{index}] 来源：{record['source']} 第 {record['page']} 页\n{record['text']}"
        )
    context = "\n\n".join(parts)

    response = call_api(
        CLIENT.chat.completions.create,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=context, question=question)},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def do_answer(question: str, top_k: int) -> dict:
    """完整流程，返回给前端的数据。"""
    results = retrieve(question, top_k)
    best = results[0][0] if results else 0.0

    if best < THRESHOLD:
        return {
            "found": False,
            "answer": f"课件里没有找到相关内容（最高相关度 {best:.3f}）。\n\n"
                      "换个问法试试，或者这个问题超出了课件范围。",
            "sources": [],
        }

    text = generate(question, results)
    sources = [
        {
            "name": record["source"],
            "page": record["page"],
            "score": round(score, 3),
            "text": record["text"],
        }
        for score, record in results if score >= THRESHOLD
    ]
    return {"found": True, "answer": text, "sources": sources}


# ============ 前端页面 ============
# 注意：用 r""" 原始字符串，避免 Python 提前把 \n \s 这类转义吃掉
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>课件问答助手</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f5f6f8; color: #1f2328;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 26px; margin: 0 0 6px; }
  .sub { color: #656d76; font-size: 14px; margin-bottom: 22px; }
  .card {
    background: #fff; border: 1px solid #d8dee4; border-radius: 10px;
    padding: 18px; margin-bottom: 16px;
  }
  .row { display: flex; gap: 10px; }
  input[type=text] {
    flex: 1; padding: 12px 14px; font-size: 15px;
    border: 1px solid #d8dee4; border-radius: 8px; outline: none;
  }
  input[type=text]:focus { border-color: #0969da; }
  button {
    padding: 12px 22px; font-size: 15px; cursor: pointer;
    border: none; border-radius: 8px; background: #1f883d; color: #fff;
    font-weight: 600;
  }
  button:hover { background: #1a7f37; }
  button:disabled { background: #94d3a2; cursor: not-allowed; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 0; }
  .chip {
    padding: 7px 12px; font-size: 13px; cursor: pointer;
    background: #eef1f4; border: 1px solid #d8dee4; border-radius: 20px;
  }
  .chip:hover { background: #e3e8ed; }
  .answer { line-height: 1.8; font-size: 15px; }
  .answer pre {
    background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 8px;
    padding: 12px 14px; overflow-x: auto; margin: 10px 0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 13.5px; line-height: 1.6; white-space: pre;
  }
  .answer code {
    font-family: Consolas, "Courier New", monospace;
    background: #f6f8fa; padding: 1px 5px; border-radius: 4px; font-size: 13.5px;
  }
  .answer pre code { background: none; padding: 0; }
  .answer strong { font-weight: 700; }
  .answer ul { margin: 6px 0 10px; padding-left: 22px; }
  .answer li { margin-bottom: 4px; }
  .label { font-weight: 700; font-size: 15px; margin-bottom: 10px; }
  .warn { color: #9a6700; }
  .src { border-top: 1px dashed #d8dee4; padding-top: 12px; margin-top: 12px; }
  .src:first-child { border-top: none; padding-top: 0; margin-top: 0; }
  .src-head { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
  .score { color: #656d76; font-weight: 400; }
  .src-text {
    font-size: 13px; color: #424a53; white-space: pre-wrap;
    background: #f6f8fa; border-radius: 6px; padding: 10px;
    max-height: 180px; overflow-y: auto; line-height: 1.6;
  }
  .hint { color: #656d76; font-size: 13px; }
  .spin { color: #0969da; font-size: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>📚 课件问答助手</h1>
  <div class="sub">基于 RAG：先从课件里检索相关内容，再让 AI 照着回答。不知道就说不知道。</div>

  <div class="card">
    <div class="row">
      <input type="text" id="q" placeholder="例如：if 语句的格式是什么？" autocomplete="off">
      <button id="go">提问</button>
    </div>
    <div class="chips" id="chips"></div>
  </div>

  <div id="out"></div>

  <div class="hint" id="meta"></div>
</div>

<script>
const examples = ["if 语句的格式是什么", "switch 语句怎么用", "scanf 怎么用",
                  "为什么要学习编程", "什么是位运算"];
const chips = document.getElementById("chips");
examples.forEach(function (text) {
  const el = document.createElement("div");
  el.className = "chip";
  el.textContent = text;
  el.onclick = function () { document.getElementById("q").value = text; ask(); };
  chips.appendChild(el);
});

const out = document.getElementById("out");
const qInput = document.getElementById("q");
const goBtn = document.getElementById("go");

qInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter") ask();
});

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---- 轻量 Markdown 渲染：代码块、粗体、列表 ----
function renderMarkdown(text) {
  const codeBlocks = [];
  const NL = String.fromCharCode(10);
  let body = String(text);

  // 1. 先把 ``` 代码块抽出来，换成占位符保护起来
  const fence = String.fromCharCode(96, 96, 96);   // 就是三个反引号
  const parts = body.split(fence);

  if (parts.length >= 3) {
    body = parts[0];
    for (let i = 1; i + 1 < parts.length; i += 2) {
      let code = parts[i];
      // 去掉开头的语言标记（如 c、python）
      const firstBreak = code.indexOf(NL);
      if (firstBreak !== -1 && firstBreak <= 12) {
        const lang = code.slice(0, firstBreak).trim();
        if (/^[a-zA-Z0-9+#-]*$/.test(lang)) {
          code = code.slice(firstBreak + 1);
        }
      }
      codeBlocks.push(code.replace(/[\r\n]+$/, ""));
      body += String.fromCharCode(0) + "CB" + (codeBlocks.length - 1) + String.fromCharCode(0);
    }
    body += parts[parts.length - 1];
  }

  // 2. 转义 HTML，防止 XSS
  body = escapeHtml(body);

  // 3. 粗体 **文字**
  body = body.replace(/\*\*([^*<>]+)\*\*/g, "<strong>$1</strong>");

  // 4. 列表项 - 文字
  body = body.replace(/^[ \t]*[-*][ \t]+(.+)$/gm, "<li>$1</li>");
  body = body.replace(/(?:<li>[\s\S]*?<\/li>[\r\n]*)+/g, function (m) {
    return "<ul>" + m.replace(/[\r\n]+/g, "") + "</ul>";
  });

  // 5. 换行
  body = body.replace(/[\r\n]/g, "<br>");

  // 6. 把占位符换回真正的代码块
  body = body.replace(/\x00CB(\d+)\x00/g, function (m, index) {
    return "<pre><code>" + escapeHtml(codeBlocks[Number(index)]) + "</code></pre>";
  });

  return body;
}

async function ask() {
  const question = qInput.value.trim();
  if (!question) return;

  goBtn.disabled = true;
  out.innerHTML = '<div class="card"><div class="spin">正在检索并生成答案…（约 3-8 秒）</div></div>';

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: question, top_k: 3})
    });
    const data = await resp.json();

    if (data.error) {
      out.innerHTML = '<div class="card warn">出错了：' + escapeHtml(data.error) + '</div>';
      return;
    }

    let html = '<div class="card">';
    html += '<div class="label">💡 回答</div>';
    html += '<div class="answer' + (data.found ? '' : ' warn') + '">'
          + renderMarkdown(data.answer) + '</div>';

    if (data.sources && data.sources.length) {
      html += '<div style="margin-top:18px"><div class="label">📖 参考来源</div>';
      data.sources.forEach(function (s) {
        html += '<div class="src">';
        html += '<div class="src-head">' + escapeHtml(s.name)
              + ' 第 ' + s.page + ' 页 <span class="score">（相关度 ' + s.score + '）</span></div>';
        html += '<div class="src-text">' + escapeHtml(s.text) + '</div>';
        html += '</div>';
      });
      html += '</div>';
    }
    html += '</div>';
    out.innerHTML = html;

  } catch (err) {
    out.innerHTML = '<div class="card warn">请求失败：' + escapeHtml(String(err)) + '</div>';
  } finally {
    goBtn.disabled = false;
  }
}

goBtn.onclick = ask;

fetch("/api/meta").then(r => r.json()).then(function (m) {
  document.getElementById("meta").textContent =
    "课件 " + m.chunks + " 块　|　向量模型 " + m.embed_model + "　|　回答模型 " + m.chat_model;
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    """处理浏览器请求。"""

    def log_message(self, fmt, *args):
        """简化日志。"""
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/meta":
            data = {
                "chunks": len(RECORDS),
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
            }
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)

        question = payload.get("question", "").strip()
        top_k = int(payload.get("top_k", 3))

        print(f"  收到提问：{question}")

        try:
            result = do_answer(question, top_k)
            print("  回答完成")
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  出错：{type(exc).__name__}: {exc}")

        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    url = f"http://localhost:{PORT}"

    print("=" * 58)
    print("  课件问答助手 · 网页版")
    print("=" * 58)
    print(f"  课件块数：{len(RECORDS)}")
    print(f"  向量模型：{EMBED_MODEL}")
    print(f"  回答模型：{CHAT_MODEL}")
    print()
    print(f"  请在浏览器打开：{url}")
    print()
    print("  停止服务：按 Ctrl + C")
    print("=" * 58)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
