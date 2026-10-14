"""网络诊断：排查 API 连不上的原因。

用法：
    python diag_net.py
"""

import os
import socket
import ssl
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

print("=" * 62)
print("网络诊断")
print("=" * 62)

# 1. 配置检查
print()
print("【1】当前配置")
print(f"    BASE_URL：{BASE_URL}")
print(f"    API_KEY ：{'已设置，长度 ' + str(len(API_KEY)) if API_KEY else '❌ 没读到'}")

# 2. 代理环境变量
print()
print("【2】代理环境变量（会干扰 API 调用）")
proxy_vars = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
if proxy_vars:
    for k, v in proxy_vars.items():
        print(f"    ⚠️ {k} = {v}")
    print("    → 如果这些代理不可用，API 会连不上")
else:
    print("    ✅ 没有代理设置")

# 3. DNS 解析
print()
print("【3】DNS 解析")
host = urlparse(BASE_URL).hostname or ""
print(f"    域名：{host}")
try:
    ip = socket.gethostbyname(host)
    print(f"    ✅ 解析到：{ip}")
except Exception as exc:
    print(f"    ❌ 解析失败：{exc}")

# 4. TCP 连接（是否通）
print()
print("【4】TCP 连接测试（443 端口）")
try:
    sock = socket.create_connection((host, 443), timeout=10)
    print("    ✅ TCP 连接成功")
    sock.close()
except Exception as exc:
    print(f"    ❌ TCP 连接失败：{type(exc).__name__}: {exc}")

# 5. HTTPS 请求
print()
print("【5】HTTPS 请求测试")
try:
    import httpx

    # 强制不使用代理
    with httpx.Client(trust_env=False, timeout=15) as client:
        r = client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
        print(f"    ✅ 不走代理：HTTP {r.status_code}")
except Exception as exc:
    print(f"    ❌ 不走代理失败：{type(exc).__name__}: {exc}")

try:
    import httpx

    # 使用环境里的代理设置
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
        print(f"    ✅ 走环境代理：HTTP {r.status_code}")
except Exception as exc:
    print(f"    ❌ 走环境代理失败：{type(exc).__name__}: {exc}")

# 6. 真实调用 embeddings
print()
print("【6】真实调用 embeddings 接口")
try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    r = client.embeddings.create(
        model=os.getenv("DASHSCOPE_EMBED_MODEL"),
        input=["测试"],
    )
    print(f"    ✅ 调用成功，维度 {len(r.data[0].embedding)}")
except Exception as exc:
    print(f"    ❌ 调用失败：{type(exc).__name__}: {exc}")

print()
print("=" * 62)
