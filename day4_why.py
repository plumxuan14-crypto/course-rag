"""诊断：为什么脚本用的是 v3 而不是 .env 里的 v4？

用法：
    python day4_why.py
"""

import os
import sys
from pathlib import Path

print("=" * 62)
print("诊断：配置从哪儿来")
print("=" * 62)

# 1. 当前工作目录
print()
print("【1】当前工作目录")
print(f"    {Path.cwd()}")

# 2. .env 文件情况
print()
print("【2】.env 文件检查")
env_path = Path(".env")
if env_path.exists():
    print(f"    找到 .env：{env_path.resolve()}")
    print(f"    大小：{env_path.stat().st_size} 字节")
    print("    内容（只看 DASHSCOPE 开头的行）：")
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("DASHSCOPE"):
            if "KEY" in line:
                name = line.split("=")[0]
                print(f"      {name}=<已隐藏>")
            else:
                print(f"      {line}")
else:
    print("    ❌ 当前目录没有 .env 文件！")

# 3. 看看有没有别的 .env（往上翻目录）
print()
print("【3】往上层目录找 .env")
here = Path.cwd()
for parent in [here] + list(here.parents)[:3]:
    candidate = parent / ".env"
    if candidate.exists():
        print(f"    发现：{candidate}")

# 4. 加载前，环境里有什么
print()
print("【4】load_dotenv() 之前，环境里的 DASHSCOPE 变量")
before = {k: v for k, v in os.environ.items() if k.startswith("DASHSCOPE")}
if before:
    for k, v in before.items():
        print(f"    {k} = {v}")
else:
    print("    （空的，没有任何 DASHSCOPE 变量）")

# 5. 加载 .env
print()
print("【5】执行 load_dotenv()")
from dotenv import load_dotenv

loaded = load_dotenv()
print(f"    load_dotenv() 返回：{loaded}   （True=找到并加载了文件）")

# 6. 加载后，环境里有什么
print()
print("【6】加载之后，环境里的 DASHSCOPE 变量")
after = {k: v for k, v in os.environ.items() if k.startswith("DASHSCOPE")}
if after:
    for k, v in after.items():
        if "KEY" in k:
            print(f"    {k} = <已隐藏，长度 {len(v)}>")
        else:
            print(f"    {k} = {v}")
else:
    print("    （空的！说明 .env 没被加载）")

# 7. 脚本实际会用的值
print()
print("【7】脚本实际会用到的模型")
model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
print(f"    DASHSCOPE_EMBED_MODEL = {model}")

if model == "text-embedding-v3":
    print()
    print("    ⚠️ 用的是默认值 v3（说明环境变量没读到）")
    print("    或者 .env 里确实写着 v3")
elif model == "text-embedding-v4":
    print()
    print("    ✅ 读到 v4 了，配置正常")

# 8. 找找项目里有没有硬编码 v3 的地方
print()
print("【8】在 .py 文件里搜索 'text-embedding'")
for py in sorted(Path(".").glob("*.py")):
    if py.name == "day4_why.py":
        continue
    try:
        text = py.read_text(encoding="utf-8")
    except Exception:
        continue
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "text-embedding" in line:
            print(f"    {py.name}:{lineno}  {line.strip()}")

print()
print("=" * 62)
