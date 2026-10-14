# course-rag

一个能回答课程课件问题的问答机器人。问它「if 语句的格式是什么」，它会从课件里找到相关内容，用大白话回答并告诉你出自第几页。

**核心特点：只根据课件原文回答，课件里没有的内容它会直接说「不知道」，不会编造。**

---

## 目录

- [这是什么](#这是什么)
- [效果如何](#效果如何)
- [怎么做的](#怎么做的)
- [遇到过什么问题](#遇到过什么问题)
- [怎么运行](#怎么运行)
- [项目结构](#项目结构)
- [下一步想做什么](#下一步想做什么)

---

## 这是什么

### 要解决的问题

ChatGPT 很聪明，但它**没看过我们老师的课件**。

问它「我们老师讲的顺序结构是什么」，它只能泛泛而谈，甚至编一个听起来合理的答案。这种现象叫**幻觉（hallucination）**。

### 解决思路：RAG

**RAG = Retrieval-Augmented Generation（检索增强生成）**

打个比方：**让 AI 开卷考试。**

```
普通用法（闭卷）：
  你问 → AI 凭记忆回答 → 可能记错，可能编造

RAG（开卷）：
  你问 → 先去课件里翻出最相关的几页
       → 把这几页和问题一起交给 AI
       → AI 照着这几页回答，并告诉你答案在哪一页
```

**这三个字母对应的正是完整流程**：

| 字母 | 含义 | 在本项目里 |
|---|---|---|
| **R** | Retrieval 检索 | 从 109 块课件里找出最相关的 3 块 |
| **A** | Augmented 增强 | 把这 3 块原文拼进提示词 |
| **G** | Generation 生成 | 大模型根据这些原文回答 |

---

## 效果如何

### 检索准确率：80%

用 10 个带标准答案的问题做评测（`day4_eval.py`）：

```
章节命中率：  8/10 = 80%     （Top3 结果里包含了正确章节）
关键词命中率：8/10 = 80%     （Top3 结果里包含了期望的关键词）
```

**典型成功案例**：

```
问题：if 语句的格式

检索到的内容：
  · 第3章 分支结构.pptx 第 6 页（相关度 0.817）
  · 第3章 分支结构.pptx 第 5 页（相关度 0.754）
  · 第3章 分支结构.pptx 第 13 页（相关度 0.741）

回答：
  if 语句有以下三种格式：

  格式1（单语句）：
  if(表达式) 单语句

  格式2（语句块）：
  if(表达式) {
      分支语句块;
  }

  格式3（多分支，if-else if-else）：
  ...
```

### 防幻觉验证

```
问题：今天中午吃什么

回答：课件里没有找到相关内容（最高相关度 0.313）

问题：量子力学的测不准原理是什么

回答：课件里没有找到相关内容
```

**它能分辨「课件里有」和「课件里没有」，这是 RAG 可用的前提。**

### 20% 的失败是为什么

失败案例集中在**内容稀疏的目录页**：

```
问题：什么是顺序结构

检索结果（错误）：
  · 第3章 分支结构.pptx 第 6 页（相关度 0.551）
    内容：1-7 THANKYOU 二、if嵌套 一、if结构 三、switch开关语句
```

**问题在于**：这一页是**目录页**，列出了十几个主题，所以它和**任何**技术问题都"沾点边"，相似度虚高，挤掉了真正讲定义的那一页。

**而且**：我这份课件第 2 章虽然标题叫「顺序结构」，但内容主要是位运算、scanf/printf、数据类型，**并没有「顺序结构」的正式定义**。所以检索失败之外，语料本身也缺这块内容。

---

## 怎么做的

### 整体流程

```
① 课件（PPTX / PDF）
        ↓  day1_read_pdf.py
② 纯文本（extracted.txt，116 页，13367 字）
        ↓  day2_chunk.py
③ 文本块（chunks.json，109 块，平均 153 字）
        ↓  day3_embed.py
④ 向量（vectors.json，1024 维）
        ↓  day4_search.py / ask.py
⑤ 提问 → 找最相似的 3 块 → 交给大模型 → 带出处的答案
```

### 关键设计决策

#### 1. 为什么切块是 600 字 / 重叠 100 字

**必须切块**，因为：
- 模型一次吃不下 13367 字
- 检索时「整本书」太粗，没法精确定位

**必须有重叠**，因为：

```
如果切块边界正好落在一句话中间：

块1: ......用第一式减去第二式的
块2: 2倍，可以得到下面的结果......

有人问「消元过程中怎么处理系数」时，
块1 和块2 各只有半句，相似度都被稀释，两块都搜不到。
重叠 100 字就是为了兜住跨边界的内容。
```

**实测结果**：109 块，最短 52 字，最长 499 字，平均 153 字。

**针对幻灯片的特殊处理**：课件里很多页只有一行标题（平均只有 115 字/页），直接切会产生大量碎片块。解决方法是：

```
一页只有 5 个字（标题）
    ↓ 不丢弃
    ↓ 保留它，但借用上一块的内容补足到 50 字以上
```

实测：**37 次借用上文，只丢弃 2 个过短的块**。

#### 2. 为什么用向量检索，而不是关键词搜索

**关键词搜索只能字面匹配**：

```
搜「排列里前面比后面大的数对」
    ↓
找不到写着「逆序数」的那一页 —— 因为字面对不上
```

**向量检索是按意思匹配**：

```
嵌入模型把每段文字变成一串数字（1024 个）
    ↓
意思相近的文字，这串数字也相近
    ↓
「找意思相近的」= 「找坐标最近的」
```

**实测的语义区分能力**：

| 类型 | 平均相似度 |
|---|---|
| 相关问题（如「scanf 怎么用」） | **0.677** |
| 无关问题（如「明朝为什么灭亡」） | **0.311** |
| **差距** | **+0.366** ✅ |

差距 0.366 说明模型能清楚区分「相关」和「不相关」。

#### 3. 混合检索：向量 + 关键词

**纯向量的问题**：对代码术语（`scanf`、`switch`）不够敏感。

**解决**：在向量分数上叠加关键词命中加分。

```python
def keyword_bonus(question, text):
    """问句里的 2 字片段出现在原文里，就加分（最多 +0.15）"""
    tokens = [question[i:i+2] for i in range(len(question) - 1)]
    unique = set(tokens)
    hits = sum(1 for t in unique if t in text)
    return (hits / len(unique)) * 0.15
```

**这是工业界 RAG 的标准做法**：向量负责「同义表达」，关键词负责「精确术语」。

#### 4. 防幻觉的提示词设计

这是 `ask.py` 里的 `SYSTEM_PROMPT`：

```python
SYSTEM_PROMPT = """你是一个课程课件问答助手。请严格根据我提供的课件原文回答问题。

规则：
1. 只使用【课件原文】里的信息回答，不要编造。
2. 如果原文里没有答案，就直接说"课件里没有相关内容"，不要猜测。
3. 回答要简洁清楚，条理分明，适合大学生理解。
4. 如果原文里有代码示例，可以引用出来。
5. 不要提到"根据原文""根据上下文"这类话，直接回答问题本身。
"""
```

**第 2 条是关键**：给它「弃权」的权利。

如果不写这条，模型遇到没答案的问题会硬编一个听起来合理的答案。

**另外**：`temperature=0.3`（低温度 = 更稳定、更少发挥）。

#### 5. 向量模型选型：v3 → v4

用同样的评测集对比两个模型：

| 模型 | 相关问题平均分 | 无关问题平均分 | 差距 |
|---|---|---|---|
| text-embedding-v3 | 0.671 | **0.420** ⚠️ | +0.251 |
| **text-embedding-v4** | 0.677 | **0.311** ✅ | **+0.366** |

**关键在「无关分」**：v3 会给无关内容 0.42 分，v4 只给 0.31。v4 的区分度明显更好，而且价格一样，所以换用 v4。

---

## 遇到过什么问题

**这一节是项目的精华 —— 全部是真实踩过的坑。**

### 1. Python 装了但命令行找不到

**现象**：

```
PS> python --version
Python was not found; run without arguments to install from the Microsoft Store
```

**原因**：安装 Python 时没勾选 `Add python.exe to PATH`，所以系统不知道 `python` 命令在哪。

**解决**：重新运行安装包 → Modify → 勾上 `Add Python to environment variables`。或者手工把安装目录加进用户 PATH。

### 2. pip 下载超时

**现象**：

```
SSL: UNEXPECTED_EOF_WHILE_READING
Failed to establish a secure connection to files.pythonhosted.org
```

**原因**：pip 默认从国外源下载，国内访问经常被中断。

**解决**：换清华镜像（一劳永逸）：

```powershell
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. PDF 提取出来是乱码

**现象**：提取出的文字长这样：

```
2c̃aD̆g̈8 ̃" /ç@gsinθ ≠ 0d̆± / Dn = / |||||||2 cosθ 11 2 cosθ 11
```

**原因**：PPT 导出 PDF 时，字体没有嵌入完整的字符映射表（ToUnicode），`pypdf` 只能拿到字形编号，猜不出是什么字。

**解决**：从 `pypdf` 换成 `pdfplumber`。

```python
# 旧
from pypdf import PdfReader
reader = PdfReader(path)
text = reader.pages[0].extract_text()

# 新
import pdfplumber
with pdfplumber.open(path) as pdf:
    text = pdf.pages[0].extract_text()
```

**效果**：中文正常了，公式仍不完美（PDF 里公式是二维排版，纯文本只能线性输出，下标位置信息会丢失）。

### 4. PowerPoint 临时文件导致读取失败

**现象**：

```
pptX.exc.PackageNotFoundError: Package not found at '~$第1章 概论2026.pptx'
```

**原因**：PPT 正在 PowerPoint 里打开着，Office 会生成一个 `~$xxx.pptx` 的隐藏锁文件（0 字节）。代码用 `*.pptx` 扫描时把它也当成课件了，但它不是有效的 PPT，读不了。

**解决**：代码里跳过这类文件。

```python
for path in Path(".").glob("*.pptx"):
    if path.name.startswith("~$"):
        continue      # 跳过 Office 锁文件
    files.append(path)
```

**附带教训**：这个文件还**删不掉**（被 Office 进程持有句柄），报错是「正由另一进程使用」。**即使不删也没关系，代码跳过它就行。**

### 5. 中文输入变乱码，检索分数掉到 0.06

**现象**：问「什么是顺序结构」，检索最高分只有 **0.065**（正常应该是 0.6+）。

**原因**：Windows PowerShell 5.1 默认用 GBK 编码读写控制台，而 Python 期望 UTF-8。所以**输入的中文到 Python 里已经变成乱码了**，拿去算向量自然对不上。

**解决**：每个新窗口设置三个编码（**三条缺一不可**）：

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8    # 这条最容易漏
```

**验证方法**：

```powershell
python -c "s=input('输入点什么: '); print('你输入的是:', s)"
```

输入中文，看打印出来是不是原样。

**更好的做法**：写进 PowerShell 配置文件 `$PROFILE`，每个新窗口自动生效。

### 6. `load_dotenv` 初始化顺序 bug（最难查的一个）

**现象**：`.env` 文件里明明写着 `text-embedding-v4`，但程序运行时打印的是 `text-embedding-v3`，导致生成的向量和查询用的模型不一致，**检索分数全是 0.065**。

**原因**：代码加载 `.env` 的时机太晚。

```python
def get_client():
    load_dotenv()          # ← .env 在这里才加载
    ...

def do_embed():
    model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
    #      ↑ 这行在 get_client() 之前执行
    #        此时 .env 还没加载 → 读不到 → 用默认值 v3！
    ...
    client = get_client()   # .env 到这里才加载，太晚了
```

**排查过程**（依次排除）：

1. 先怀疑向量和文字错位 → 写自检脚本对比 id，**结果完全一致**
2. 再怀疑环境变量残留 → `Get-ChildItem Env:DASHSCOPE*`，**结果是空的**
3. 检查 `.env` 文件本身 → **内容完全正确（就是 v4）**
4. 打印 `load_dotenv()` 的返回值和加载前后的环境变量 → **加载后就是 v4，正常**
5. **最后搜索所有 `.py` 文件里的 `text-embedding` 字面量** → 发现 `day3_embed.py:35` 有个 `os.getenv(..., "text-embedding-v3")` 的默认值，而它执行在 `load_dotenv()` 之前

**解决**：

```python
# ==== 模块最顶部，先加载 .env，再读任何环境变量 ====
load_dotenv(override=True)

EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v3")
```

**`override=True`** 的作用：让 `.env` 里的值优先，强制覆盖环境里的同名变量（默认行为是环境变量优先，很容易踩坑）。

**教训**：

> **静默失败比崩溃难查一百倍。** 这个 bug 不报错，程序照常运行，只是用了错误的值。

**顺带加的防护**：在 `vectors.json` 里记录生成时用的模型名，启动时自动校验：

```python
stored_model = records[0].get("model", "未记录")
if stored_model != EMBED_MODEL:
    print("⚠️ 向量是用另一个模型生成的，请重新运行 day3_embed.py")
    return
```

### 7. 索引过期（stale index）

**现象**：换了向量模型（v3 → v4）之后，**没有重新生成 `vectors.json`**，导致：

```
vectors.json 里是 v3 算的向量
.env 配置的是 v4
        ↓
两者语义空间不同 → 相似度接近 0（0.0165）
```

**这是「索引过期」问题** —— 真实生产环境里非常常见。换 embedding 模型意味着**要重建整个索引库**（可能几百万条）。

**工业界的标准做法**：

1. **记录索引是用哪个模型生成的**（存进元数据）
2. **启动时校验**：当前模型 ≠ 索引模型 → 报警或自动重建
3. **大系统用双写**：新旧索引并存，灰度切换

**解决**：改完模型配置后，必须重新运行 `day3_embed.py`。

### 8. Python 3.15 装不了 PyTorch

**现象**：

```
ERROR: ResolutionImpossible
some packages in these conflicts have no matching distribution: torch
```

**原因**：Python 3.15 太新，PyTorch 官方还没提供对应版本的 wheel。

**解决**：改用 API 方案（阿里云百炼的 `text-embedding-v4`），绕开本地 PyTorch 依赖。好处是不用装几个 GB 的包，速度快，效果还更好。

**如果以后要用本地模型**（跑 Transformer、微调），需要另外装一个 Python 3.12。

### 9. 目录页「通吃」导致检索失败

**现象**：问「什么是顺序结构」，检索到的是第 3 章的目录页，而不是正文。

**原因**：目录页列出十几个主题，和**任何**技术问题都「沾点边」，相似度虚高。

**当前状态**：**未修复**。这是那 20% 失败案例的主要成因。

**候选解法**：

```python
def is_directory_page(text):
    """判断目录页：行多、行短、没有句子特征"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    short = sum(1 for l in lines if len(l) <= 18)
    if short / len(lines) < 0.7:
        return False
    signals = ["是指", "就是", "表示", "例如", "称为", "定义", "作用"]
    return not any(s in text for s in signals)
```

---

## 怎么运行

### 环境要求

- Windows（Linux / macOS 同理）
- Python 3.10 以上（本项目在 3.15 上跑通，但注意 PyTorch 装不了）
- 阿里云百炼 API Key（有免费额度）

### 步骤

**1. 安装依赖**

```powershell
python -m pip install pypdf pdfplumber python-pptx openai python-dotenv
```

**2. 配置 API Key**

在项目根目录创建 `.env` 文件：

```
DASHSCOPE_API_KEY=你的key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBED_MODEL=text-embedding-v4
DASHSCOPE_CHAT_MODEL=qwen-plus
```

> ⚠️ **`.env` 必须写进 `.gitignore`**，绝对不能提交到 GitHub。

**3. 把课件放进项目目录**

支持 `.pptx` 和 `.pdf`。文件名随意，代码会自动找。

**4. 依次运行**

```powershell
# ① 提取文字（生成 extracted.txt）
python day1_read_pdf.py --extract

# ② 切块（生成 chunks.json）
python day2_chunk.py

# ③ 生成向量（生成 vectors.json，需要联网调 API）
python day3_embed.py

# ④ 开始提问
python ask.py
```

**5. 验证效果**

```powershell
# 自检：向量一致性、维度、检索分数
python day4_check.py

# 评测：10 个问题的命中率
python day4_eval.py
```

### 换课件怎么办

代码和数据是分开的，换课件只要三步：

```powershell
Remove-Item *.pptx, *.pdf, extracted.txt, chunks.json, vectors.json
# 把新课件复制进来
python day1_read_pdf.py --extract
python day2_chunk.py
python day3_embed.py
```

**代码一行都不用改。**

---

## 项目结构

```
course-rag/
├── .env                      # API 配置（不提交）
├── .gitignore
├── README.md
│
├── 第1章 概论2026.pptx        # 原始课件（不提交）
├── 第2章 2026 顺序结构.pptx
├── 第3章 分支结构.pptx
│
├── day1_read_pdf.py          # ① 课件 → extracted.txt
├── day2_chunk.py             # ② 文字 → chunks.json
├── day3_embed.py             # ③ 块 → vectors.json（调 API）
├── day4_search.py            # ④ 纯检索（只看原文）
├── ask.py                    # ⑤ 完整问答（检索 + 生成）
│
├── day4_check.py             # 自检工具
├── day4_eval.py              # 评测集（10 个问题）
├── day4_why.py               # 配置来源诊断
├── day5_filter.py            # 目录页过滤实验
├── check_progress.py         # 进度总览
│
├── extracted.txt             # 提取的纯文本
├── chunks.json               # 切好的块
└── vectors.json              # 向量（1.1 MB）
```

---

## 下一步想做什么

### 已完成

- [x] 课件解析（PPTX / PDF）
- [x] 文本切块（含短页处理和碎片过滤）
- [x] 向量化（text-embedding-v4，1024 维）
- [x] 混合检索（向量 + 关键词）
- [x] 生成答案 + 出处标注
- [x] 防幻觉
- [x] 评测集（80% 命中率）

### 计划中

- [ ] **网页版界面**（Streamlit，浏览器里用，不再敲命令行）
- [ ] **目录页过滤**（把那 20% 的失败案例修掉）
- [ ] **补充语料**（加入教材、讲义、课堂笔记 —— 这是提升效果最有效的办法）
- [ ] **多轮对话**（现在每次提问都是独立的，不能追问）
- [ ] **按主题切块**（现在是按页切，跨页的知识点会被切断）

### 已知的局限

1. **语料太少**：只有 13367 字（116 页 PPT），平均每页 115 字，很多页只有标题。**这是效果的主要瓶颈。**
2. **公式提取不完整**：PPT 里的数学公式是二维排版，转成纯文本会丢失结构信息。
3. **图片内容无法提取**：流程图、示意图里的文字提取不到，需要 OCR 或多模态模型。
4. **目录页干扰**：列出多个主题的页会挤掉正文页。
5. **单轮问答**：不支持「那它和 switch 有什么区别」这样的追问。

---

## 技术栈

| 环节 | 用了什么 |
|---|---|
| 课件解析 | `python-pptx`、`pypdf`、`pdfplumber` |
| 向量模型 | 阿里云百炼 `text-embedding-v4`（1024 维） |
| 生成模型 | 阿里云百炼 `qwen-plus` |
| API 客户端 | `openai`（百炼兼容 OpenAI 格式） |
| 配置管理 | `python-dotenv` |
| 相似度计算 | 余弦相似度（手写实现） |

**没有用任何 RAG 框架**（如 LangChain / LlamaIndex）—— 全部逻辑手写，为了彻底理解每一步在做什么。

### 10. Python 3.15 装不了 streamlit

**现象**：

    error: Microsoft Visual C++ 14.0 or greater is required
    ERROR: Failed to build 'pyarrow' when installing build dependencies

**原因**：streamlit 依赖 pyyaml 和 pyarrow，这两个包都没有 Python 3.15 的
预编译 wheel，pip 只能从源码编译，而编译需要 Visual Studio 的 C++ 工具链。

**这已经是第二次因为 3.15 太新而踩坑**（第一次是 PyTorch）。

**解决**：改用 Python 标准库自写网页版（`http.server`），零依赖：

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

**教训**：生产环境不要用刚发布的大版本。Python 3.15 发布初期，
大量库还没适配。

### 11. Git 分叉：网页提交和本地提交冲突

**现象**：

    git push
     ! [rejected]  main -> main (non-fast-forward)
    error: failed to push some refs

**原因**：在 GitHub 网页上创建 LICENSE，同时本地也有新提交（网页版代码），
两条线各自前进，历史「分叉」了。

**解决**：

    git pull      # 拉取远程并合并
    git push      # 再推送

**坑中坑**：`git pull` 触发合并时会打开 Vim 让写合并说明，
不熟悉 Vim 的人会完全卡住。

**解法**：

    # 方案 A：改用记事本
    git config --global core.editor notepad

    # 方案 B：直接用默认说明，不打开编辑器
    git commit --no-edit

    # 方案 C：如果卡在 Vim 里
    按 Esc，输入 :wq，回车

**教训**：尽量在一处操作（全在本地 或 全在网页）。混用容易分叉。

### 12. Python 三引号字符串吃掉了 JavaScript 的转义符

**现象**：网页版把模型回答里的 Markdown 原样显示出来，
`**粗体**` 和代码块的三反引号都没有被渲染。

**原因**：前端渲染函数写在 Python 的三引号字符串里：

    PAGE = """...javascript...
      body.replace(/```[a-zA-Z]*\\n?([\\s\\S]*?)```/g, ...)
    ..."""
                                        ↑↑
    Python 把 \\n 变成了字面的「反斜杠 + n」两个字符，
    而 JS 正则需要的是真正的换行符，所以永远匹配不到。

**排查过程**：
1. 确认浏览器缓存不是问题（Ctrl+F5 强刷）
2. 确认文件确实更新了（Select-String 搜 renderMarkdown）
3. 逐条检查正则，发现转义层数不对

**解决**：

    1. 用 Python 的原始字符串 raw string：
       PAGE = r"""..."""

    2. 更彻底的做法：干脆不用反斜杠转义，用代码生成特殊字符：
       const NL = String.fromCharCode(10);            // 换行符
       const fence = String.fromCharCode(96, 96, 96); // 三个反引号
       然后用 split() 切分代码块，而不是用正则匹配。

**教训**：跨语言嵌套字符串时，转义符是最容易出错的地方。
能用别的办法绕开反斜杠，就绕开。
