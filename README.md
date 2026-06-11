# AI Agent 学习工程

> **面向新手的 LangChain + LangGraph + MCP 实战学习工程**
>
> 从零到一，手把手教你开发自己的 AI Agent。

---

## 目录结构概览

```
ai-agent-learning/
├── README.md                          ← 你在这里
├── requirements.txt                   ← 依赖安装清单
├── .env.example                       ← 环境变量模板
├── config.py                          ← 全局配置
│
├── 01_basics/                         ← 【第1章】LangChain v1.x 基础
│   ├── 01_01_model_io.py              │   1.1 模型调用（Ollama 本地模型）
│   ├── 01_02_lcel_chains.py           │   1.2 LCEL 链式调用（管道符）
│   ├── 01_03_tools_agent.py           │   1.3 工具定义与 Agent
│   ├── 01_04_rag.py                   │   1.4 RAG 检索增强生成
│   └── test_01_basics.py              │   测试用例
│
├── 02_langgraph/                      ← 【第2章】LangGraph 图式编程
│   ├── 02_01_first_graph.py           │   2.1 你的第一个图
│   ├── 02_02_state_graph.py           │   2.2 状态图与节点
│   ├── 02_03_conditional_routing.py   │   2.3 条件路由：根据状态决定走向
│   ├── 02_04_react_agent.py           │   2.4 ReAct Agent：思考-行动循环
│   ├── 02_05_memory.py                │   2.5 记忆机制
│   ├── 02_06_human_in_loop.py         │   2.6 人机协作（Human-in-the-Loop）
│   └── test_02_langgraph.py           │   测试用例
│
├── 03_mcp/                            ← 【第3章】MCP 协议实战
│   ├── mcp_server_demo.py             │   FastMCP Server 示例
│   ├── mcp_client_demo.py             │   FastMCP Client 示例
│   ├── mcp_langgraph_integration.py   │   MCP + LangGraph 集成
│   └── test_03_mcp.py                 │   测试用例
│
└── 04_project/                        ← 【第4章】综合实战项目
    └── patent_assistant/              │   专利流程智能助手（雏形）
        ├── agent.py                   │     主 Agent 逻辑
        ├── tools.py                   │     自定义工具
        ├── server.py                  │     FastAPI 部署服务
        └── test_patent.py             │     测试用例
```

## 快速开始

### 第 0 步：环境准备（约 10 分钟）

```bash
# 1. 安装 Ollama（如果还没有）
#    Windows: 从 https://ollama.com 下载安装包
#    Mac:     brew install ollama
#    Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取本地模型（中文能力强的免费模型）
ollama pull qwen3:8b

# 3. 克隆 / 进入本项目目录
cd ai-agent-learning

# 4. 创建 Python 虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 5. 安装依赖（使用清华镜像加速）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt

# 6. 复制环境变量文件（可选，默认值已够用）
cp .env.example .env
```

### 第 0.5 步：验证环境

```bash
# 确认 Ollama 在运行
ollama list

# 运行配置检查
python config.py
```

输出应该类似：
```
========================================
  AI Agent 学习工程 — 当前配置
========================================
  LLM 提供方 : ollama
  模型名称   : qwen3:8b
  API 地址   : http://localhost:11434
  数据目录   : ./data/chroma
========================================
```

### 第 1 步：运行第一个示例

```bash
# 进入基础篇目录
cd 01_basics

# 运行第一个示例 —— 调用本地大模型
python 01_01_model_io.py
```

你应该能看到 Ollama 的回复。

---

## 学习路线图

| 章节 | 内容 | 核心概念 | 前置知识 | 时间 |
|:----:|------|---------|:--------:|:----:|
| **第1章** | LangChain v1.x 基础 | Model I/O、LCEL、Tools、RAG | Python 基础 | ~5 天 |
| **第2章** | LangGraph 图式编程 | StateGraph、Nodes、Edges、Agent | 第1章 | ~7 天 |
| **第3章** | MCP 协议实战 | FastMCP Server/Client | Python 异步 | ~4 天 |
| **第4章** | 综合实战项目 | 整合全部知识点 | 前3章 | ~5 天 |

> **建议学习方式**：每个 `.py` 文件都先自己通读代码和注释 → 手敲一遍运行 → 再看测试用例理解边界情况。

---

## 每章知识点速查

### 第 1 章：LangChain v1.x 基础

| 编号 | 文件 | 学到什么 |
|:----:|------|---------|
| 1.1 | `01_01_model_io.py` | 如何调用 LLM、构造消息、流式输出 |
| 1.2 | `01_02_lcel_chains.py` | LCEL 管道符 `\|`、链式组合、并行执行 |
| 1.3 | `01_03_tools_agent.py` | `@tool` 装饰器、让 LLM 自动调用函数 |
| 1.4 | `01_04_rag.py` | Embedding、向量存储、文档检索问答 |

### 第 2 章：LangGraph 图式编程

| 编号 | 文件 | 学到什么 |
|:----:|------|---------|
| 2.1 | `02_01_first_graph.py` | 最简单的图：Node + Edge + Compile |
| 2.2 | `02_02_state_graph.py` | TypedDict 定义状态、状态在节点间传递 |
| 2.3 | `02_03_conditional_routing.py` | 条件边、根据状态动态路由到不同节点 |
| 2.4 | `02_04_react_agent.py` | `create_react_agent()` 一键创建 Agent |
| 2.5 | `02_05_memory.py` | MemorySaver 短期记忆、跨轮对话持久化 |
| 2.6 | `02_06_human_in_loop.py` | `interrupt()` 让 Agent 暂停等人工确认 |

### 第 3 章：MCP 协议实战

| 编号 | 文件 | 学到什么 |
|:----:|------|---------|
| 3.1 | `mcp_server_demo.py` | 用 FastMCP 写一个 MCP Server |
| 3.2 | `mcp_client_demo.py` | Client 连接 Server 并调用工具 |
| 3.3 | `mcp_langgraph_integration.py` | 把 MCP 工具接入 LangGraph Agent |

### 第 4 章：综合实战

| 文件 | 学到什么 |
|------|---------|
| `patent_assistant/agent.py` | 把前 3 章的知识整合成一个完整 Agent |

---

## 运行所有测试

```bash
# 在项目根目录下
python -m pytest -v
```

或者按章节分别跑：

```bash
# 只测 LangChain 基础
pytest 01_basics/test_01_basics.py -v

# 只测 LangGraph
pytest 02_langgraph/test_02_langgraph.py -v
```

---

## 常见问题

### Q: Ollama 启动失败？
A: 确保 Ollama 服务正在运行。Windows 下检查系统托盘有没有 Ollama 图标；Mac/Linux 终端输入 `ollama serve`。

### Q: 模型下载太慢？
A: 可以换成更小的模型：`ollama pull qwen3:4b`，然后在 `.env` 里改 `OLLAMA_MODEL=qwen3:4b`。

### Q: 中文乱码？
A: 确保终端编码是 UTF-8。Windows PowerShell 可能需要 `chcp 65001`。

### Q: 想用云端 API 而不是本地模型？
A: 编辑 `.env` 文件，取消注释 `OPENAI_API_KEY` 和相关配置即可。

---

## 技术栈版本

| 组件 | 版本 |
|------|------|
| Python | >= 3.11 |
| langchain | >= 0.3.0 (v1.x) |
| langgraph | >= 0.2.0 (v1.x) |
| fastmcp | >= 3.0.0 |
| chromadb | >= 0.5.0 |

---

*祝学习顺利！遇到问题随时看代码里的注释，每个关键步骤都有详细说明。*
