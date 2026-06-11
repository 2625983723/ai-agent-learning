"""
================================================================
  3.3 MCP + LangGraph 集成 — 打通全部能力
================================================================

【学习目标】
  - 学会把 MCP Server 的工具接入 LangGraph Agent
  - 掌握 langchain-mcp-adapters 的用法
  - 理解生产环境中 MCP + Agent 的架构

【前置知识】
  - 03_mcp_server_demo.py（MCP Server 开发）
  - 02_04_react_agent.py（ReAct Agent）

【核心概念图解】

  完整架构:

  ┌────────────────────────────────────┐
  │            用户                  │
  └──────────────┬───────────────┘
                 │ 提问
                 ▼
  ┌────────────────────────────────────┐
  │     LangGraph ReAct Agent        │
  │                                │
  │  ┌──────────┐                │
  │  │   LLM    │ ← 思考            │
  │  └────┬─────┘                │
  │       │ 需要工具               │
  │       ▼                       │
  │  ┌─────────────────────────┐  │
  │  │  MCP Tool Adapters       │  │
  │  │  (把 MCP 工具转成        │  │
  │  │   LangChain 工具）       │  │
  │  └──────┬──────────────────┘  │
  └───────────┼────────────────────┘
                  │ 调用
                  ▼
  ┌────────────────────────────────────┐
  │          MCP Server             │
  │  (对外提供工具/资源）         │
  └────────────────────────────────────┘

  关键组件: langchain_mcp_adapters
    - 把 MCP Server 的工具列表"翻译"成 LangChain Tool 格式
    - LangGraph Agent 就能直接调用 MCP 工具了！
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_adapter_concept():
    """
    【示例 3-3a】MCP 工具适配器概念讲解

    langchain-mcp-adapters 做的事情很简单：
    把 MCP Server 暴露的 tool 转换成 LangChain 能用的 Tool 对象。
    """
    print("\n" + "=" * 60)
    print("  示例 3-3a: MCP 工具适配原理")
    print("=" * 60)

    concept_code = '''
"""MCP 工具适配原理"""

# ─── 没有适配器时 ──────────────────
# LangGraph Agent 只能调用 LangChain Tool
# 你的 MCP Server 的工具 = 不可见！

# ─── 使用适配器后 ─────────────────
from langchain_mcp_adapters import load_mcp_tools

# 1. 连接到 MCP Server
async with Client("mcp_patent_server.py") as client:
    # 2. 把 MCP 工具"翻译"成 LangChain 工具
    mcp_tools = await load_mcp_tools(client)

    # 3. mcp_tools 现在是一个 LangChain Tool 列表
    #    可以直接传给 create_react_agent()！
    agent = create_react_agent(llm, tools=mcp_tools)
'''

    print("\n  核心代码:\n")
    for ln in concept_code.strip().splitlines():
        print(f"  {ln}")


def demo_full_integration_code():
    """
    【示例 3-3b】完整集成代码 —— 可直接运行

    这是生产环境中最常用的模式：
    LangGraph Agent + MCP Server 工具
    """
    print("\n" + "=" * 60)
    print("  示例 3-3b: 完整集成代码")
    print("=" * 60)

    full_code = r'''"""
MCP + LangGraph 完整集成示例
==================================
Architecture:
  LangGraph Agent → MCP Tools → Patent MCP Server
"""

import asyncio
from pathlib import Path
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from fastmcp import Client
from langchain_mcp_adapters import load_mcp_tools

# ─── 配置 ─────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen3:8b"

# MCP Server 文件路径（与当前脚本同目录）
SERVER_PATH = str(Path(__file__).parent / "mcp_patent_server.py")


# ─── 主异步函数 ─────────────────────────
async def main():
    print("🔗 正在连接 MCP Server...")

    # 1. 创建 MCP Client 并连接
    async with Client(SERVER_PATH) as client:
        print(f"  ✅ 已连接: {client.name}")

        # 2. 加载 MCP 工具，并转换成 LangChain 格式
        mcp_tools = await load_mcp_tools(client)
        print(f"  📋 加载了 {len(mcp_tools)} 个 MCP 工具:")
        for t in mcp_tools:
            print(f"     - {t.name}: {t.description[:40]}...")

        # 3. 创建 LLM
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=MODEL,
            temperature=0.3,
        )

        # 4. 创建 ReAct Agent（把 MCP 工具加进去！）
        agent = create_react_agent(
            model=llm,
            tools=mcp_tools,   # ← 这里传入 MCP 工具！
            prompt=(
                "你是一个专利咨询助手。"
                "请使用可用工具查询准确信息后回答。"
            ),
        )

        # 5. 运行 Agent
        print(f"\n{'─'*45}")
        print("[用户] 发明专利的保护期限是多久？费用大概多少？")
        print(f"{'─'*45}")

        result = await agent.ainvoke(
            {"messages": [
                {"role": "user",
                 "content": "发明专利的保护期限是多久？费用大概多少？"}
            ]}
        )

        # 6. 显示结果
        last_msg = result["messages"][-1]
        print(f"\n[AI] {last_msg.content}")


# ─── 入口 ───────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
'''

    # 打印代码（带行号）
    print("\n  完整代码:\n")
    for i, ln in enumerate(full_code.strip().splitlines(), 1):
        # 跳过空行，保持可读性
        if ln.strip() == "":
            print()
        else:
            print(f"  {ln}")


def demo_architecture_diagram():
    """
    【示例 3-3c】架构图 —— 理解数据流向
    """
    print("\n" + "=" * 60)
    print("  示例 3-3c: 完整架构数据流")
    print("=" * 60)

    print(r"""
  ┌─────────────────────────────────────────────────┐
  │  ① 用户提问                              │
  │     "发明专利期限多长？费用多少？"          │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  ② LangGraph ReAct Agent                     │
  │     ┌───────────┐                           │
  │     │  LLM 思考  │  "需要调工具获取数据"    │
  │     └─────┬─────┘                           │
  │           │ 调用 query_patent_type("发明")    │
  │           ▼                                    │
  │     ┌──────────────────────────┐              │
  │     │  langchain-mcp-adapters │              │
  │     │  工具格式转换层            │              │
  │     └────────┬─────────────────┘              │
  └──────────────┼─────────────────────────────────┘
                     │ MCP 协议 (stdio)
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  ③ MCP Server (mcp_patent_server.py)       │
  │     ┌──────────────────┐                      │
  │     │ query_patent_type│ ← 执行工具函数      │
  │     └────────┬─────────┘                      │
  │          返回结果                              │
  └──────────────┼─────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  ④ Agent 收到工具返回结果                   │
  │     LLM 思考: "信息已足够，可以回答了"       │
  │     输出: "发明保护20年，费用约3400元..."    │
  └─────────────────────────────────────────────────┘
    """)


def demo_multi_server_integration():
    """
    【示例 3-3d】多 MCP Server 集成

    生产环境中，你可能有多个 MCP Server：
      - 专利信息 Server
      - 日历 Server
      - 邮件 Server
      - ...

    如何让一个 Agent 同时用多个 Server 的工具？
    """
    print("\n" + "=" * 60)
    print("  示例 3-3d: 多 MCP Server 集成")
    print("=" * 60)

    multi_code = r'''"""
多 MCP Server 集成 — 一个 Agent 用多个 Server 的工具
"""

import asyncio
from fastmcp import Client
from langchain_mcp_adapters import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ─── 定义多个 MCP Server ─────────────────
SERVERS = {
    "patent": "mcp_patent_server.py",
    "calendar": "mcp_calendar_server.py",
    "email": "mcp_email_server.py",
}


async def main():
    # 存放所有工具的列表
    all_tools = []

    # 逐个连接 Server，加载工具
    for name, path in SERVERS.items():
        print(f"🔗 连接 {name} Server...")
        async with Client(path) as client:
            tools = await load_mcp_tools(client)
            all_tools.extend(tools)
            print(f"  ✅ 加载 {len(tools)} 个工具")

    print(f"\n📋 共加载 {len(all_tools)} 个工具")

    # 创建 Agent（传入所有工具）
    llm = ChatOllama(base_url="...", model="...")
    agent = create_react_agent(llm, tools=all_tools)

    # 运行
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "..."}]}
    )
    print(result["messages"][-1].content)


asyncio.run(main())
'''

    print("\n  多 Server 集成代码:\n")
    for ln in multi_code.strip().splitlines():
        print(f"  {ln}")

    print("\n  💡 提示:")
    print("    每个 Server 运行在独立进程中（stdio 模式），")
    print("    Agent 通过适配器统一调用，对 LLM 来说是无感的。")


def demo_save_runnable_file():
    """
    把可运行的集成代码保存为文件
    """
    print("\n" + "=" * 60)
    print("  保存可运行文件")
    print("=" * 60)

    code = r'''"""
MCP + LangGraph 集成 — 可运行示例
==========================================
运行前准备:
  1. 确保 Ollama 运行中: ollama list
  2. 确保 mcp_patent_server.py 在同一目录
  3. 安装依赖:
     pip install langchain-mcp-adapters
"""

import asyncio
from pathlib import Path
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from fastmcp import Client
from langchain_mcp_adapters import load_mcp_tools

# ─── 配置 ───────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"
SERVER_FILE = str(Path(__file__).parent / "mcp_patent_server.py")


async def main():
    print("🔗 连接 MCP Server...")

    async with Client(SERVER_FILE) as client:
        print(f"  ✅ 已连接: {client.name}")

        # 加载 MCP 工具
        tools = await load_mcp_tools(client)
        print(f"  📋 加载工具数: {len(tools)}")
        for t in tools:
            print(f"     - {t.name}")

        # 创建 LLM + Agent
        llm = ChatOllama(base_url=OLLAMA_URL, model=MODEL_NAME)
        agent = create_react_agent(llm, tools=tools, prompt="你是专利助手。")

        # 测试问题
        questions = [
            "发明专利保护期限多久？",
            "申请实用新型要多少钱？",
        ]

        for q in questions:
            print(f"\n{'─'*40}")
            print(f"[问] {q}")
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": q}]}
            )
            print(f"[答] {result['messages'][-1].content}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 请确保:")
        print("  1. Ollama 正在运行")
        print(f"  2. {SERVER_FILE} 文件存在")
        print("  3. 已安装: pip install langchain-mcp-adapters")
'''

    output_path = Path(__file__).parent / "mcp_langgraph_integration.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\n  ✅ 已保存: {output_path}")
    print(f"\n  🚀 运行方式:")
    print(f"     1. 先启动 MCP Server（另一个终端）:")
    print(f"        python mcp_patent_server.py")
    print(f"\n     2. 运行集成脚本:")
    print(f"        python mcp_langgraph_integration.py")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  MCP 协议 — 3.3 MCP+LangGraph 集成        ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_adapter_concept()
        demo_full_integration_code()
        demo_architecture_diagram()
        demo_multi_server_integration()
        demo_save_runnable_file()

        print("\n✅ 全部示例完成！")
        print("\n📌 下一步建议:")
        print("  1. 先运行 mcp_patent_server.py 启动 Server")
        print("  2. 再运行 mcp_langgraph_integration.py 测试集成")
        print("  3. 尝试添加更多 MCP Server（多 Server 模式）")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
