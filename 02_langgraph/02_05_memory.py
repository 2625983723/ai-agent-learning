"""
================================================================
  2.5 记忆机制 (Memory) — 让 Agent 有"记忆"
================================================================

【学习目标】
  - 理解 Agent 的两种记忆：短期（对话历史）vs 长期（跨会话）
  - 掌握 MemorySaver：最简单的内存型记忆
  - 学会使用 checkpointer 实现对话持久化
  - 了解 thread_id 的概念和作用

【前置知识】
  - 02_04_react_agent.py（ReAct Agent）

【核心概念】

  没有记忆的 Agent:
    第1轮: 用户"我叫李华" → AI"你好李华！"
    第2轮: 用户"我叫什么？" → AI"抱歉我不知道..."  ← 忘了！

  有记忆的 Agent:
    第1轮: 用户"我叫李华" → AI"你好李华！"
           ┌──────────────────┐
           │ Memory (记忆)     │
           │ name="李华"       │
           └──────────────────┘
    第2轮: 用户"我叫什么？" → AI"你叫李华！"  ← 记住了！

  LangGraph 的记忆体系:

  ┌────────────┬─────────────┬──────────────────────────┐
  │ 类型       │ 组件        │ 说明                     │
  ├────────────┼─────────────┼──────────────────────────┤
  │ 短期记忆   │ MemorySaver │ 存在内存中，程序重启丢失   │
  │ 短期记忆   │ SqliteSaver │ 存在 SQLite 文件，可持久化│
  │ 长期记忆   │ Store API   │ 跨会话记住用户偏好等      │
  └────────────┴─────────────┴──────────────────────────┘

  关键概念：thread_id（线程 ID）
    - 每个用户/每个会话有一个唯一的 thread_id
    - 同一个 thread_id 的消息会被归为一组
    - 不同 thread_id 之间完全隔离（就像不同的聊天窗口）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def demo_memorysaver_basic():
    """
    【示例 2-5a】MemorySaver 基础用法

    只需两步就能让 Agent 拥有记忆：
    1. 创建一个 checkpointer（MemorySaver）
    2. 编译图时传入它

    之后每次 invoke 时传入 config={"configurable": {"thread_id": "xxx"}}
    就能自动保存和恢复对话历史。
    """
    print("\n" + "=" * 60)
    print("  示例 2-5a: MemorySaver 基础")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    # ---- Step 1: 创建 LLM 和 Agent ----
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    agent = create_react_agent(
        model=llm,
        tools=[],          # 本示例不需要工具，专注演示记忆
        prompt=(
            "你是一个有记忆的助手。"
            "请记住用户告诉你的信息，在后续对话中可以引用。\n"
            "回答要简短。"
        ),
    )

    # ---- Step 2: 创建 Checkpointer ----
    # MemorySaver 把对话历史保存在内存中（Python 字典）
    memory = MemorySaver()

    # ---- Step 3: 带 Checkpointer 编译 Agent ----
    app = agent.compile(checkpointer=memory)

    # ---- Step 4: 使用 thread_id 进行多轮对话 ----
    thread = {"configurable": {"thread_id": "user-001"}}

    print(f"\n[第1轮]")
    r1 = app.invoke(
        {"messages": [{"role": "user", "content": "我叫李华，我是一名RPA开发工程师"}]},
        config=thread,
    )
    msg1 = r1["messages"][-1]
    print(f"[User] 我叫李华，我是一名RPA开发工程师")
    print(f"[AI] {msg1.content}")

    print(f"\n[第2轮] （Agent 应该记得第一轮的信息）")
    r2 = app.invoke(
        {"messages": [{"role": "user", "content": "我的职业是什么？"}]},
        config=thread,              # 同一个 thread_id！
    )
    msg2 = r2["messages"][-1]
    print(f"[User] 我的职业是什么？")
    print(f"[AI] {msg2.content}")

    # 验证：查看保存的消息数量
    total_msgs = len(r2["messages"])
    print(f"\n  [记忆状态] 当前线程共 {total_msgs} 条消息")


def demo_multi_thread_isolation():
    """
    【示例 2-5b】多线程隔离 —— 不同用户互不干扰

    thread_id 就像聊天窗口的标签：
    - thread_id="alice" 是 Alice 的窗口
    - thread_id="bob" 是 Bob 的窗口
    两个窗口的对话完全独立，互不影响。

    这就是为什么同一个 Agent 服务可以同时服务多个用户。
    """
    print("\n" + "=" * 60)
    print("  示例 2-5b: 多线程隔离")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    agent = create_react_agent(model=llm, prompt="简短回答。")

    memory = MemorySaver()
    app = agent.compile(checkpointer=memory)

    # Alice 的对话
    alice_thread = {"configurable": {"thread_id": "alice"}}
    app.invoke(
        {"messages": [{"role": "user", "content": "我最喜欢的颜色是蓝色"}]},
        config=alice_thread,
    )

    # Bob 的对话 —— 他喜欢红色
    bob_thread = {"configurable": {"thread_id": "bob"}}
    app.invoke(
        {"messages": [{"role": "user", "content": "我最喜欢的颜色是红色"}]},
        config=bob_thread,
    )

    # 分别问两人最喜欢的颜色
    print("\n[Alice 被问到]")
    r_alice = app.invoke(
        {"messages": [{"role": "user", "content": "我最喜欢什么颜色？"}]},
        config=alice_thread,
    )
    print(f"  [Alice] {r_alice['messages'][-1].content}")

    print("\n[Bob 被问到]")
    r_bob = app.invoke(
        {"messages": [{"role": "user", "content": "我最喜欢什么颜色？"}]},
        config=bob_thread,
    )
    print(f"  [Bob] {r_bob['messages'][-1].content}")

    print("\n  ✅ 两个用户的对话互不干扰")


def demo_state_history():
    """
    【示例 2-5c】查看历史状态快照

    MemorySaver 不仅保存最终结果，
    还保存了每一步的中间状态——这叫"时间旅行"功能。

    你可以回看 Agent 在任何时刻的完整状态。
    """
    print("\n" + "=" * 60)
    print("  示例 2-5c: 历史状态快照")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.tools import tool

    @tool
def add(a: int, b: int) -> int:
    """加法计算器"""
        return a + b

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    agent = create_react_agent(model=llm, tools=[add])
    memory = MemorySaver()
    app = agent.compile(checkpointer=memory)

    thread = {"configurable": {"thread_id": "demo-history"}}

    # 执行一次对话
    result = app.invoke(
        {"messages": [{"role": "user", "content": "请帮我算 25+17"}]},
        config=thread,
    )

    # 查看所有历史 checkpoint
    checkpoints = list(memory.list(config=thread["configurable"]))
    print(f"\n  共有 {len(checkpoints)} 个检查点（每步执行后自动保存）:")

    for i, ckpt in enumerate(checkpoints):
        ts = ckpt.metadata.get("step", "?")
        print(f"    检查点 {i}: step={ts}, id={ckpt.id[:12]}...")

    # 查看最新状态的完整消息记录
    latest = result["messages"]
    print(f"\n  最新状态中的消息:")
    for j, msg in enumerate(latest):
        role = type(msg).__name__.replace("Message", "")
        content = str(msg.content)[:60]
        print(f"    [{j}] {role}: {content}")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.5 记忆机制教程               ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_memorysaver_basic()
        demo_multi_thread_isolation()
        demo_state_history()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
