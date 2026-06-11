"""
================================================================
  2.2 状态图 (State Graph) — 深入理解状态管理
================================================================

【学习目标】
  - 理解 State（状态）在 LangGraph 中的核心地位
  - 掌握 TypedDict 定义复杂状态
  - 学会 Annotated + Reducer 处理状态冲突
  - 理解状态如何在节点之间自动合并

【前置知识】
  - 02_01_first_graph.py（基本图结构）

【核心概念】

  状态 (State) = 图中所有节点共享的"全局变量"

  ┌──────────────────────────────┐
  │         State (状态)          │
  │                              │
  │   messages: [...]            │
  │   user_query: "..."          │
  │   search_results: [...]      │
  │   answer: "..."              │
  │                              │
  │  所有节点都能读和写           │
  └──────────────────────────────┘
       ↑ 写        ↑ 读        ↑ 写
    [节点A]     [节点B]      [节点C]

  关键规则：
    - 节点函数返回的字典会自动**合并**到状态中
    - 同一个 key 如果被多次写入，后写的覆盖前写的（默认行为）
    - 可以用 Annotated + Reducer 自定义合并策略（如 append）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_typed_dict_state():
    """
    【示例 2-2a】用 TypedDict 定义丰富的状态类型

    TypedDict 让你的状态有清晰的类型提示，
    IDE 能自动补全，代码不容易写错。
    """
    print("\n" + "=" * 60)
    print("  示例 2-2a: TypedDict 状态定义")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, List, Optional

    # 定义一个包含多种类型的复杂状态
    class AgentState(TypedDict):
        # 字符串类型
        query: str                  # 用户查询
        intent: str                 # 意图识别结果

        # 列表类型
        messages: list              # 对话消息列表
        search_results: List[str]   # 搜索结果

        # 可选类型
        answer: Optional[str]       # 最终答案（可能为空）
        confidence: float           # 置信度 (0.0 ~ 1.0)

    # --- 节点函数 ---

    def classify_intent(state: AgentState) -> dict:
        """节点1：识别用户意图"""
        q = state["query"].lower()
        if "专利" in q or "申请" in q:
            intent = "patent_query"
        elif "天气" in q or "温度" in q:
            intent = "weather_query"
        else:
            intent = "general_chat"

        print(f"  [意图分类] '{state['query']}' → {intent}")
        return {"intent": intent}

    def generate_response(state: AgentState) -> dict:
        """节点2：根据意图生成回复"""
        intent = state["intent"]
        responses = {
            "patent_query": "好的，我来帮您查询专利相关信息。",
            "weather_query": "抱歉，我暂时无法查询天气。",
            "general_chat": "请问有什么可以帮您的？",
        }
        answer = responses.get(intent, "未知意图")
        print(f"  [生成回复] 意图={intent} → {answer}")
        return {"answer": answer, "confidence": 0.85}

    # --- 构建并运行 ---
    g = StateGraph(AgentState)
    g.add_node("classify", classify_intent)
    g.add_node("respond", generate_response)
    g.add_edge(START, "classify")
    g.add_edge("classify", "respond")
    g.add_edge(respond, END)

    app = g.compile()

    # 测试不同输入
    test_queries = [
        "我想了解发明专利的申请流程",
        "北京今天天气怎么样？",
        "你好，介绍一下你自己",
    ]

    for q in test_queries:
        print(f"\n{'─' * 40}")
        result = app.invoke({
            "query": q,
            "intent": "",
            "messages": [],
            "search_results": [],
            "answer": "",
            "confidence": 0.0,
        })
        print(f"\n  结果: intent={result['intent']}, answer={result['answer']}")


def demo_state_merge_behavior():
    """
    【示例 2-2b】理解状态的自动合并机制

    这是新手最容易困惑的地方：
    当多个节点都返回同一个 key 时，会发生什么？
    """
    print("\n" + "=" * 60)
    print("  示例 2-2b: 状态合并机制")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Annotated
    import operator

    # --- 场景1: 默认行为（后覆盖前）---
    class DefaultState(TypedDict):
        value: str
        log: list[str]

    def write_foo(state: DefaultState) -> dict:
        return {"value": "foo"}

    def write_bar(state: DefaultState) -> dict:
        return {"value": "bar"}       # 会覆盖 foo！

    g1 = StateGraph(DefaultState)
    g1.add_node("write_foo", write_foo)
    g1.add_node("write_bar", write_bar)
    g1.add_edge(START, "write_foo")
    g1.add_edge("write_foo", "write_bar")
    g1.add_edge(write_bar, END)

    r1 = g1.compile().invoke({"value": "", "log": []})
    print(f"\n  默认合并: 'foo' 然后 'bar' → 最终值: '{r1['value']}'")
    assert r1["value"] == "bar", "后写入的应该覆盖先写入的"

    # --- 场景2: 使用 Reducer 追加而不是覆盖 ---
    #
    # Annotated[type, reducer] 告诉 LangGraph：
    # "这个字段不要覆盖，要用 reducer 函数合并"
    # operator.add 对于列表就是追加操作

    class AppendState(TypedDict):
        messages: Annotated[list, operator.add]
        current: str

    def step1(state: AppendState) -> dict:
        return {
            "messages": ["[Step1] 开始处理"],
            "current": "step1_done",
        }

    def step2(state: AppendState) -> dict:
        return {
            "messages": ["[Step2] 正在工作中..."],
            "current": "step2_done",
        }

    def step3(state: AppendState) -> dict:
        return {
            "messages": ["[Step3] 全部完成！"],
            "current": "all_done",
        }

    g2 = StateGraph(AppendState)
    for name, fn in [("s1", step1), ("s2", step2), ("s3", step3)]:
        g2.add_node(name, fn)
    g2.add_edge(START, "s1")
    g2.add_edge("s1", "s2")
    g2.add_edge("s2", "s3")
    g2.add_edge("s3", END)

    r2 = g2.compile().invoke({"messages": [], "current": ""})
    print(f"\n  追加合并 (operator.add):")
    for msg in r2["messages"]:
        print(f"    {msg}")
    assert len(r2["messages"]) == 3, "三个步骤的消息都被保留了"


def demo_message_list_pattern():
    """
    【示例 2-2c】消息列表模式 —— 最常用的状态设计

    在 AI Agent 开发中，最常见的状态设计是维护一个消息列表。
    每个 node 往列表里追加一条消息。

  ┌─────────────────────────────────────────┐
  │  State.messages                         │
  │                                         │
  │  [0] SystemMessage("你是助手")          │ ← 初始状态
  │  [1] HumanMessage("帮我查专利")          │ ← 用户输入
  │  [2] AIMessage("好的，我来查...")       │ ← 节点A追加
  │  [3] ToolMessage("发明专利信息...")     │ ← 工具调用结果
  │  [4] AIMessage("根据检索结果...")       │ ← 节点B追加
  │                                         │
    └─────────────────────────────────────────┘
    """
    print("\n" + "=" * 60)
    print("  示例 2-2c: 消息列表模式")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Annotated
    import operator
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    def chatbot_node(state: ChatState) -> dict:
        """聊天机器人节点：模拟回复"""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            reply_text = f"[Bot] 收到您的消息：{last_msg.content[:20]}..."
            return {"messages": [AIMessage(content=reply_text)]}
        return {}

    def summary_node(state: ChatState) -> dict:
        """总结节点"""
        count = len(state["messages"])
        return {
            "messages": [AIMessage(content=f"[Summary] 本轮对话共 {count} 条消息")]
        }

    g = StateGraph(ChatState)
    g.add_node("chatbot", chatbot_node)
    g.add_node("summary", summary_node)
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", "summary")
    g.add_edge(summary, END)

    app = g.compile()

    initial_messages = [
        SystemMessage(content="你是一个有用的助手。"),
        HumanMessage(content="你好！"),
    ]
    result = app.invoke({"messages": initial_messages})

    print("\n  完整消息记录:")
    for i, msg in enumerate(result["messages"]):
        role = type(msg).__name__.replace("Message", "")
        content = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        print(f"    [{i}] {role}: {content}")

    assert len(result["messages"]) == len(initial_messages) + 2


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.2 状态管理教程               ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_typed_dict_state()
        demo_state_merge_behavior()
        demo_message_list_pattern()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
