"""
================================================================
  2.4 ReAct Agent — 让 Agent 真正"智能"
================================================================

【学习目标】
  - 掌握 create_react_agent() 的完整用法
  - 理解 ReAct 循环：Reason → Act → Observe
  - 学会自定义工具 + Agent 组合使用
  - 掌握消息格式和结果解析

【前置知识】
  - 01_03_tools_agent.py（Tools 基础）
  - 02_02_state_graph.py（状态管理）
  - 02_03_conditional_routing.py（条件路由）

【核心概念】

  create_react_agent() 是 LangGraph 提供的预构建 Agent，
  它内部已经帮你做好了：

  ┌───────────────────────────────────────┐
  │        create_react_agent() 内部      │
  │                                       │
  │   messages (历史)                      │
  │       ↓                               │
  │   ┌──────────┐                        │
  │   │   LLM    │ ← 思考: 要不要调工具？ │
  │   └────┬─────┘                        │
  │        │ 需要调用                     │
  │        ▼                              │
  │   ┌──────────┐                        │
  │   │  Tools   │ ← 执行: 调用函数       │
  │   └────┬─────┘                        │
  │        │ 返回结果                     │
  │        ▼                              │
  │   回到 LLM ← 观察: 工具返回了什么     │
  │        ↓                              │
  │   不需要更多工具 → 输出最终答案         │
  └───────────────────────────────────────┘

  你只需要提供：
    1. LLM（大脑）
    2. Tools（手脚）—— 可选
    3. Prompt（角色设定） —— 可选

  其余的全部由框架自动处理！
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ============================================================
# Part 1: 定义工具集
# ============================================================

from langchain_core.tools import tool


@tool
def get_patent_type_info(patent_type: str) -> str:
    """
    获取指定专利类型的详细信息。

    Args:
        patent_type: 专利类型，可选值：发明、实用新型、外观设计

    Returns:
        该专利类型的详细说明
    """
    db = {
        "发明": {
            "全称": "发明专利",
            "保护对象": "产品、方法或改进的技术方案",
            "审查方式": "形式审查 + 实质审查",
            "周期": "18-24个月",
            "费用": "申请费900元+实审费2500元",
            "保护期": "20年（从申请日起算）",
            "特点": "授权最难但保护力度最强",
        },
        "实用新型": {
            "全称": "实用新型专利",
            "保护对象": "产品的形状、构造的技术方案",
            "审查方式": "仅形式审查",
            "周期": "6-8个月",
            "费用": "申请费500元",
            "保护期": "10年",
            "特点": "授权快、费用低，但不保护方法",
        },
        "外观设计": {
            "全称": "外观设计专利",
            "保护对象": "产品的整体或局部形状/图案/色彩",
            "审查方式": "仅形式审查",
            "周期": "4-6个月",
            "费用": "申请费500元",
            "保护期": "15年",
            "特点": "只保外观不保功能",
        },
    }

    # 模糊匹配
    for key, info in db.items():
        if key in patent_type or patent_type in key or patent_type in info["全称"]:
            lines = [f"【{info['全称']}】"]
            for k, v in info.items():
                if k != "全称":
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

    return (
        f"未找到'{patent_type}'相关的专利类型。\n"
        f"可选类型: 发明 / 实用新型 / 外观设计"
    )


@tool
def estimate_cost(patent_type: str, need_agency: bool = True) -> str:
    """
    估算专利申请的大致费用。

    Args:
        patent_type: 专利类型（发明/实用新型/外观设计）
        need_agency: 是否通过代理机构办理

    Returns:
        费用估算明细
    """
    base = {"发明": 3400, "实用新型": 500, "外观设计": 500}
    fee = base.get(patent_type, 0)

    agency_fee = 2000 if need_agency else 0
    total = fee + agency_fee

    breakdown = [
        f"【{patent_type}专利费用估算】",
        f"  官费: {fee} 元",
    ]
    if need_agency:
        breakdown.append(f"  代理费(预估): {agency_fee} 元")
    breakdown.append(f"  合计约: {total} 元")
    breakdown.append("  注: 以上为估算，实际以国知局公布为准")

    return "\n".join(breakdown)


@tool
def check_timeline(patent_type: str, current_stage: str = "准备阶段") -> str:
    """
    查询专利各阶段的预计时间线。

    Args:
        patent_type: 专利类型
        current_stage: 当前所处阶段（准备阶段/已提交/审查中）

    Returns:
        时间线说明
    """
    timelines = {
        "发明": {
            "准备阶段": "技术交底书撰写(1-2周) → 检索分析(1周)",
            "已提交": "初步审查(1-3个月)",
            "审查中": "实质审查请求后进入实审(12-18个月)",
        },
        "实用新型": {
            "准备阶段": "交底书撰写(1周) → 检索分析(3天)",
            "已提交": "初步审查(3-6个月)",
            "审查中": "通常6个月内完成授权或驳回",
        },
    }
    tl = timelines.get(patent_type, {})
    stage_info = tl.get(current_stage, f"'{current_stage}'暂无详细时间信息")
    return f"[{patent_type}专利] {current_stage}: {stage_info}"


PATENT_TOOLS = [get_patent_type_info, estimate_cost, check_timeline]


# ============================================================
# 示例代码
# ============================================================

def demo_basic_react_agent():
    """
    【示例 2-4a】最基础的 ReAct Agent

    只需 3 行核心代码就能创建一个能调用工具的 Agent！
    """
    print("\n" + "=" * 60)
    print("  示例 2-4a: 基本 ReAct Agent")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
    )

    agent = create_react_agent(
        model=llm,
        tools=PATENT_TOOLS,
        prompt=(
            "你是一个专业的中国专利咨询助手。\n"
            "你可以查询各类专利的信息、费用和时间线。\n"
            "回答要简洁准确。如果不确定，请先调用工具获取信息再回答。\n"
        ),
    )

    questions = [
        "发明专利的保护期限是多少？",
        "申请一个实用新型大概要花多少钱？",
        "我已经提交了发明专利申请，接下来要等多久？",
    ]

    for q in questions:
        print(f"\n{'─' * 45}")
        print(f"[User] {q}")
        print(f"[AI] ", end="", flush=True)

        result = agent.invoke({"messages": [{"role": "user", "content": q}]})

        last_msg = result["messages"][-1]
        print(last_msg.content)


def demo_message_history():
    """
    【示例 2-4b】查看 Agent 完整的消息历史

    ReAct Agent 的执行过程被完整记录在消息列表中，
    每一步思考、每个工具调用都有迹可循。
    """
    print("\n" + "=" * 60)
    print("  示例 2-4b: Agent 消息历史分析")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    agent = create_react_agent(model=llm, tools=PATENT_TOOLS)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "发明和实用新型的费用各是多少？"}]
    })

    print(f"\n  完整消息记录 (共 {len(result['messages'])} 条):\n")

    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content_preview = msg.content[:80].replace("\n", " ")

        # 显示不同的消息类型标签
        tag = {
            "HumanMessage": "[用户]",
            "AIMessage": "[AI]",
            "ToolMessage": "[工具]",
            "SystemMessage": "[系统]",
        }.get(msg_type, f"[{msg_type}]")

        extra = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            extra = f" | 调用工具: {[tc['name'] for tc in msg.tool_calls]}"
        if hasattr(msg, "name") and msg.name:
            extra = f" | 工具名: {msg.name}"

        print(f"  [{i:02d}] {tag}{extra}")
        print(f"       {content_preview}")
        print()


def demo_multi_turn_conversation():
    """
    【示例 2-4c】多轮对话 —— 保持上下文连续性

    Agent 能记住之前的对话内容，实现真正的多轮交互。
    """
    print("\n" + "=" * 60)
    print("  示例 2-4c: 多轮对话")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    agent = create_react_agent(
        model=llm,
        tools=PATENT_TOOLS,
        prompt="你是一个专业的专利咨询助手。",
    )

    # 多轮对话 —— 把历史消息不断传入
    conversation_history = []

    turns = [
        "我想了解发明专利的基本情况",
        "那申请费用呢？需要代理吗？",
        "实用新型呢？和发明比哪个更划算？",
    ]

    for user_input in turns:
        print(f"\n{'─' * 40}")
        print(f"[User] {user_input}")

        conversation_history.append({"role": "user", "content": user_input})

        result = agent.invoke({"messages": conversation_history})
        last_msg = result["messages"][-1]

        print(f"[AI] {last_msg.content}")

        # 把 AI 的回复也加入历史（保持上下文）
        conversation_history.append({
            "role": "assistant",
            "content": last_msg.content,
        })


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.4 ReAct Agent 教程           ║")
    print("╚════════════════════════════════════════════╝")
    print(f"\n使用模型: {OLLAMA_MODEL}")

    try:
        demo_basic_react_agent()
        demo_message_history()
        demo_multi_turn_conversation()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
