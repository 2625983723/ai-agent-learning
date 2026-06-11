"""
================================================================
  1.3 工具 (Tools) 与 Agent — 让 LLM 能"动手做事"
================================================================

【学习目标】
  - 理解什么是 Tool（工具）：给 LLM 装上"手"
  - 学会用 @tool 装饰器定义自己的工具
  - 掌握 Agent 的概念：LLM + Tools = Agent
  - 用 create_react_agent() 创建一个能调用工具的 Agent

【前置知识】
  - 01_01_model_io.py（模型调用）
  - 01_02_lcel_chains.py（LCEL 链式调用）

【核心概念图解】

  没有 Tools 的 LLM:
    用户: "北京今天天气怎么样？"
    LLM : "抱歉，我无法查询实时天气信息..."  ← 只能靠训练数据回答

  有 Tools 的 Agent:
    用户: "北京今天天气怎么样？"
     ┌──────────────┐
     │    Agent      │ ← LLM 的脑子 + 工具的手
     │              │
     │ 思考(Reason)  │ "用户想知道天气，我需要调用 get_weather 工具"
     │ 行动(Act)     │ → get_weather("北京")  ← 调用工具！
     │ 观察(Observe) │ 返回: "晴天, 25°C"
     │ 思考(Reason)  │ "根据结果，我可以回答用户了..."
     │ 回答(Response)│ "北京今天晴天，温度 25 度左右。"
     └──────────────┘

  这就是 ReAct 模式: Reason → Act → Observe → 循环
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ============================================================
# Part 1: 定义自定义工具
# ============================================================

# @tool 是 LangChain 提供的装饰器，可以把普通 Python 函数变成
# AI 可以调用的"工具"。就像给函数贴了一个标签，告诉 LLM：
# "这个函数可以调用，功能是 xxx"

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。

    Args:
        expression: 数学表达式，如 "2+3*4", "(10-3)/2"
                   支持加减乘除和括号

    Returns:
        计算结果的字符串形式
    """
    try:
        # 安全起见：只允许数字和基本运算符
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"错误: 表达式包含不允许的字符 '{expression}'"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """
    获取当前的日期和时间。
    不需要任何参数。

    Returns:
        当前时间的字符串表示
    """
    from datetime import datetime
    now = datetime.now()
    return now.strftime("当前时间: %Y年%m月%d日 %H:%M:%S")


@tool
def search_patent_info(keyword: str) -> str:
    """
    搜索专利相关信息。

    Args:
        keyword: 专利搜索关键词，如 "发明专利", "实用新型"

    Returns:
        专利相关信息的字符串
    """
    # 这里是模拟数据——实际项目中会连接真实数据库或 API
    mock_db = {
        "发明专利": {
            "定义": "对产品、方法或者其改进所提出的新的技术方案",
            "保护期限": "20 年",
            "审查周期": "18-24 个月",
            "费用": "申请费 900 元 + 实审费 2500 元",
        },
        "实用新型": {
            "定义": "针对产品的形状、构造提出的实用新技术方案",
            "保护期限": "10 年",
            "审查周期": "6-8 个月",
            "费用": "申请费 500 元",
        },
        "外观设计": {
            "定义": "对产品的整体或局部形状、图案等的新设计",
            "保护期限": "15 年",
            "审查周期": "4-6 个月",
            "费用": "申请费 500 元",
        },
    }

    keyword_clean = keyword.strip()

    if keyword_clean in mock_db:
        info = mock_db[keyword_clean]
        lines = [f"【{keyword}专利信息】"]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    else:
        return (
            f"未找到关键词 '{keyword}' 相关的专利类型。\n"
            f"可用的关键词有: {', '.join(mock_db.keys())}"
        )


# 把所有工具放到一个列表里
ALL_TOOLS = [calculate, get_current_time, search_patent_info]


def demo_tool_definition():
    """
    【示例 1-3a】查看工具的定义信息

    当你用 @tool 定义工具后，LangChain 会自动提取：
    - 工具名称（函数名）
    - 描述信息（docstring）
    - 参数格式（从函数签名推断）

    这些信息会被发送给 LLM，让 LLM "知道"有哪些工具可以用。
    """
    print("\n" + "=" * 60)
    print("  示例 1-3a: 工具定义与信息")
    print("=" * 60)

    print(f"\n共定义了 {len(ALL_TOOLS)} 个工具:\n")

    for t in ALL_TOOLS:
        print(f"  📦 {t.name}")
        print(f"     描述: {t.description[:60]}...")
        if t.args_schema:
            print(f"     参数: {list(t.args_schema.model_fields.keys())}")
        print()


def demo_single_tool_call():
    """
    【示例 1-3b】手动调用单个工具

    工具本质上还是 Python 函数，可以直接调用。
    Agent 调用工具的方式和你手动调用是一样的。
    """
    print("\n" + "=" * 60)
    print("  示例 1-3b: 手动调用工具")
    print("=" * 60)

    # 直接像普通函数一样调用
    r1 = calculate.invoke({"expression": "100 * 2 + 50"})
    print(f"\n[calculate] {r1}")

    r2 = get_current_time.invoke({})
    print(f"[time]       {r2}")

    r3 = search_patent_info.invoke({"keyword": "发明专利"})
    print(f"[patent]\n{r3}")


def demo_react_agent():
    """
    【示例 1-3c】创建 ReAct Agent —— 核心中的核心！

    create_react_agent() 是 LangChain/LangGraph 提供的预构建 Agent，
    它自动实现了 ReAct 循环:

    ┌──────────────────────────────────┐
    │         ReAct 循环               │
    │                                  │
    │  ┌─────────┐                     │
    │  │ LLM 思考 │ ◀── 观察工具返回   │
    │  └────┬────┘                     │
    │       │ 决定是否调用工具          │
    │       ▼                          │
    │  ┌─────────┐                     │
    │  │ 执行工具 │                     │
    │  └────┬────┘                     │
    │       │                          │
    │       └──▶ 回到思考（循环）       │
    │                                  │
    │  直到 LLM 说"不需要更多工具了"    │
    │  → 输出最终答案                  │
    └──────────────────────────────────┘

    你只需要提供:
      1. 一个 LLM（大脑）
      2. 一组工具（手脚）
      3. （可选）系统提示词
    其余的全部由框架自动处理！
    """
    print("\n" + "=" * 60)
    print("  示例 1-3c: ReAct Agent")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent

    # ---- Step 1: 创建 LLM ----
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,          # 低温度让 Agent 更稳定
    )

    # ---- Step 2: 创建 ReAct Agent ----
    # 就这一行！LLM + Tools = Agent
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=(                    # 可选：给 Agent 设定角色
            "你是一个智能助手，可以使用以下工具来帮助用户：\n"
            "- 计算数学表达式\n"
            "- 查询当前时间\n"
            "- 查询专利相关信息\n\n"
            "请根据用户的问题，选择合适的工具来获取准确的信息。"
        ),
    )

    # ---- Step 3: 和 Agent 对话 ----
    test_questions = [
        "现在是几点？",
        "帮我算一下 (15 + 27) * 3",
        "发明专利的保护期限是多少？",
    ]

    for q in test_questions:
        print(f"\n{'─' * 40}")
        print(f"[User] {q}")
        print(f"[AI] ", end="", flush=True)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]}
        )

        # 从返回的消息中提取最后一条回复
        last_message = result["messages"][-1]
        print(last_message.content)


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangChain v1.x — 1.3 Tools & Agent 教程    ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_tool_definition()
        demo_single_tool_call()
        demo_react_agent()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
