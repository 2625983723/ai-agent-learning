"""
================================================================
  2.3 条件路由 (Conditional Routing) — 让图能"做判断"
================================================================

【学习目标】
  - 掌握 add_conditional_edges：根据状态动态决定下一步去哪
  - 理解路由函数：接收状态，返回下一个节点名
  - 学会实现 if/else 分支、多路分支

【前置知识】
  - 02_01_first_graph.py（基本图）
  - 02_02_state_graph.py（状态管理）

【核心概念】

  普通边 (add_edge):
    A → B    总是从 A 到 B，固定不变

  条件边 (add_conditional_edges):
    A ─┬→ B   根据状态判断去向
       ├→ C
       └→ D

    就像铁路的"道岔"，根据信号切换轨道。

  路由函数:
    def my_router(state) -> str:
        # 根据 state 中的数据做判断
        if state["score"] > 80:
            return "pass_node"    # 去通过节点
        else:
            return "fail_node"     # 去失败节点
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_if_else_routing():
    """
    【示例 2-3a】if/else 二选一路由

    场景：根据分数判断是否及格
    """
    print("\n" + "=" * 60)
    print("  示例 2-3a: if/else 条件路由")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class GradeState(TypedDict):
        name: str
        score: int
        result: str

    def judge(state: GradeState) -> dict:
        """评分节点：不做判断只记录"""
        print(f"  [评分] {state['name']} 得分: {state['score']}")
        return {}

    def pass_node(state: GradeState) -> dict:
        """通过节点"""
        result = f"✅ {state['name']} 通过！({state['score']}分)"
        print(f"  {result}")
        return {"result": result}

    def fail_node(state: GradeState) -> dict:
        """不通过节点"""
        result = f"❌ {state['name']} 未通过。({state['score']}分)"
        print(f"  {result}")
        return {"result": result}

    # --- 路由函数 ---
    def route_by_score(state: GradeState) -> str:
        """根据分数决定下一步走向哪个节点"""
        if state["score"] >= 60:
            return "pass_node"
        else:
            return "fail_node"

    # --- 构建图 ---
    g = StateGraph(GradeState)

    g.add_node("judge", judge)
    g.add_node("pass_node", pass_node)
    g.add_node("fail_node", fail_node)

    g.add_edge(START, "judge")

    # 关键：条件边！从 judge 出发，根据 route_by_score 的返回值选择路径
    g.add_conditional_edges(
        "judge",              # 从哪个节点出发
        route_by_score,       # 路由函数（决定去向）
        {
            "pass_node": "pass_node",   # 返回值 → 目标节点
            "fail_node": "fail_node",
        },
    )

    g.add_edge("pass_node", END)
    g.add_edge("fail_node", END)

    app = g.compile()

    # 测试不同分数
    for name, score in [("张三", 85), ("李四", 45), ("王五", 60)]:
        print(f"\n{'─' * 30}")
        r = app.invoke({"name": name, "score": score, "result": ""})
        print(f"  → 最终: {r['result']}")


def demo_intent_classifier():
    """
    【示例 2-3b】意图分类路由 —— Agent 实际应用模式

    这是 AI Agent 最常用的路由方式：
    用户说一句话 → 判断意图 → 分发给不同的处理节点

    类比：就像公司的前台接电话后转给不同部门。
    """
    print("\n" + "=" * 60)
    print("  示例 2-3b: 意图分类路由")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class IntentState(TypedDict):
        user_input: str
        intent: str
        response: str

    def router_node(state: IntentState) -> dict:
        """路由器节点：分析用户输入，识别意图（这里用关键词匹配模拟）"""
        text = state["user_input"].lower()

        if any(w in text for w in ["专利", "申请", "发明", "实用新型"]):
            intent = "patent"
        elif any(w in text for w in ["天气", "温度", "下雨", "晴天"]):
            intent = "weather"
        elif any(w in text for w in ["计算", "加", "减", "乘"]):
            intent = "calculate"
        else:
            intent = "chat"

        print(f"  [路由] '{state['user_input']}' → 意图: {intent}")
        return {"intent": intent}

    def handle_patent(state: IntentState) -> dict:
        resp = "[专利助手] 已为您查询到相关专利信息。发明专利保护期限20年。"
        print(f"  {resp}")
        return {"response": resp}

    def handle_weather(state: IntentState) -> dict:
        resp = "[天气服务] 抱歉，暂无法获取实时天气数据。"
        print(f"  {resp}")
        return {"response": resp}

    def handle_calculate(state: IntentState) -> dict:
        resp = "[计算器] 请提供具体的数学表达式。"
        print(f"  {resp}")
        return {"response": resp}

    def handle_chat(state: IntentState) -> dict:
        resp = f"[闲聊] 收到：{state['user_input']}"
        print(f"  {resp}")
        return {"response": resp}

    # 路由函数
    def intent_router(state: IntentState) -> str:
        return state["intent"]

    # 构建图
    g = StateGraph(IntentState)
    g.add_node("router", router_node)
    g.add_node("patent_handler", handle_patent)
    g.add_node("weather_handler", handle_weather)
    g.add_node("calc_handler", handle_calculate)
    g.add_node("chat_handler", handle_chat)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        intent_router,
        {
            "patent": "patent_handler",
            "weather": "weather_handler",
            "calculate": "calc_handler",
            "chat": "chat_handler",
        }
    )
    for node_name in ["patent_handler", "weather_handler", "calc_handler", "chat_handler"]:
        g.add_edge(node_name, END)

    app = g.compile()

    test_inputs = [
        "我想了解发明专利的流程",
        "今天会下雨吗？",
        "帮我算一下 123 + 456",
        "你好啊！",
    ]

    for user_text in test_inputs:
        print(f"\n{'─' * 40}")
        print(f"[用户] {user_text}")
        r = app.invoke({
            "user_input": user_text,
            "intent": "",
            "response": "",
        })
        print(f"  → {r['response']}")


def demo_loop_with_conditional():
    """
    【示例 2-3c】带循环的条件路由 —— 重试机制

    条件路由不仅可以分发到不同的终点，
    还可以回到前面的节点形成循环。

    场景：
      处理任务 → 检查结果 → 如果失败则重试（最多3次）
    """
    print("\n" + "=" * 60)
    print("  示例 2-3c: 循环条件路由 — 重试机制")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class RetryState(TypedDict):
        task: str
        attempts: int
        max_attempts: int
        success: bool
        status: str

    def process_task(state: RetryState) -> dict:
        """执行任务的节点（模拟有概率失败）"""
        attempt = state["attempts"] + 1
        # 模拟：第3次尝试一定成功
        is_success = (attempt >= 3)
        print(f"  [执行] 第 {attempt} 次尝试... {'成功!' if is_success else '失败'}")

        return {
            "attempts": attempt,
            "success": is_success,
            "status": "done" if is_success else "retrying",
        }

    def check_and_retry(state: RetryState) -> str:
        """检查是否需要重试的路由函数"""
        if state["success"]:
            return "success_end"
        elif state["attempts"] >= state["max_attempts"]:
            return "max_retries_exceeded"
        else:
            return "process_task"   # 回到执行节点重试！

    def success_node(state: RetryState) -> dict:
        msg = f"🎉 任务完成! 共尝试 {state['attempts']} 次"
        print(f"  {msg}")
        return {"status": msg}

    def fail_node(state: RetryState) -> dict:
        msg = f"⚠️ 达到最大重试次数 ({state['max_attempts']} 次)，放弃"
        print(f"  {msg}")
        return {"status": msg}

    # 构建带循环的图
    g = StateGraph(RetryState)
    g.add_node("process", process_task)
    g.add_node("success_end", success_node)
    g.add_node("fail_end", fail_node)

    g.add_edge(START, "process")
    g.add_conditional_edges(
        "process",
        check_and_retry,
        {
            "success_end": "success_end",
            "max_retries_exceeded": "fail_end",
            "process_task": "process",     # ← 循环回 process!
        }
    )
    g.add_edge("success_end", END)
    g.add_edge("fail_end", END)

    app = g.compile()
    r = app.invoke({
        "task": "提交专利文件",
        "attempts": 0,
        "max_attempts": 5,
        "success": False,
        "status": "",
    })

    print(f"\n  最终状态: attempts={r['attempts']}, success={r['success']}")
    assert r["attempts"] == 3  # 第3次才成功


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.3 条件路由教程               ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_if_else_routing()
        demo_intent_classifier()
        demo_loop_with_conditional()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
