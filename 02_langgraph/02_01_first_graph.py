"""
================================================================
  2.1 你的第一个图 (First Graph) — LangGraph 入门
================================================================

【学习目标】
  - 理解"图式编程"的思维：用节点(Node)和边(Edge)描述流程
  - 学会创建 StateGraph、添加 Node、添加 Edge、编译运行
  - 掌握 LangGraph 最基本的"Hello World"

【前置知识】
  - Python 基础（函数、字典）
  - 第 1 章 LangChain 基础（了解即可）

【核心概念图解】

  传统编程（线性）:
    Step 1 → Step 2 → Step 3 → 结束
    像一条直线，只能从头走到尾。

  图式编程 (LangGraph):
         ┌──→ [节点 B] ──┐
         │               ↓
    [起点] → [节点 A] ──→ [终点]
         │               ↑
         └──→ [节点 C] ──┘

    可以分支、可以循环、可以并行。
    每个节点是一个函数，边决定执行顺序。

  核心组件:
    ┌─────────────┬──────────────────────────────────┐
    │ 组件        │ 说明                              │
    ├─────────────┼──────────────────────────────────┤
    │ State       │ 状态：节点之间共享的数据（字典）   │
    │ Node（节点）│ 一个 Python 函数，处理状态并返回  │
    │ Edge（边）  │ 连接节点，决定从 A 之后去哪里      │
    │ Graph       │ 把 Node 和 Edge 组装成完整流程     │
    │ Compile     │ 编译图，变成可运行的对象           │
    │ Invoke      │ 运行图，传入初始状态              │
    └─────────────┴──────────────────────────────────┘
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_minimal_graph():
    """
    【示例 2-1a】最小化的图 —— 只有两个节点 + 一条边

    这是最简单的 LangGraph 程序：
    节点 A → 节点 B

    即使这么简单，它也包含了 LangGraph 的所有核心步骤：
    1. 定义状态类型
    2. 创建 StateGraph
    3. 添加节点
    4. 添加边
    5. 编译
    6. 运行
    """
    print("\n" + "=" * 60)
    print("  示例 2-1a: 最小化的图")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    # ---- Step 1: 定义状态 ----
    # TypedDict 是一个带类型提示的字典，
    # 它定义了在各个节点之间传递的数据结构。
    class SimpleState(TypedDict):
        message: str          # 存储消息文本
        counter: int          # 计数器

    # ---- Step 2: 定义节点函数 ----
    # 每个节点都是一个 Python 函数
    # 参数是当前的状态(state)，返回值是更新后的状态
    def node_a(state: SimpleState) -> dict:
        """第一个节点：处理输入消息"""
        print(f"  [节点A] 收到: {state.get('message', '(空)')}")
        return {
            "message": f"[A处理过] {state['message']}",
            "counter": state.get("counter", 0) + 1,
        }

    def node_b(state: SimpleState) -> dict:
        """第二个节点：最终输出"""
        print(f"  [节点B] 收到: {state.get('message')}")
        return {
            "message": f"[B完成] {state['message']}",
            "counter": state["counter"] + 1,
        }

    # ---- Step 3: 构建图 ----
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    # 添加边：定义节点之间的连接关系
    graph.add_edge(START, "node_a")    # 起点 → 节点A
    graph.add_edge("node_a", "node_b") # 节点A → 节点B
    graph.add_edge("node_b", END)      # 节点B → 终点

    # ---- Step 4: 编译图 ----
    app = graph.compile()

    # ---- Step 5: 运行图 ----
    initial_state = {
        "message": "Hello LangGraph!",
        "counter": 0,
    }
    result = app.invoke(initial_state)

    print(f"\n  最终结果:")
    print(f"    message : {result['message']}")
    print(f"    counter : {result['counter']}")
    assert result["counter"] == 2, "应该经过两个节点"


def demo_three_nodes():
    """
    【示例 2-1b】三个节点的图 —— 更接近真实场景

    流程:
    START → 数据准备 → 数据处理 → 输出结果 → END
    """
    print("\n" + "=" * 60)
    print("  示例 2-1b: 三节点流水线")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class PipelineState(TypedDict):
        input_text: str
        processed: str
        result: str

    def prepare(state: PipelineState) -> dict:
        text = state["input_text"].strip().lower()
        print(f"  [准备] 清洗文本: '{state['input_text']}' → '{text}'")
        return {"processed": text}

    def process(state: PipelineState) -> dict:
        text = f"★{state['processed'].upper()}★"
        print(f"  [处理] 格式化: '{state['processed']}' → '{text}'")
        return {"result": text}

    def output(state: PipelineState) -> dict:
        print(f"  [输出] 最终结果: {state['result']}")
        return {}

    # 构建图
    g = StateGraph(PipelineState)
    g.add_node("prepare", prepare)
    g.add_node("process", process)
    g.add_node("output", output)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "process")
    g.add_edge("process", "output")
    g.add_edge(output, END)

    app = g.compile()

    result = app.invoke({"input_text": "  Hello LangGraph  ", "processed": "", "result": ""})

    print(f"\n  结果: {result}")


def demo_visualize():
    """
    【示例 2-1c】可视化图的 Mermaid 结构

    LangGraph 可以导出图的结构为 Mermaid 格式，
    方便你在文档或工具中查看流程图。
    """
    print("\n" + "=" * 60)
    print("  示例 2-1c: 图的可视化结构")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class State(TypedDict):
        value: int

    def double(state) -> dict:
        return {"value": state["value"] * 2}

    def add_one(state) -> dict:
        return {"value": state["value"] + 1}

    g = StateGraph(State)
    g.add_node("double", double)
    g.add_node("add_one", add_one)
    g.add_edge(START, "double")
    g.add_edge("double", "add_one")
    g.add_edge(add_one, END)

    app = g.compile()

    # 打印 Mermaid 格式的图结构
    mermaid = app.get_graph().draw_mermaid()
    print("\n  Mermaid 图结构:\n")

    for line in mermaid.splitlines():
        print(f"    {line}")

    # 运行验证
    result = app.invoke({"value": 5})
    print(f"\n  运行测试: value=5 → double→10 → add_one→11")
    print(f"  实际输出: {result['value']}")
    assert result["value"] == 11


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.1 第一个图教程               ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_minimal_graph()
        demo_three_nodes()
        demo_visualize()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
