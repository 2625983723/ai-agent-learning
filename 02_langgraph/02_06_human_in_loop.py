"""
================================================================
  2.6 人机协作 (Human-in-the-Loop) — 关键操作需人工确认
================================================================

【学习目标】
  - 掌握 interrupt()：让 Agent 在指定节点暂停等待人工确认
  - 理解 Command 模式：人工审批后如何恢复执行
  - 学会设计需要审批的工作流（如提交前的最终确认）

【前置知识】
  - 02_04_react_agent.py（ReAct Agent）
  - 02_05_memory.py（记忆机制）

【核心概念】

  传统 Agent（全自动）:
    用户: "帮我提交这份专利申请"
    Agent: 自动提交... 完成！
    ⚠️ 问题: 如果填错了怎么办？用户没有机会检查！

  Human-in-the-Loop（人机协作）:
    用户: "帮我提交这份专利申请"
    Agent:
      ① 填写表单...
      ② **暂停** —— "请确认以下信息是否正确"
         - 发明名称: XXX
         - 申请人: XXX
         - 费用: XXX 元
      ③ **人工审核**
         → 用户: "没问题，提交吧"
      ④ Agent 继续执行: 提交完成！

  适用场景:
    - 金融交易确认（"确定要转账？"）
    - 文件删除/修改确认（"确定要删除？"）
    - 邮件发送前预览（"确认发送？"）
    - 专利提交前最终核对 ← 你的实际工作场景！
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def demo_basic_interrupt():
    """
    【示例 2-6a】基础 interrupt 用法

    最简单的人机交互：Agent 执行到某个节点后暂停，
    等待人工确认后才能继续。
    """
    print("\n" + "=" * 60)
    print("  示例 2-6a: 基础 interrupt")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from typing import TypedDict
    from langgraph.checkpoint.memory import MemorySaver

    class ApprovalState(TypedDict):
        request: str
        approval_status: str       # pending / approved / rejected
        result: str

    def prepare(state: ApprovalState) -> dict:
        """准备阶段"""
        print(f"\n  [准备] 处理请求: {state['request']}")
        return {"approval_status": "pending", "result": ""}

    def human_approval_node(state: ApprovalState) -> dict:
        """审批节点：interrupt 暂停等待人工输入"""
        print(f"\n  [⏸ 审批] 等待人工确认...")
        print(f"  待审批内容: {state['request']}")

        # interrupt() 会在这里暂停图执行！
        user_response = interrupt(value={
            "action": state["request"],
            "message": f"请确认是否执行: {state['request']}"
        })

        # 用户回复后会从这里继续执行
        print(f"  [✓ 收到回复] {user_response}")
        return {
            "approval_status": "approved",
            "result": f"已批准并执行: {state['request']}"
        }

    def finalize(state: ApprovalState) -> dict:
        """结束节点"""
        print(f"\n  [完成] 最终状态: {state['result']}")
        return {}

    # 构建带 interrupt 的图
    g = StateGraph(ApprovalState)
    g.add_node("prepare", prepare)
    g.add_node("human_approval", human_approval_node)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "human_approval")
    g.add_edge("human_approval", finalize)
    g.add_edge(finalize, END)

    # 必须有 checkpointer 才能使用 interrupt！
    app = g.compile(checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": "approval-demo"}}

    print("\n▶ 第一次运行 (会在 interrupt 处暂停)")
    try:
        result = app.invoke(
            {"request": "提交专利申请文件", "approval_status": "", "result": ""},
            config=thread,
        )
    except Exception as e:
        error_type = type(e).__name__
        print(f"  ⏸ 图已暂停于审核节点 ({error_type})")

    print(f"\n  💡 此时程序暂停在 human_approval 节点")
    print(f"     实际应用中，这里会弹出 UI 让用户操作")


def demo_patent_submission_workflow():
    """
    【示例 2-6b】专利提交审批流程 —— 实际场景模拟

    完整流程:
    START → 填写表单 → ⏸人工审核 → 提交/驳回 → END
    """
    print("\n" + "=" * 60)
    print("  示例 2-6b: 专利提交审批流程")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from typing import TypedDict, Annotated
    import operator
    from langgraph.checkpoint.memory import MemorySaver

    class PatentSubmitState(TypedDict):
        patent_name: str
        applicant: str
        patent_type: str
        form_data: str
        logs: Annotated[list[str], operator.add]
        status: str                 # draft / reviewing / submitted / rejected
        final_result: str

    def fill_form(state: PatentSubmitState) -> dict:
        """填写表单节点"""
        form_summary = (
            f"[表单]\n  发明名称: {state['patent_name']}\n"
            f"  申请人: {state['applicant']}\n"
            f"  类型: {state['patent_type']}专利\n"
            f"  提交时间: 2026-06-11\n"
        )
        log = "[步骤1] 表单已填写完成"
        print(f"\n  {log}")
        print(f"  {'  '.join(form_summary.split(chr(10)))}")

        return {"form_data": form_summary, "logs": [log], "status": "reviewing"}

    def review_and_approve(state: PatentSubmitState) -> dict:
        """人工审核节点（含 interrupt）"""
        print(f"\n  [⏸ 等待人工审核]")
        print(f"  {'─'*35}")

        approval = interrupt(value={
            "action": "review_patent_submission",
            "form_data": state["form_data"],
            "prompt": (
                f"请审查:\n{state['form_data']}\n\n"
                f"回复 'approve' 同意, 'reject' 驳回。"
            )
        })

        user_input = str(approval).strip().lower()
        log = f"[步骤2] 收到审核意见: {user_input}"

        if "approve" in user_input or "同意" in user_input or "通过" in user_input:
            return {
                "logs": [log],
                "status": "submitted",
                "final_result": "✅ 专利申请已成功提交！",
            }
        else:
            return {
                "logs": [log],
                "status": "rejected",
                "final_result": f"❌ 申请被退回。原因: {user_input}",
            }

    def execute_result(state: PatentSubmitState) -> dict:
        """根据结果执行最终操作"""
        if state["status"] == "submitted":
            log = "[步骤3] 已提交到国知局电子申请系统"
            print(f"\n  🎉 {state['final_result']}")
        else:
            log = "[步骤3] 已通知用户申请被驳回"
            print(f"\n  {state['final_result']}")
        return {"logs": [log]}

    # 构建图
    g = StateGraph(PatentSubmitState)
    g.add_node("fill_form", fill_form)
    g.add_node("review", review_and_approve)
    g.add_node("execute", execute_result)

    g.add_edge(START, "fill_form")
    g.add_edge("fill_form", "review")
    g.add_edge("review", "execute")
    g.add_edge(execute, END)

    app = g.compile(checkpointer=MemorySaver())
    thread_config = {"configurable": {"thread_id": "patent-submit-001"}}

    initial_state = {
        "patent_name": "一种基于AI的专利自动化处理方法",
        "applicant": "李华科技有限公司",
        "patent_type": "发明",
        "form_data": "",
        "logs": [],
        "status": "draft",
        "final_result": "",
    }

    print(f"\n▶ 开始专利提交流程...")
    print(f"{'='*55}")

    try:
        result = app.invoke(initial_state, config=thread_config)
        print(f"\n{'='*55}")
        print(f"\n  操作日志:")
        for log in result.get("logs", []):
            print(f"    • {log}")
        print(f"\n  最终状态: {result['status']}")

    except Exception as e:
        print(f"\n  ⏸ 流程已暂停于审核节点 ({type(e).__name__})")
        print(f"  这就是 Human-in-the-Loop 的效果！")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangGraph — 2.6 人机协作教程               ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_basic_interrupt()
        demo_patent_submission_workflow()

        print("\n✅ 全部示例运行完毕！")
        print("\n💡 Human-in-the-Loop 是生产级 Agent 的必备能力，")
        print("   特别是涉及资金、提交、删除等不可逆操作时。")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
