"""
patent_assistant/agent.py
=====================================
主 Agent 逻辑 —— 整合全部知识点

架构概览:
  ┌───────────────────────────────────┐
  │               State                    │
  │  query / intent / messages / ...  │
  └──────────────┬────────────────────┘
                     │
    ┌──────────────▼────────────────────┐
    │          Router Node              │
    │  根据 query 判断 intent         │
    └──┬───────────┬────────────┬───┘
       │           │            │
       ▼           ▼            ▼
  ┌──────┐  ┌──────┐  ┌──────────┐
  │RAG   │  │MCP   │  │ Chat-only │
  │Node  │  │Node  │  │ Node     │
  └──┬───┘  └──┬───┘  └────┬─────┘
     │           │            │
     └───────────┼────────────┘
                 ▼
          ┌──────────────┐
          │  Memory Node  │ ← 保存对话历史
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │  Approval?   │ ← interrupt？
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │   Response    │
          │   Output      │
          └──────────────┘
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal
import operator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

# ── 导入本地模块 ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHROMA_PERSIST_DIR


# ============================================================
# Part 1: 定义 Agent 状态 (State)
# ============================================================

class AgentState(TypedDict):
    """
    Agent 的完整状态定义。

    每个节点接收这个状态，处理后返回需要更新的字段。
    Annotated[Sequence, operator.add] 表示 messages 字段是"追加模式"——
    新消息会被追加到列表末尾，而不是覆盖。
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str                          # 当前用户查询
    intent: Literal["rag", "mcp", "chat", "unknown"]
    sources: list[str]                    # RAG 检索到的来源
    needs_approval: bool                # 是否需要人工确认
    final_answer: str                  # 最终回复


# ============================================================
# Part 2: 定义各个节点 (Nodes)
# ============================================================


def router_node(state: AgentState) -> dict:
    """
    路由器节点 —— 分析用户意图，设置 intent 字段。

    这是 Conditional Routing 的关键节点：
    后续用 add_conditional_edges 根据 intent 值分发到不同分支。
    """
    query = state["query"].lower()

    # 关键词匹配（实际项目中可以用 LLM 做更智能的分类）
    rag_keywords = ["什么是", "介绍", "流程", "步骤", "怎么"]
    mcp_keywords = ["查询", "费用", "时间", "专利类型", "期限"]

    if any(kw in query for kw in rag_keywords):
        intent = "rag"
    elif any(kw in query for kw in mcp_keywords):
        intent = "mcp"
    elif any(kw in query for kw in ["你好", "你是", "聊天"]):
        intent = "chat"
    else:
        intent = "unknown"

    print(f"  [Router] query='{state['query'][:20]}...' → intent={intent}")
    return {"intent": intent, "messages": []}   # messages 空列表不会追加


def rag_node(state: AgentState) -> dict:
    """
    RAG 节点 —— 从本地向量库检索相关文档，生成回答。

    对应知识点: 01_04_rag.py（RAG 检索增强生成）
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    print(f"  [RAG] 正在检索: '{state['query']}'")

    # 初始化 Embeddings + Chroma
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text",
    )
    vectorstore = Chroma(
        collection_name="patent_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR / "patent"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG Prompt
    prompt = ChatPromptTemplate.from_template(
        "你是专利咨询助手。\n"
        "根据以下参考资料回答用户问题。\n"
        "如参考资料无相关内容，请明确说明。\n\n"
        "=== 参考资料 ===\n{context}\n\n"
        "=== 用户问题 ===\n{question}\n\n"
        "=== 回答 ==="
    )

    # RAG Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        | StrOutputParser()
    )

    answer = chain.invoke(state["query"])

    sources = [f"doc_{i}" for i in range(3)]   # 简化：实际应返回真实来源

    print(f"  [RAG] 回答生成完成 ({len(answer)} 字符)")
    return {
        "messages": [AIMessage(content=answer)],
        "sources": sources,
        "final_answer": answer,
    }


def mcp_node(state: AgentState) -> dict:
    """
    MCP 工具调用节点 —— 通过 MCP Client 调用外部工具。

    对应知识点: 03_mcp/（MCP 协议实战）
    """
    print(f"  [MCP] 正在调用 MCP 工具: '{state['query']}'")

    # 简化实现：实际应连接真实的 MCP Server
    # 这里用模拟回复代替
    mock_responses = {
        "费用": "发明专利申请费用约 3400 元（含官费+代理费），实用新型约 500 元。",
        "期限": "发明专利保护期限 20 年，实用新型 10 年，外观设计 15 年。",
        "时间": "发明专利审查周期 18-24 个月，实用新型 6-8 个月。",
    }

    answer = "关于您的问题：\n"
    for key, val in mock_responses.items():
        if key in state["query"]:
            answer = val
            break
    if answer.startswith("关于"):
        answer = "暂未查询到相关信息，请联系专利代理机构获取详细数据。"

    print(f"  [MCP] 工具调用完成")
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


def chat_node(state: AgentState) -> dict:
    """
    闲聊节点 —— 直接用 LLM 生成回复，不调用工具或检索。
    """
    print(f"  [Chat] 直接回复（无工具/RAG）")

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "final_answer": response.content,
    }


def memory_node(state: AgentState) -> dict:
    """
    记忆节点 —— 将本轮对话存入记忆。

    对应知识点: 02_05_memory.py（记忆机制）
    注意: 使用 MemorySaver 时，消息列表会自动通过 thread_id 持久化，
    此节点主要做日志记录和状态汇总。
    """
    msg_count = len(state.get("messages", []))
    print(f"  [Memory] 当前对话消息数: {msg_count}")
    # 不需要返回任何更新——MemorySaver 在 compile 时自动处理
    return {}


def approval_node(state: AgentState) -> dict:
    """
    人工审批节点 —— 如果 needs_approval=True，暂停等待人工确认。

    对应知识点: 02_06_human_in_loop.py（人机协作）

    适用场景:
      - 用户询问"帮我提交专利申请"（涉及实际操作）
      - 费用超过一定阈值
      - 删除/修改操作
    """
    if not state.get("needs_approval", False):
        print(f"  [Approval] 无需人工确认，继续执行")
        return {}

    print(f"  [Approval] ⏸ 等待人工确认...")
    print(f"  [Approval] 问题: {state['query']}")
    print(f"  [Approval] 建议回复: {state['final_answer'][:50]}...")

    # interrupt —— 暂停执行，等待外部 resume
    approval_result = interrupt(value={
        "query": state["query"],
        "proposed_answer": state["final_answer"],
        "prompt": "请审核以上回复是否准确？输入 'approve' 通过，或输入修改意见。",
    })

    print(f"  [Approval] ✅ 收到人工反馈: {str(approval_result)[:50]}")
    return {}


def response_node(state: AgentState) -> dict:
    """
    最终输出节点 —— 整理并输出最终回复。

    生产环境中，这里可以：
      - 记录日志
      - 将回复发送到前端 WebSocket
      - 更新数据库
    """
    answer = state.get("final_answer", "(无回复）")
    print(f"  [Response] 最终回复已生成 ({len(answer)} 字符)")
    return {}


# ============================================================
# Part 3: 路由函数 (Routing Functions)
# ============================================================


def route_by_intent(state: AgentState) -> str:
    """根据 intent 字段路由到对应处理节点"""
    intent = state.get("intent", "unknown")
    mapping = {
        "rag": "rag_node",
        "mcp": "mcp_node",
        "chat": "chat_node",
        "unknown": "chat_node",    # 默认走闲聊
    }
    return mapping.get(intent, "chat_node")


def route_after_processing(state: AgentState) -> str:
    """处理节点之后：判断是否需要审批"""
    if state.get("needs_approval", False):
        return "approval_node"
    return "response_node"


# ============================================================
# Part 4: 构建图 (Build Graph)
# ============================================================


def build_agent_graph():
    """
    构建完整的 Agent 状态图。

    流程图:
      START
        │
        ▼
      router_node
        │
        ├───▶ rag_node ────┐
        ├───▶ mcp_node ────┤
        └───▶ chat_node ───┤
                             ▼
                         memory_node
                             │
                             ├───(needs_approval=True)──▶ approval_node ──▶ response_node
                             └───(needs_approval=False)──────────────────▶ response_node
                                                                         │
                                                                         ▼
                                                                       END
    """
    graph = StateGraph(AgentState)

    # 添加所有节点
    graph.add_node("router", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("mcp_node", mcp_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("approval_node", approval_node)
    graph.add_node("response_node", response_node)

    # 入口 → Router
    graph.add_edge(START, "router")

    # Router → 各处理节点（条件路由）
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "rag_node": "rag_node",
            "mcp_node": "mcp_node",
            "chat_node": "chat_node",
        },
    )

    # 所有处理节点 → Memory
    for node in ["rag_node", "mcp_node", "chat_node"]:
        graph.add_edge(node, "memory_node")

    # Memory → 条件路由（是否审批）
    graph.add_conditional_edges(
        "memory_node",
        route_after_processing,
        {
            "approval_node": "approval_node",
            "response_node": "response_node",
        },
    )

    # Approval → Response
    graph.add_edge("approval_node", "response_node")

    # Response → END
    graph.add_edge("response_node", END)

    # 编译（启用 MemorySaver 实现多轮对话记忆）
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# ============================================================
# Part 5: 命令行交互入口
# ============================================================


def run_cli():
    """
    命令行交互模式 —— 方便快速测试 Agent。
    """
    app = build_agent_graph()

    # thread_id 区分不同用户的对话会话
    thread_id = "cli-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    print("╔═══════════════════════════════════════════╗")
    print("║   专利流程智能助手 — CLI 模式           ║")
    print("║   输入 'exit' 退出                              ║")
    print("╚═══════════════════════════════════════════╝")

    while True:
        try:
            user_input = input("\n[用户] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user_input.lower() in ("exit", "quit", "退出"):
            print("👋 再见！")
            break

        if not user_input:
            continue

        # 调用 Agent
        result = app.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "intent": "unknown",
                "sources": [],
                "needs_approval": False,
                "final_answer": "",
            },
            config=config,
        )

        # 提取最终回复
        msgs = result.get("messages", [])
        if msgs:
            last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            if last_ai:
                print(f"[AI] {last_ai.content}")

        # 显示来源（如果是 RAG 检索）
        sources = result.get("sources", [])
        if sources:
            print(f"  📎 参考来源: {', '.join(sources)}")


if __name__ == "__main__":
    run_cli()
