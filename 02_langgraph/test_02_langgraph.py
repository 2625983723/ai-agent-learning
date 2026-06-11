"""
================================================================
  第2章 测试用例 — LangGraph 图式编程
================================================================

运行方式:
  # 只跑单元测试（不需要 Ollama）
  pytest 02_langgraph/test_02_langgraph.py -v -m "not integration"

  # 跑全部测试
  pytest 02_langgraph/test_02_langgraph.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 单元测试 —— 不需要模型，纯图结构验证
# ============================================================


class TestFirstGraph:
    """2.1 第一个图"""

    def test_minimal_graph_runs(self):
        """最简图能正常编译和运行"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class S(TypedDict):
            msg: str
            count: int

        def node_a(state):
            return {"msg": "a", "count": 1}

        def node_b(state):
            return {"msg": state["msg"] + "b", "count": state["count"] + 1}

        g = StateGraph(S)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)

        result = g.compile().invoke({"msg": "", "count": 0})
        assert result["msg"] == "ab"
        assert result["count"] == 2

    def test_three_node_pipeline(self):
        """三节点流水线正确执行"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class S(TypedDict):
            val: int

        def double(state): return {"val": state["val"] * 2}
        def add5(state): return {"val": state["val"] + 5}

        g = StateGraph(S)
        g.add_node("double", double)
        g.add_node("add5", add5)
        g.add_edge(START, "double")
        g.add_edge("double", "add5")
        g.add_edge(add5, END)

        r = g.compile().invoke({"val": 3})    # 3*2=6+5=11
        assert r["val"] == 11


class TestStateGraph:
    """2.2 状态图"""

    def test_state_merge_overwrite(self):
        """默认状态合并：后写入覆盖先写入"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class S(TypedDict):
            value: str

        g = StateGraph(S)

        def write_foo(s): return {"value": "foo"}
        def write_bar(s): return {"value": "bar"}

        g.add_node("f", write_foo)
        g.add_node("b", write_bar)
        g.add_edge(START, "f")
        g.add_edge("f", "b")
        g.add_edge(b, END)

        r = g.compile().invoke({"value": ""})
        assert r["value"] == "bar"     # bar 覆盖了 foo

    def test_state_append_with_reducer(self):
        """使用 operator.add reducer 追加列表"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict, Annotated
        import operator

        class S(TypedDict):
            items: Annotated[list, operator.add]

        def add_a(s): return {"items": ["a"]}
        def add_b(s): return {"items": ["b"]}
        def add_c(s): return {"items": ["c"]}

        g = StateGraph(S)
        for name, fn in [("a", add_a), ("b", add_b), ("c", add_c)]:
            g.add_node(name, fn)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)

        r = g.compile().invoke({"items": []})
        assert r["items"] == ["a", "b", "c"]

    def test_message_list_pattern(self):
        """消息列表模式：每个节点追加一条消息"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict, Annotated
        import operator
        from langchain_core.messages import HumanMessage, AIMessage

        class S(TypedDict):
            messages: Annotated[list, operator.add]

        def bot(s):
            return {"messages": [AIMessage(content="reply")]}

        g = StateGraph(S)
        g.add_node("bot", bot)
        g.add_edge(START, "bot")
        g.add_edge(bot, END)

        init_msgs = [HumanMessage(content="hello")]
        r = g.compile().invoke({"messages": list(init_msgs)})
        assert len(r["messages"]) == 2
        assert isinstance(r["messages"][1], AIMessage)


class TestConditionalRouting:
    """2.3 条件路由"""

    def test_if_else_routing(self):
        """二选一条件路由"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class S(TypedDict):
            score: int
            result: str

        def pass_fn(s): return {"result": "pass"}
        def fail_fn(s): return {"result": "fail"}

        def route(s): return "pass" if s["score"] >= 60 else "fail"

        g = StateGraph(S)
        g.add_node("pass", pass_fn)
        g.add_node("fail", fail_fn)
        g.add_edge(START, "judge")   # judge 是隐含的起始点
        # 实际上我们需要从 START 出发的边到某个节点
        # 用条件边来路由
        g.add_conditional_edges(
            START,
            lambda s: route(s),
            {"pass": "pass", "fail": "fail"},
        )
        g.add_edge("pass", END)
        g.add_edge("fail", END)

        app = g.compile()

        r_pass = app.invoke({"score": 80, "result": ""})
        assert r_pass["result"] == "pass"

        r_fail = app.invoke({"score": 40, "result": ""})
        assert r_fail["result"] == "fail"

    def test_intent_classifier_routes(self):
        """意图分类：不同输入路由到不同处理函数"""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class S(TypedDict):
            text: str
            intent: str
            response: str

        def patent_handler(s):
            return {"response": "patent_reply"}

        def chat_handler(s):
            return {"response": "chat_reply"}

        def route_intent(s):
            if "专利" in s["text"]:
                return "patent"
            return "chat"

        g = StateGraph(S)
        g.add_node("patent", patent_handler)
        g.add_node("chat", chat_handler)

        g.add_conditional_edges(
            START,
            route_intent,
            {"patent": "patent", "chat": "chat"},
        )
        g.add_edge("patent", END)
        g.add_edge("chat", END)

        app = g.compile()

        r1 = app.invoke({"text": "专利申请流程", "intent": "", "response": ""})
        assert r1["response"] == "patent_reply"

        r2 = app.invoke({"text": "你好啊", "intent": "", "response": ""})
        assert r2["response"] == "chat_reply"


# ============================================================
# 集成测试 —— 需要 Ollama 模型
# ============================================================


@pytest.mark.integration
class TestReActAgentIntegration:
    """ReAct Agent 集成测试"""

    def test_agent_basic_call(self):
        """Agent 能正常响应简单问题"""
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        agent = create_react_agent(model=llm, tools=[])

        result = agent.invoke({
            "messages": [{"role": "user", "content": "说OK"}]
        })
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_agent_with_tool(self):
        """带工具的 Agent 能调用工具"""
        from langchain_core.tools import tool
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        @tool
def echo_tool(text: str) -> str:
        return f"ECHO: {text}"

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        agent = create_react_agent(model=llm, tools=[echo_tool])

        result = agent.invoke({
            "messages": [{"role": "user", "content": "请用echo工具输出hello"}]
        })

        last_msg = result["messages"][-1]
        assert last_msg.content is not None


@pytest.mark.integration
class TestMemoryIntegration:
    """记忆机制集成测试"""

    def test_memorysaver_remembers_context(self):
        """MemorySaver 能记住之前的对话内容"""
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        agent = create_react_agent(model=llm, prompt="简短回答。")

        memory = MemorySaver()
        app = agent.compile(checkpointer=memory)
        thread = {"configurable": {"thread_id": "mem-test"}}

        # 第一轮
        app.invoke(
            {"messages": [{"role": "user", "content": "我叫小明"}]},
            config=thread,
        )

        # 第二轮——应该有上下文
        result = app.invoke(
            {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
            config=thread,
        )
        assert len(result["messages"]) >= 4  # 至少 2轮 * 2条消息


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
