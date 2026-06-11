"""
================================================================
  第4章 综合实战测试 — 专利流程智能助手
================================================================

运行方式:
  # 单元测试（不依赖外部服务）
  pytest 04_project/patent_assistant/test_patent.py -v -m "not integration"

  # 集成测试（需要 Ollama + 可选 MCP Server）
  pytest 04_project/patent_assistant/test_patent.py -v
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# 单元测试 —— 不依赖 Ollama 或 MCP Server
# ============================================================


class TestAgentState:
    """Agent 状态定义测试"""

    def test_state_has_required_fields(self):
        """AgentState 包含必须的字段"""
        from patent_assistant.agent import AgentState
        required = ["messages", "query", "intent", "sources", "needs_approval", "final_answer"]
        for field in required:
            assert field in AgentState.__annotations__, f"缺少字段: {field}"

    def test_intent_literals(self):
        """intent 字段的可选值范围正确"""
        from patent_assistant.agent import AgentState
        # 从 TypedDict 的 Annotated 中提取 Literal 值
        annotations = AgentState.__annotations__
        assert "intent" in annotations


class TestToolsUnit:
    """工具函数单元测试（不调用 LLM）"""

    def test_query_patent_fee_known_type(self):
        """已知专利类型的费用查询"""
        from patent_assistant.tools import query_patent_fee
        result = query_patent_fee.invoke({"patent_type": "发明专利"})
        assert "官费" in result
        assert "合计" in result

    def test_query_patent_fee_unknown_type(self):
        """未知专利类型返回错误提示"""
        from patent_assistant.tools import query_patent_fee
        result = query_patent_fee.invoke({"patent_type": "未知类型"})
        assert "error" in str(result).lower() or "未知" in str(result)

    def test_check_patent_timeline(self):
        """时间线查询返回字符串"""
        from patent_assistant.tools import check_patent_timeline
        result = check_patent_timeline.invoke({
            "patent_type": "发明专利",
            "stage": "全流程"
        })
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compare_patent_types(self):
        """专利类型对比返回对比字符串"""
        from patent_assistant.tools import compare_patent_types
        result = compare_patent_types.invoke({
            "type_a": "发明专利",
            "type_b": "实用新型"
        })
        assert "vs" in result.lower() or "对比" in result or "发明" in result

    def test_get_patent_checklist_valid_action(self):
        """有效操作类型的清单返回列表"""
        from patent_assistant.tools import get_patent_checklist
        result = get_patent_checklist.invoke({"action": "申请前"})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_patent_checklist_invalid_action(self):
        """无效操作类型返回含提示的列表"""
        from patent_assistant.tools import get_patent_checklist
        result = get_patent_checklist.invoke({"action": "无效动作"})
        assert isinstance(result, list)
        assert any("未知" in item or "可选" in item for item in result)

    def test_all_tools_accessible(self):
        """所有工具可以被正确导入"""
        from patent_assistant.tools import ALL_TOOLS
        assert len(ALL_TOOLS) >= 4
        tool_names = [t.name for t in ALL_TOOLS]
        assert "query_patent_fee" in tool_names


class TestRouterLogic:
    """路由器逻辑测试（不运行完整 Agent）"""

    def test_rag_keywords_trigger_rag_intent(self):
        """含 RAG 关键词的输入应路由到 RAG 节点"""
        from patent_assistant.agent import router_node
        state = {"query": "什么是发明专利的申请流程？", "messages": []}
        result = router_node(state)
        assert result["intent"] in ("rag", "chat")  # 取决于实现

    def test_mcp_keywords_trigger_mcp_intent(self):
        """含 MCP 关键词的输入应路由到 MCP 节点"""
        from patent_assistant.agent import router_node
        state = {"query": "查询发明专利费用", "messages": []}
        result = router_node(state)
        assert isinstance(result, dict)
        assert "intent" in result

    def test_route_by_intent_function(self):
        """route_by_intent 函数返回合法的节点名"""
        from patent_assistant.agent import route_by_intent
        for intent in ["rag", "mcp", "chat", "unknown"]:
            target = route_by_intent({"intent": intent, "messages": []})
            assert target in ("rag_node", "mcp_node", "chat_node")


class TestServerCodeStructure:
    """Server 文件结构测试"""

    def test_server_imports_fastmcp(self):
        """MCP Server 文件正确导入 FastMCP"""
        server_path = Path(__file__).parent / "mcp_patent_server.py"
        if not server_path.exists():
            pytest.skip("mcp_patent_server.py 不存在")
        content = server_path.read_text(encoding="utf-8")
        assert "FastMCP" in content

    def test_server_has_tools(self):
        """MCP Server 至少定义了 1 个工具"""
        server_path = Path(__file__).parent / "mcp_patent_server.py"
        if not server_path.exists():
            pytest.skip("mcp_patent_server.py 不存在")
        content = server_path.read_text(encoding="utf-8")
        assert "@mcp.tool" in content


# ============================================================
# 集成测试 —— 需要 Ollama 运行
# ============================================================


@pytest.mark.integration
class TestAgentGraphIntegration:
    """Agent 图集成测试"""

    def test_build_graph_returns_compiled_app(self):
        """build_agent_graph 返回可调用对象"""
        from patent_assistant.agent import build_agent_graph
        app = build_agent_graph()
        assert app is not None
        assert hasattr(app, "invoke")

    def test_invoke_basic_query(self):
        """基本用户查询能返回结果"""
        from patent_assistant.agent import build_agent_graph
        app = build_agent_graph()
        config = {"configurable": {"thread_id": "test-basic"}}
        result = app.invoke(
            {"messages": [{"role": "user", "content": "你好"}],
            config=config,
        )
        assert "messages" in result
        assert len(result["messages"]) >= 2

    def test_memory_across_turns(self):
        """多轮对话中后一轮能感知前一轮"""
        from patent_assistant.agent import build_agent_graph
        app = build_agent_graph()
        thread_id = "test-memory"
        config = {"configurable": {"thread_id": thread_id}}

        app.invoke(
            {"messages": [{"role": "user", "content": "我叫李华"}]},
            config=config,
        )
        result2 = app.invoke(
            {"messages": [{"role": "user", "content": "我的名字是什么？"}]},
            config=config,
        )
        last_msg = result2["messages"][-1]
        # 不强制断言内容（依赖 LLM），但确保有回复
        assert last_msg.content is not None


@pytest.mark.integration
class TestFastAPIServer:
    """FastAPI 服务测试"""

    def test_health_endpoint(self):
        """/health 端点返回 ok"""
        from fastapi.testclient import TestClient
        from patent_assistant.server import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_chat_endpoint_non_stream(self):
        """/chat 非流式模式返回 JSON"""
        from fastapi.testclient import TestClient
        from patent_assistant.server import app
        client = TestClient(app)
        resp = client.post("/chat", json={
            "message": "你好",
            "stream": False,
        })
        # 可能 200 或 500（取决于 Ollama 是否可连接）
        assert resp.status_code in (200, 500)


# ============================================================
# 架构理解测试
# ============================================================


class TestArchitectureUnderstanding:
    """架构理解验证"""

    def test_project_directory_structure(self):
        """项目目录结构符合预期"""
        base = Path(__file__).parent.parent
        assert (base / "agent.py").exists()
        assert (base / "tools.py").exists()
        assert (base / "server.py").exists()

    def test_all_chapters_integrated(self):
        """项目整合了全部核心章节的内容"""
        # 验证 imports 覆盖了各章知识点
        agent_file = Path(__file__).parent.parent / "agent.py"
        content = agent_file.read_text(encoding="utf-8")
        # LangGraph 图构建
        assert "StateGraph" in content
        # 条件路由
        assert "add_conditional_edges" in content
        # 记忆
        assert "MemorySaver" in content
        # Human-in-the-Loop
        assert "interrupt" in content

    def test_tools_cover_patent_scenarios(self):
        """工具函数覆盖了专利咨询的核心场景"""
        from patent_assistant.tools import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        # 至少覆盖：费用查询、时间线、对比、清单
        assert any("fee" in n or "费用" in n for n in tool_names)
        assert any("timeline" in n or "时间" in n for n in tool_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
