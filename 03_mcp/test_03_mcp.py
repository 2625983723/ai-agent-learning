"""
================================================================
  第3章 测试用例 — MCP 协议实战
================================================================

运行方式:
  # 只跑单元测试（不需要 MCP Server 运行）
  pytest 03_mcp/test_03_mcp.py -v -m "not integration"

  # 先启动 Server，再跑集成测试
  # 终端1: python 03_mcp/mcp_patent_server.py
  # 终端2: pytest 03_mcp/test_03_mcp.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 单元测试 —— 不需要实际连接 MCP Server
# ============================================================


class TestMCPConcepts:
    """MCP 概念理解测试"""

    def test_transport_modes(self):
        """三种传输方式"""
        modes = ["stdio", "sse", "streamable-http"]
        assert "stdio" in modes
        assert "sse" in modes
        assert "streamable-http" in modes

    def test_fastmcp_decorators(self):
        """FastMCP 的三大装饰器"""
        decorators = ["@mcp.tool", "@mcp.resource", "@mcp.prompt"]
        for d in decorators:
            assert d.startswith("@mcp.")

    def test_mcp_adapter_package(self):
        """langchain-mcp-adapters 包存在"""
        try:
            import langchain_mcp_adapters
            assert True
        except ImportError:
            pytest.skip("langchain-mcp-adapters 未安装")


class TestServerCodeStructure:
    """Server 代码结构测试（不实际运行）"""

    def test_server_file_exists(self):
        """Server 文件存在"""
        server_path = Path(__file__).parent / "mcp_patent_server.py"
        assert server_path.exists(), "mcp_patent_server.py 不存在，请先运行 mcp_server_demo.py 生成"

    def test_server_has_fastmcp_import(self):
        """Server 文件导入了 FastMCP"""
        server_path = Path(__file__).parent / "mcp_patent_server.py"
        if not server_path.exists():
            pytest.skip("Server 文件不存在")
        content = server_path.read_text(encoding="utf-8")
        assert "FastMCP" in content, "应该导入 FastMCP"
        assert "@mcp.tool" in content, "应该定义 MCP 工具"

    def test_client_file_exists(self):
        """Client 示例文件存在"""
        client_path = Path(__file__).parent / "mcp_client_example.py"
        assert client_path.exists(), "mcp_client_example.py 不存在"


class TestPatentServerLogic:
    """专利 Server 内部逻辑测试（不启动 Server）"""

    def test_query_patent_type_logic(self):
        """专利类型查询逻辑（模拟）"""
        db = {
            "发明": {"期限": "20年"},
            "实用新型": {"期限": "10年"},
        }
        result = db.get("发明")
        assert result is not None
        assert result["期限"] == "20年"

    def test_estimate_cost_logic(self):
        """费用估算逻辑（模拟）"""
        base_fees = {"发明": 3400, "实用新型": 500}
        patent_type = "发明"
        fee = base_fees.get(patent_type, 0)
        agency = 2000
        total = fee + agency
        assert total == 5400

    def test_patent_type_literals(self):
        """专利类型常量"""
        valid_types = {"发明", "实用新型", "外观"}
        assert "发明" in valid_types
        assert len(valid_types) == 3


# ============================================================
# 集成测试 —— 需要 MCP Server 和 Ollama 运行
# ============================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPServerIntegration:
    """MCP Server 集成测试（需要启动 Server）"""

    @pytest.mark.skip(reason="需要手动启动 MCP Server")
    async def test_list_tools(self):
        """连接 Server 并列出工具"""
        from fastmcp import Client

        server_path = str(Path(__file__).parent / "mcp_patent_server.py")
        async with Client(server_path) as client:
            tools = await client.list_tools()
            assert len(tools) >= 2
            tool_names = [t.name for t in tools]
            assert "query_patent_type" in tool_names

    @pytest.mark.skip(reason="需要手动启动 MCP Server")
    async def test_call_tool(self):
        """调用 MCP 工具"""
        from fastmcp import Client

        server_path = str(Path(__file__).parent / "mcp_patent_server.py")
        async with Client(server_path) as client:
            result = await client.call_tool(
                "query_patent_type", {"patent_type": "发明"}
            )
            assert result is not None


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPLangGraphIntegration:
    """MCP + LangGraph 集成测试"""

    @pytest.mark.skip(reason="需要 Ollama + MCP Server")
    async def test_agent_with_mcp_tools(self):
        """LangGraph Agent 调用 MCP 工具"""
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from fastmcp import Client
        from langchain_mcp_adapters import load_mcp_tools
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        server_path = str(Path(__file__).parent / "mcp_patent_server.py")

        async with Client(server_path) as client:
            tools = await load_mcp_tools(client)

            llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            agent = create_react_agent(model=llm, tools=tools)

            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "发明专利期限多久？"}]
            })

            assert "messages" in result
            assert len(result["messages"]) > 0


# ============================================================
# 架构理解测试
# ============================================================


class TestArchitectureUnderstanding:
    """架构理解测试"""

    def test_mcp_role(self):
        """MCP 的角色定位"""
        # MCP Server: 提供工具/资源
        # MCP Client: 连接 Server 调用工具
        # LangGraph Agent: 使用工具完成任务
        roles = {"server", "client", "agent"}
        assert "server" in roles
        assert "client" in roles
        assert "agent" in roles

    def test_data_flow(self):
        """数据流方向"""
        # User → Agent → MCP Client → MCP Server → Tool Result → Agent → User
        flow = ["user", "agent", "client", "server", "result", "agent", "user"]
        assert flow[0] == "user"
        assert flow[-1] == "user"
        assert "server" in flow

    def test_when_to_use_mcp(self):
        """什么场景用 MCP"""
        # MCP 适合：工具需要独立部署、多客户端复用、标准化接口
        good_cases = [
            "工具需要独立升级部署",
            "多个 Agent 共用同一套工具",
            "工具需要用其他语言实现",
        ]
        assert len(good_cases) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
