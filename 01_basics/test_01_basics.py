"""
================================================================
  第1章 测试用例 — LangChain v1.x 基础
================================================================

测试策略：
  - 每个示例函数都有对应的测试
  - 使用 pytest 框架
  - 不依赖 Ollama 的测试用 @pytest.mark.unit 标记（快速跑）
  - 需要模型的测试用 @pytest.mark.integration 标记

运行方式:
  # 跑全部测试（跳过需要模型的）
  pytest 01_basics/test_01_basics.py -v -m "not integration"

  # 跑全部测试（包括需要模型的）
  pytest 01_basics/test_01_basics.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 单元测试 —— 不需要 Ollama 模型，可以快速运行
# ============================================================


class TestConfig:
    """配置模块测试"""

    def test_config_import(self):
        """配置模块能正常导入"""
        from config import get_llm_config, OLLAMA_MODEL
        cfg = get_llm_config()
        assert cfg["provider"] == "ollama"
        assert isinstance(OLLAMA_MODEL, str)
        assert len(OLLAMA_MODEL) > 0

    def test_chroma_persist_dir(self):
        """Chroma 数据目录路径正确"""
        from config import CHROMA_PERSIST_DIR
        assert "chroma" in str(CHROMA_PERSIST_DIR)


class TestToolDefinition:
    """工具定义测试（不需要模型）"""

    def test_calculate_tool_exists(self):
        """calculate 工具可导入且结构正确"""
        from basics_01_03_tools_agent import calculate
        assert calculate.name == "calculate"
        assert "数学" in calculate.description or "计算" in calculate.description

    def test_calculate_basic_math(self):
        """calculate 工具能做基本运算"""
        from basics_01_03_tools_agent import calculate
        result = calculate.invoke({"expression": "2+3"})
        assert "5" in result.content

    def test_calculator_complex(self):
        """calculate 工具处理复杂表达式"""
        from basics_01_03_tools_agent import calculate
        result = calculate.invoke({"expression": "(10+5)*3-20"})
        assert "25" in result.content

    def test_time_tool(self):
        """get_current_time 工具返回包含时间信息"""
        from basics_01_03_tools_agent import get_current_time
        result = get_current_time.invoke({})
        assert "当前时间" in result.content or len(result.content) > 0

    def test_patent_search_found(self):
        """patent search 能找到已知关键词"""
        from basics_01_03_tools_agent import search_patent_info
        result = search_patent_info.invoke({"keyword": "发明专利"})
        assert "发明专利" in result.content
        assert "20 年" in result.content

    def test_patent_search_not_found(self):
        """patent search 对未知关键词给出友好提示"""
        from basics_01_03_tools_agent import search_patent_info
        result = search_patent_info.invoke({"keyword": "未知类型"})
        assert "未找到" in result.content or "可用" in result.content

    def test_all_tools_count(self):
        """工具列表数量正确"""
        from basics_01_03_tools_agent import ALL_TOOLS
        assert len(ALL_TOOLS) == 3
        tool_names = [t.name for t in ALL_TOOLS]
        assert "calculate" in tool_names
        assert "get_current_time" in tool_names
        assert "search_patent_info" in tool_names


class TestSampleDocuments:
    """RAG 示例文档数据测试"""

    def test_document_count(self):
        """有足够的示例文档用于 RAG 测试"""
        from basics_01_04_rag import SAMPLE_DOCUMENTS
        assert len(SAMPLE_DOCUMENTS) >= 3

    def test_documents_contain_patent_content(self):
        """文档内容与专利相关"""
        from basics_01_04_rag import SAMPLE_DOCUMENTS
        combined = " ".join(SAMPLE_DOCUMENTS)
        assert "专利" in combined
        assert "发明" in combined


class TestLCELConcepts:
    """LCEL 概念理解测试（代码层面的验证）"""

    def test_prompt_template_variable(self):
        """PromptTemplate 能正确识别变量"""
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("解释{topic}")
        variables = prompt.input_variables
        assert "topic" in variables

    def test_str_output_parser(self):
        """StrOutputParser 存在且可调用"""
        from langchain_core.output_parsers import StrOutputParser
        parser = StrOutputParser()
        # parser 可以解析 AIMessage 对象
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="hello world")
        result = parser.invoke(msg)
        assert result == "hello world"


# ============================================================
# 集成测试 —— 需要 Ollama 模型
# ============================================================


@pytest.mark.integration
class TestModelIOIntegration:
    """Model I/O 集成测试"""

    def test_basic_invoke(self):
        """基础 invoke 调用成功"""
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        response = llm.invoke([HumanMessage(content="说'你好'")])
        assert response.content is not None
        assert len(response.content) > 0

    def test_stream_returns_chunks(self):
        """stream() 返回多个 chunk"""
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        chunks = list(llm.stream([HumanMessage(content="说OK")]))
        assert len(chunks) > 0


@pytest.mark.integration
class TestChainIntegration:
    """LCEL 链式调用集成测试"""

    def test_simple_chain_works(self):
        """简单链能正常执行"""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        chain = (
            ChatPromptTemplate.from_template("说{word}")
            | ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            | StrOutputParser()
        )
        result = chain.invoke({"word": "你好"})
        assert isinstance(result, str)
        assert len(result.strip()) > 0


@pytest.mark.integration
class TestAgentIntegration:
    """Agent 集成测试"""

    def test_react_agent_can_use_tool(self):
        """ReAct Agent 能调用工具并返回结果"""
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from basics_01_03_tools_agent import get_current_time, ALL_TOOLS
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        agent = create_react_agent(
            model=llm,
            tools=[get_current_time],
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "现在几点了？"}]
        })

        # 应该有消息返回
        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert last_msg.content is not None


if __name__ == "__main__":
    # 允许直接 python 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
