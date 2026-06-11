"""
================================================================
  第4章 综合实战：专利流程智能助手
================================================================

【项目目标】
  把前 3 章学到的所有知识，整合成一个完整可用的 AI Agent：
    - LangChain v1.x（LCEL 链式调用）
    - LangGraph v1.x（StateGraph、条件路由、记忆、Human-in-the-Loop）
    - MCP 协议（Server + Client 集成）
    - RAG（向量检索增强）

【功能清单】
  ✅ 意图识别与路由（Conditional Routing）
  ✅ 专利信息查询（MCP Server 工具）
  ✅ RAG 本地文档问答（Chroma + Embedding）
  ✅ 多轮对话记忆（MemorySaver）
  ✅ 敏感操作人工确认（Human-in-the-Loop / interrupt）
  ✅ FastAPI 部署服务（SSE 流式输出）

【运行方式】
  # 终端 1: 启动 MCP Server
  python -m uvicorn patent_assistant.server:app --reload

  # 终端 2: 测试 CLI 交互
  python patent_assistant/agent.py

  # 终端 3: 调用 API
  curl http://localhost:8000/chat -X POST -H "Content-Type: application/json" \
       -d '{"message": "发明专利保护期限多久？"}'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHROMA_PERSIST_DIR
