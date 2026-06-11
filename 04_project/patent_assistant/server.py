"""
patent_assistant/server.py
==============================
FastAPI 部署服务 —— 将 Agent 通过 HTTP API 对外提供

对应知识点: FastAPI 部署（生产级）

启动方式:
  python -m uvicorn patent_assistant.server:app --reload --port 8000

API 端点:
  POST /chat       → 发送消息，获取回复（SSE 流式）
  GET  /health     → 健康检查
  GET  /threads    → 列出活跃会话
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── 把项目根目录加入 Python 路径 ───────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# ── 导入 Agent 构建函数 ─────────────────────────────
from .agent import build_agent_graph

# ══════════════════════════════════════════════════════╗
#  FastAPI 应用初始化
# ══════════════════════════════════════════════════════╝

app = FastAPI(
    title="专利流程智能助手 API",
    description="基于 LangGraph 的专利咨询 AI Agent 服务",
    version="1.0.0",
)


# ══════════════════════════════════════════════════════╗
#  数据模型（Pydantic）
# ══════════════════════════════════════════════════════╝


class ChatRequest(BaseModel):
    """/chat 端点的请求体"""
    message: str = Field(..., description="用户发送的消息", min_length=1)
    thread_id: Optional[str] = Field(
        default=None,
        description="会话 ID；不提供则自动生成",
    )
    stream: bool = Field(default=True, description="是否启用 SSE 流式输出")


class ChatResponse(BaseModel):
    """/chat 端点的响应体（非流式）"""
    thread_id: str
    response: str
    message_count: int


# ══════════════════════════════════════════════════════╗
#  全局 Agent 应用实例（启动后复用）
# ══════════════════════════════════════════════════════╝

_agent_app = None
_agent_lock = asyncio.Lock()


async def get_agent_app():
    """获取（或懒初始化）Agent 应用实例"""
    global _agent_app
    if _agent_app is None:
        async with _agent_lock:
            # 双重检查（防止并发重复初始化）
            if _agent_app is None:
                _agent_app = build_agent_graph()
    return _agent_app


# ══════════════════════════════════════════════════════╗
#  API 端点
# ══════════════════════════════════════════════════════╝


@app.get("/health")
async def health_check():
    """健康检查 —— 负载均衡器轮询用"""
    return {"status": "ok", "service": "patent-assistant", "version": "1.0.0"}


@app.post("/chat", response_model=None)
async def chat(request: ChatRequest):
    """
    发送消息给 Agent，获取回复。

    两种模式:
      - stream=True（默认）: 返回 SSE 流式响应
      - stream=False        : 返回完整 JSON 响应
    """
    import uuid

    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"

    config = {"configurable": {"thread_id": thread_id}}

    app = await get_agent_app()

    if request.stream:
        # ── SSE 流式输出 ─────────────────────
        return StreamingResponse(
            _stream_agent_response(app, config, request.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Thread-ID": thread_id,
            },
        )
    else:
        # ── 非流式（完整返回）─────────────────
        result = await app.ainvoke(
            {
                "messages": [{"role": "user", "content": request.message}],
                "query": request.message,
                "intent": "unknown",
                "sources": [],
                "needs_approval": False,
                "final_answer": "",
            },
            config=config,
        )

        # 提取最后一条 AI 消息
        msgs = result.get("messages", [])
        last_ai = next((m for m in reversed(msgs) if m.type == "ai"), None)
        response_text = last_ai.content if last_ai else "(无回复)"

        return ChatResponse(
            thread_id=thread_id,
            response=response_text,
            message_count=len(msgs),
        )


async def _stream_agent_response(
    app,
    config: dict,
    message: str,
) -> AsyncGenerator[str, None]:
    """
    生成 SSE 格式的流式响应。

    SSE (Server-Sent Events) 格式:
      data: {JSON}\n\n
    """
    try:
        async for chunk in app.astream(
            {
                "messages": [{"role": "user", "content": message}],
                "query": message,
                "intent": "unknown",
                "sources": [],
                "needs_approval": False,
                "final_answer": "",
            },
            config=config,
            stream_mode="messages",   # 只流式输出消息 chunk
        ):
            # chunk 格式: ([消息对象], metadata)
            msg_obj = chunk[0] if isinstance(chunk, tuple) else chunk
            content = getattr(msg_obj, "content", str(msg_obj))

            if content:
                # SSE 格式封装
                payload = json.dumps({"delta": content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        # 流结束标记
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_payload}\n\n"


# ══════════════════════════════════════════════════════╗
#  启动入口
# ══════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import uvicorn

    print("╔═══════════════════════════════════════════╗")
    print("║   专利流程智能助手 — FastAPI 服务        ║")
    print("╚═══════════════════════════════════════════╝")
    print(f"\n  接口文档: http://localhost:8000/docs")
    print(f"  健康检查: http://localhost:8000/health")
    print(f"  聊天端点: POST http://localhost:8000/chat\n")

    uvicorn.run(
        "patent_assistant.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
