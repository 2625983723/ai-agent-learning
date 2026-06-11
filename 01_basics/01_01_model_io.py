"""
================================================================
  1.1 模型输入输出 (Model I/O) — LangChain v1.x 写法
================================================================

【学习目标】
  - 学会如何用 LangChain v1.x 调用大语言模型（LLM）
  - 理解消息类型：System / Human / AI Message
  - 掌握 invoke() 和 stream() 两种调用方式
  - 了解 Ollama 本地模型的使用方法

【前置知识】
  - Python 基础语法
  - 已安装 Ollama 并拉取了模型（如 qwen3:8b）

【运行方式】
  python 01_01_model_io.py

【核心概念】
  ┌─────────────────────────────────────────────────────┐
  │                    Model I/O                        │
  │                                                     │
  │   输入 (Input)          模型 (LLM)         输出      │
  │   ┌────────┐    →     ┌──────────→     ┌──────────┐ │
  │   │Messages│          │ Ollama/Qwen│     │ Response │ │
  │   └────────┘          └──────────┘     └──────────┘ │
  │                                                     │
  │   Messages 类型:                                    │
  │   - SystemMessage:  给 AI 的系统指令（角色/规则）    │
  │   - HumanMessage:   用户说的话                      │
  │   - AIMessage:      AI 的回复                       │
  └─────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，方便导入 config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def demo_basic_invoke():
    """
    【示例 1-1a】最基本的 LLM 调用 —— invoke() 同步调用

    这是最简单的用法：发一条消息，等模型回复。
    就像你跟人聊天一样，你说一句，他回一句。

    运行后你会看到：
      - 模型的完整回复文本
      - 回复的元数据（用了哪个模型、token 数量等）
    """
    print("\n" + "=" * 60)
    print("  示例 1-1a: 基础 invoke 调用")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    # ---- Step 1: 创建 LLM 实例 ----
    # ChatOllama 是专门用于对话场景的 LLM 封装
    # 它连接到你本地运行的 Ollama 服务
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.7,       # 温度：0=保守，1=有创意
    )

    # ---- Step 2: 构造用户消息 ----
    # HumanMessage 代表"人类说的话"
    message = HumanMessage(content="你好！请用一句话介绍你自己。")

    # ---- Step 3: 调用模型 ----
    # invoke() 是同步调用：发送请求 → 等待回复 → 返回结果
    response = llm.invoke([message])

    # ---- Step 4: 查看结果 ----
    print(f"\n[用户] 你好！请用一句话介绍你自己。")
    print(f"\n[AI] {response.content}")

    # response 对象还包含很多有用的元数据
    print(f"\n--- 元数据 ---")
    print(f"  响应类型: {type(response).__name__}")
    print(f"  Token 用量: {getattr(response, 'usage_metadata', 'N/A')}")
    print(f"  响应 ID : {response.id[:12]}..." if hasattr(response, 'id') else "")

    return response


def demo_multi_messages():
    """
    【示例 1-1b】多轮对话 —— System + Human + AI 消息组合

    真实场景中，我们通常需要：
      1. SystemMessage: 先告诉 AI 它的角色和规则
      2. HumanMessage: 用户的问题
      3. （可选）AIMessage: 之前的对话历史

    消息列表的顺序很重要！就像剧本一样，按顺序念台词。
    """
    print("\n" + "=" * 60)
    print("  示例 1-1b: 多轮对话（System + Human）")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import (
        SystemMessage,
        HumanMessage,
        AIMessage,
    )

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.5,
    )

    # 构造一个完整的多轮对话
    messages = [
        # 第 1 步：告诉 AI 它的角色（系统指令）
        SystemMessage(
            content="你是一个专业的专利代理助理。"
                     "你的回答要简洁专业，不超过 50 字。"
        ),

        # 第 2 步：用户的提问
        HumanMessage(
            content="什么是发明专利？它和实用新型有什么区别？"
        ),

        # 第 3 步：（可选）AI 之前的回复——模拟历史对话
        # AIMessage(
        #     content="发明专利保护的是技术方案的创新..."
        # ),

        # 第 4 步：用户的追问
        # HumanMessage(content="那申请发明专利需要多长时间呢？"),
    ]

    response = llm.invoke(messages)

    print(f"\n[System] 你是一个专业的专利代理助理。你的回答要简洁专业，不超过50字。")
    print(f"[User] 什么是发明专利？它和实用新型有什么区别？")
    print(f"\n[AI] {response.content}")


def demo_stream_output():
    """
    【示例 1-1c】流式输出 —— stream()

    大模型的响应可能很长，如果等全部生成完再显示，
    用户会等得很焦虑。

    stream() 让模型"边想边说"，一个字一个字地吐出来，
    就像 ChatGPT 的打字机效果一样。

    适用场景：
      - 需要实时展示给用户的交互界面
      - 生成长文本时避免超时
    """
    print("\n" + "=" * 60)
    print("  示例 1-1c: 流式输出 (stream)")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.7,
    )

    message = HumanMessage(
        content="请列举 3 个学习 Python 的好处，每个用一句话说明。"
    )

    print(f"\n[User] 请列举 3 个学习 Python 的好处...")
    print(f"[AI] ", end="", flush=True)

    # stream() 返回一个迭代器，每次返回一个 chunk（一小块）
    full_response = ""
    for chunk in llm.stream([message]):
        # chunk.content 是这一小段文字
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print(f"\n\n--- 流式输出完成，共收到 {len(full_response)} 字符 ---")


def demo_batch():
    """
    【示例 1-1d】批量调用 —— batch()

    当你需要同时问模型多个问题时，用 batch() 比
    多次 invoke() 更高效。

    类比：
      - invoke(): 一个一个地送快递
      - batch() : 一次性送一批快递
    """
    print("\n" + "=" * 60)
    print("  示例 1-1d: 批量调用 (batch)")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
    )

    # 准备 3 组不同的输入
    inputs = [
        [HumanMessage(content="用一句话解释什么是 API")],
        [HumanMessage(content="用一句话解释什么是数据库")],
        [HumanMessage(content="用一句话解释什么是容器化")],
    ]

    # batch() 一次性发送所有请求
    responses = llm.batch(inputs)

    questions = ["API", "数据库", "容器化"]
    for i, (resp, q) in enumerate(zip(responses, questions), 1):
        print(f"\n  问题{i} ({q}): {responses[i-1].content.strip()}")


# ============================================================
# 主入口 —— 按顺序运行所有示例
# ============================================================
if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangChain v1.x — 1.1 Model I/O 教程       ║")
    print("║  模型输入输出基础                             ║")
    print("╚════════════════════════════════════════════╝")
    print(f"\n使用模型: {OLLAMA_MODEL} ({OLLAMA_BASE_URL})")

    try:
        # 示例 1: 基础 invoke
        demo_basic_invoke()

        # 示例 2: 多轮对话
        demo_multi_messages()

        # 示例 3: 流式输出
        demo_stream_output()

        # 示例 4: 批量调用
        demo_batch()

        print("\n" + "=" * 60)
        print("  全部示例运行完毕！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n提示: 请确认 Ollama 已启动且模型已下载。")
        print("  检查命令: ollama list")
        print("  启动命令: ollama serve")
