"""
================================================================
  1.2 LCEL 链式调用 — LangChain Expression Language
================================================================

【学习目标】
  - 理解 LCEL（LangChain Expression Language）的核心概念
  - 掌握管道符 `|` 的用法：把多个组件串联成一条"流水线"
  - 学会 RunnableParallel 并行执行
  - 学会 RunnablePassthrough 传递数据

【前置知识】
  - 已掌握 01_01_model_io.py 的内容（LLM 基础调用）

【核心概念图解】

  传统写法（v0.x 旧版）:
    result = model.invoke(prompt)
    output = parser.parse(result)

  LCEL 写法 (v1.x 新版):
    chain = prompt | model | parser    ← 用 | 把组件串起来
    chain.invoke({"topic": "Python"})   ← 一行搞定！

  ┌────────┐    │    ┌──────────┐    │    ┌───────────┐
  │ Prompt │────▶│───▶│   LLM    │────▶│───▶│ Output    │
  │ 模板   │         │ (模型)   │         │ Parser    │
  └────────┘         └──────────┘         └───────────┘

  就像自来水管一样：
  数据从左边流入，经过每个环节处理，最终从右边流出。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def demo_simple_chain():
    """
    【示例 1-2a】最简单的链 —— Prompt → Model → 输出

    这是 LCEL 最核心的用法：用 `|` 符号把组件串起来。

    组件就像乐高积木，`|` 就是连接积木的接口。
    """
    print("\n" + "=" * 60)
    print("  示例 1-2a: 最简单的 LCEL 链")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # ---- Step 1: 创建提示词模板 ----
    # {topic} 是一个占位符，运行时会被实际值替换
    prompt = ChatPromptTemplate.from_template(
        "请用通俗易懂的语言解释一下{topic}，不超过3句话。"
    )

    # ---- Step 2: 创建模型 ----
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.5,
    )

    # ---- Step 3: 创建输出解析器 ----
    # StrOutputParser 只是把模型的回复提取为纯文本字符串
    parser = StrOutputParser()

    # ---- Step 4: 用管道符串起来！----
    # 这就是 LCEL 的核心语法：
    #   prompt → 给用户输入套上模板
    #   model  → 发送给大模型
    #   parser → 解析输出为纯文本
    chain = prompt | llm | parser

    # ---- Step 5: 调用链 ----
    # invoke() 的参数会自动填入模板中的 {topic}
    result = chain.invoke({"topic": "人工智能 Agent"})

    print(f"\n输入 topic = '人工智能 Agent'")
    print(f"\n输出:\n{result}")


def demo_chain_with_variables():
    """
    【示例 1-2b】带多个变量的链

    提示词模板可以包含多个占位符，
    调用时传入一个字典，键名对应占位符名称。
    """
    print("\n" + "=" * 60)
    print("  示例 1-2b: 多变量链")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 模板中有两个变量: {name} 和 {skill}
    prompt = ChatPromptTemplate.from_template(
        "你叫{name}，你擅长{skill}。"
        "请自我介绍一下，语气要活泼。"
    )

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    chain = prompt | llm | StrOutputParser()

    # 传入包含所有变量值的字典
    result = chain.invoke({
        "name": "小智",
        "skill": "专利流程自动化",
    })

    print(f"\n输出:\n{result}")


def demo_parallel_execution():
    """
    【示例 1-2c】并行执行 —— RunnableParallel

    有些任务可以同时做，不需要等待前一个做完再做下一个。

    比如：同时让 AI 写标题 + 写摘要 + 选标签，
    比一个一个做快很多。

    ┌──────────────────────────────────────┐
    │           RunnableParallel           │
    │                                      │
    │  输入 ──┬──▶ 任务A ──┐               │
    │        ├──▶ 任务B ──┼──▶ 合并输出     │
    │        └──▶ 任务C ──┘               │
    └──────────────────────────────────────┘
    """
    print("\n" + "=" * 60)
    print("  示例 1-2c: 并行执行")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

    # 定义三个并行任务，每个任务是一条独立的子链
    parallel = RunnableParallel(
        # 子任务 1: 生成标题（简洁版）
        title=ChatPromptTemplate.from_template(
            "为一篇关于{topic}的文章起一个吸引人的标题，不超过15字。"
        ) | llm,

        # 子任务 2: 写摘要（详细版）
        summary=ChatPromptTemplate.from_template(
            "用2句话概括{topic}的核心概念。"
        ) | llm,

        # 子任务 3: 直接传递原文（不做任何处理）
        original_topic=RunnablePassthrough(),
    )

    # 调用并行链
    result = parallel.invoke({"topic": "MCP 协议在 AI Agent 中的应用"})

    print(f"\n[输入] topic = 'MCP 协议在 AI Agent 中的应用'\n")
    for key, value in result.items():
        content = value.content if hasattr(value, 'content') else str(value)
        print(f"[{key.upper()}]\n{content}\n")


def demo_complex_chain():
    """
    【示例 1-2d】复杂组合链 —— 翻译 + 总结 + 打分

    这个例子展示如何把多条链组合成一个复杂的工作流：

  用户输入文本
       │
       ▼
  ┌──────────┐
  │ 翻译链    │ ──→ 英文翻译
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ 总结链    │ ──→ 一句话总结
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ 打分链    │ ──→ 评分 (1-10)
  └──────────┘
    """
    print("\n" + "=" * 60)
    print("  示例 1-2d: 复杂组合链 — 翻译+总结+打分")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

    text = (
        "LangGraph 是一个用于构建有状态、多参与者应用的框架，"
        "它基于图结构来定义应用的工作流和状态管理。"
    )

    # 构建完整的处理流水线
    pipeline = RunnableParallel(
        translation=(
            ChatPromptTemplate.from_template(
                "把以下中文翻译成流畅的英文:\n{text}"
            ) | llm
        ),
        summary=(
            ChatPromptTemplate.from_template(
                "用一句话总结以下内容:\n{text}"
            ) | llm
        ),
        score=(
            ChatPromptTemplate.from_template(
                "对以下内容的清晰度打分(1-10)，只返回数字:\n{text}"
            ) | llm
        ),
    ).invoke({"text": text})

    print(f"\n[原文]\n{text}\n")
    for key, val in pipeline.items():
        print(f"[{key}] {val.content if hasattr(val,'content') else val}\n")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangChain v1.x — 1.2 LCEL 链式调用教程     ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_simple_chain()
        demo_chain_with_variables()
        demo_parallel_execution()
        demo_complex_chain()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
