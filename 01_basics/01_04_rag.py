"""
================================================================
  1.4 RAG (检索增强生成) — 让 AI 能"读"你的文档
================================================================

【学习目标】
  - 理解 RAG 的原理：检索 + 生成 = 更准确、更可控的回答
  - 掌握 Embedding（文本向量化）的概念和用法
  - 学会使用 ChromaDB 作为向量数据库存储文档
  - 构建一个完整的"本地文档问答"系统

【前置知识】
  - 01_01_model_io.py（模型调用）
  - 01_02_lcel_chains.py（LCEL 链式调用）

【核心概念图解】

  没有 RAG 的 LLM:
    用户: "我们公司的专利申请流程是什么？"
    LLM : "我无法回答这个问题..."  ← 训练数据里没有你的公司信息

  有 RAG 的系统:
    用户: "专利申请流程是什么？"
     ┌─────────────────────────────────────┐
     │           RAG 系统工作流             │
     │                                     │
     │  ① 检索 (Retrieval)                 │
     │     问题 → Embedding → 向量搜索      │
     │                    ↓               │
     │     找到最相关的 3 个文档片段        │
     │                                     │
     │  ② 增强 (Augmentation)              │
     │     把找到的片段 + 用户问题          │
     │     组合成新的提示词                  │
     │                                     │
     │  ③ 生成 (Generation)                │
     │     LLM 基于检索到的内容回答         │
     │                                     │
     └─────────────────────────────────────┘

  类比：
    RAG 就像开卷考试——你可以翻阅资料（检索），
    然后基于资料回答问题（生成），
    而不是靠死记硬背。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHROMA_PERSIST_DIR


# ============================================================
# Part 1: 准备示例文档数据
# ============================================================

SAMPLE_DOCUMENTS = [
    """
    【专利申请流程概述】

    第一步：技术交底书撰写
    发明人需要提供完整的技术交底书，包括：技术领域、背景技术、发明内容、
    具体实施方式、附图说明等。交底书的详细程度直接影响后续申请质量。

    第二步：专利检索与分析
    在提交前进行现有技术检索，评估新颖性和创造性。推荐使用：
    国家知识产权局官网、Google Patents、Soopat 等平台。

    第三步：申请文件撰写
    由专利代理师根据技术交底书撰写正式申请书，包括：
    权利要求书、说明书、摘要、附图等。这是最关键的环节之一。

    第四步：提交与审查
    通过电子申请系统(CPC) 向国知局提交，之后进入审查阶段：
    初步审查(形式审查) → 实质审查(针对发明专利) → 授权/驳回
    """,

    """
    【各类型专利对比】

    ┌──────────┬──────────┬──────────┬──────────┐
    │   类型   │ 保护期限 │ 审查周期 │ 申请费用  │
    ├──────────┼──────────┼──────────┼──────────┤
    │ 发明专利 │ 20 年    │18-24月  │约3400元  │
    │ 实用新型 │ 10 年    │6-8月    │约500元   │
    │ 外观设计 │ 15 年    │4-6月    │约500元   │
    └──────────┴──────────┴──────────┴──────────┘

    注意事项:
    - 发明专利需要经过实质审查，授权难度最高但保护力最强
    - 实用新型不经过实审，授权快但保护力度较弱
    - 外观设计只保护产品的外观，不保护功能
    """,

    """
    【常见驳回原因及应对策略】

    驳回原因 1: 新颖性不足
    应对策略: 提交前做充分的现有技术检索，确保技术方案有创新点。
    在权利要求书中突出区别特征。

    驳回原因 2: 创造性不够（"显而易见"）
    应对策略: 提供技术效果证据，说明方案解决了什么技术问题、
    取得了什么意想不到的效果。

    驳回原因 3: 说明书公开不充分
    应对策略: 具体实施方式要写得足够详细，让本领域技术人员能复现。
    必要时补充实验数据和对比案例。

    驳回原因 4: 权利要求范围过宽或过窄
    应对策略: 合理布局权利要求的层次，从宽到窄逐步递进。
    """,
]


def demo_embedding_basics():
    """
    【示例 1-4a】Embedding 基础 —— 文本变数字

    Embedding 是把文字变成一组数字（向量）的过程。
    相似的文字 → 数字也相近 → 可以用数学方法比较相似度。

    例如:
    "猫很可爱"  → [0.12, -0.34, 0.56, ...]  （512个数字）
    "小狗很萌"  → [0.13, -0.32, 0.54, ...]  （数字接近 ↑）
    "股票跌了"  → [0.87,  0.45, -0.11, ...]  （数字差很远 ↓）
    """
    print("\n" + "=" * 60)
    print("  示例 1-4a: Embedding 文本向量化")
    print("=" * 60)

    from langchain_ollama import OllamaEmbeddings

    # 使用 Ollama 本地的 embedding 模型
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text",   # 专门做文本嵌入的小模型
    )

    # 把两段相似的文字变成向量
    text_a = "专利申请流程有哪些步骤"
    text_b = "如何提交专利申请"

    vec_a = embeddings.embed_query(text_a)
    vec_b = embeddings.embed_query(text_b)

    print(f"\n文本A: '{text_a}'")
    print(f"  向量维度: {len(vec_a)}")
    print(f"  前5个值: {[round(x, 4) for x in vec_a[:5]]}")

    print(f"\n文本B: '{text_b}'")
    print(f"  前5个值: {[round(x, 4) for x in vec_b[:5]]}")

    # 用余弦相似度计算两段文字的相似程度
    import math
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a**2 for a in vec_a))
    norm_b = math.sqrt(sum(b**2 for b in vec_b))
    similarity = dot_product / (norm_a * norm_b)

    print(f"\n余弦相似度: {similarity:.4f}")
    print("(越接近1表示越相似)")


def demo_vector_store():
    """
    【示例 1-4b】向量存储 —— ChromaDB 入门

    ChromaDB 是一个轻量级的向量数据库，专门用来存 Embedding 向量。
    它可以快速找出"最相似的 N 个文档"——这就是 RAG 的核心能力。

    工作流程:
    1. 文档 → 分块 → Embedding → 存入 ChromaDB
    2. 查询时：查询文本 → Embedding → ChromaDB 找最相似的文档
    """
    print("\n" + "=" * 60)
    print("  示例 1-4b: ChromaDB 向量存储")
    print("=" * 60)

    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    # ---- Step 1: 初始化 Embedding 和 Vector Store ----
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text",
    )

    # 创建持久化的 ChromaDB 实例
    # persist_directory: 数据保存位置（重启不丢失）
    vectorstore = Chroma(
        collection_name="patent_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR / "rag_demo"),
    )

    # ---- Step 2: 准备文档并添加 ----
    documents = []
    for i, doc_text in enumerate(SAMPLE_DOCUMENTS):
        # Document 对象包含 page_content（正文）和 metadata（元数据）
        doc = Document(
            page_content=doc_text.strip(),
            metadata={"source": f"doc_{i+1}.txt", "index": i},
        )
        documents.append(doc)

    # 添加到向量库（会自动执行 Embedding）
    ids = vectorstore.add_documents(documents=documents)
    print(f"\n已添加 {len(ids)} 个文档到向量库")

    # ---- Step 3: 检索测试 ----
    test_queries = [
        "发明专利要审查多久？",
        "怎么避免被驳回？",
        "申请费用大概是多少？",
    ]

    for query in test_queries:
        print(f"\n{'─' * 40}")
        print(f"[查询] {query}")

        # similarity_search: 找出最相似的 2 个文档片段
        results = vectorstore.similarity_search(query, k=2)
        for j, result in enumerate(results, 1):
            preview = result.page_content[:100].replace('\n', ' ')
            print(f"  结果{j} [{result.metadata['source']}]: {preview}...")


def demo_rag_chain():
    """
    【示例 1-4c】完整 RAG 链 —— 检索 + 生成

    把前面的知识串起来：
    1. 用户提问
    2. 从向量库里找相关文档（检索）
    3. 把文档 + 问题一起发给 LLM（增强 + 生成）

    这就是生产级 RAG 系统的核心模式！
    """
    print("\n" + "=" * 60)
    print("  示例 1-4c: 完整 RAG 链")
    print("=" * 60)

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough

    # ---- Step 1: 准备向量库 ----
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")

    vectorstore = Chroma(
        collection_name="patent_rag",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR / "rag_full"),
    )

    # 清空旧数据，重新添加
    try:
        vectorstore.reset_collection()
    except Exception:
        pass

    docs = [
        Document(page_content=t.strip(), metadata={"source": f"doc_{i}"})
        for i, t in enumerate(SAMPLE_DOCUMENTS, 1)
    ]
    vectorstore.add_documents(docs)

    # 创建检索器（retriever 是一个可调用的对象）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # ---- Step 2: 构建 RAG Prompt ----
    rag_prompt = ChatPromptTemplate.from_template(
        "你是一个专业的专利问答助手。\n\n"
        "请根据以下【参考资料】来回答用户的【问题】。\n"
        "如果参考资料中没有相关信息，请明确说明。\n\n"
        "=== 参考资料 ===\n"
        "{context}\n\n"
        "=== 用户问题 ===\n"
        "{question}\n\n"
        "=== 回答 ===\n"
    )

    # ---- Step 3: 构建 LLM ----
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
    )

    # ---- Step 4: 组装 RAG Chain ----
    #
    # 这里的技巧是 RunnablePassthrough()：
    # 它原封不动地把输入传递下去，
    # 让 question 字段能同时被 retriever 和 prompt 使用
    #
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # ---- Step 5: 测试 RAG 系统 ----
    questions = [
        "申请一个发明专利大概要花多少钱？",
        "如果我的专利被驳回了怎么办？",
        "外观设计和实用新型有什么区别？",
    ]

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"[用户] {q}")
        print(f"[AI]", end="", flush=True)
        answer = rag_chain.invoke(q)
        print(answer)


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  LangChain v1.x — 1.4 RAG 教程              ║")
    print("║  检索增强生成                                ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_embedding_basics()
        demo_vector_store()
        demo_rag_chain()

        print("\n✅ 全部示例运行完毕！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n提示:")
        print("  1. 确保 Ollama 已运行: ollama list")
        print("  2. 确保 nomic-embed-text 模型已下载: ollama pull nomic-embed-text")
