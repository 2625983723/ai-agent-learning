"""
全局配置模块
统一管理模型、向量库等配置，所有模块共用。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# --- 项目根目录 ---
PROJECT_ROOT = Path(__file__).parent

# --- Ollama 配置 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# --- OpenAI 兼容 API（可选）---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Chroma 向量数据库 ---
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv(
    "CHROMA_PERSIST_DIR",
    "./data/chroma"
)


def get_llm_config() -> dict:
    """
    获取当前可用的 LLM 配置字典。

    优先使用 Ollama 本地模型（免费），如果没有则回退到 OpenAI 兼容 API。

    Returns:
        dict: 包含 provider, model, base_url 等信息的字典

    使用示例:
        >>> cfg = get_llm_config()
        >>> print(cfg["provider"])  # "ollama" 或 "openai"
    """
    # 优先用 Ollama（免费本地模型）
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
    }


if __name__ == "__main__":
    # 打印当前配置，方便调试
    cfg = get_llm_config()
    print("=" * 40)
    print("  AI Agent 学习工程 — 当前配置")
    print("=" * 40)
    print(f"  LLM 提供方 : {cfg['provider']}")
    print(f"  模型名称   : {cfg['model']}")
    if cfg.get("base_url"):
        print(f"  API 地址   : {cfg['base_url']}")
    print(f"  数据目录   : {CHROMA_PERSIST_DIR}")
    print("=" * 40)
