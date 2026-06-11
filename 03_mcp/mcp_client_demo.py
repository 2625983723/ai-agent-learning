"""
================================================================
  3.2 FastMCP Client 开发 — 连接并调用 MCP Server
================================================================

【学习目标】
  - 掌握 FastMCP Client 的基本用法
  - 学会连接 MCP Server（stdio 模式）
  - 掌握工具的列表、调用、参数传递

【前置知识】
  - 03_mcp_server_demo.py（MCP Server 开发）

【核心概念图解】

  ┌──────────────────────────────────────┐
  │            MCP Client              │
  │                                      │
  │  client = Client("server.py")      │
  │       ↓                              │
  │  await client.list_tools()         │ ← 列出所有可用工具
  │       ↓                              │
  │  await client.call_tool(           │ ← 调用工具
  │      "add", {"a": 1, "b": 2}   │
  │  )                                │
  │       ↓                              │
  │  拿到结果                          │
  └──────────────────────────────────────┘

  两种连接方式:
  1. 本地文件: Client("mcp_server_demo.py")
                → 自动启动子进程（stdio 通信）
  2. 远程 URL : Client("http://localhost:8000/mcp")
                → 通过 HTTP 连接（SSE/StreamableHTTP）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Part 1: 基础 Client 示例代码（打印讲解）
# ============================================================


def demo_client_code_explained():
    """
    【示例 3-2a】Client 代码一步步讲解

    完整的 Client 代码只需 20 行左右！
    """
    print("\n" + "=" * 60)
    print("  示例 3-2a: Client 代码详解")
    print("=" * 60)

    code = '''
"""MCP Client 示例 — 一步步讲解"""
import asyncio
from fastmcp import Client

# ──── Step 1: 创建 Client ──────────────────
# 传入 Server 文件路径，Client 会自动启动子进程
client = Client("mcp_server_demo.py")

# ──── Step 2: 连接 Server ─────────────────
# 使用 async with 语法（自动管理连接生命周期）
async def main():
    async with client:
        # ──── Step 3: 列出所有工具 ───────
        tools = await client.list_tools()
        print(f"Server 提供了 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:40]}...")

        # ──── Step 4: 调用工具 ─────────────
        result = await client.call_tool(
            "query_patent_type",           # 工具名称
            {"patent_type": "发明"}      # 参数字典
        )
        print(f"\\n调用结果: {result}")

# ──── Step 5: 运行异步函数 ─────────────
asyncio.run(main())
'''

    # 逐行打印，带行号
    for i, line in enumerate(code.strip().splitlines(), 1):
        # 跳过空行
        if line.strip() == "":
            print()
        else:
            print(f"  {line}")


def demo_transport_comparison():
    """
    【示例 3-2b】不同传输方式的 Client 连接方式

    Client 的构造函数参数决定了连接方式。
    """
    print("\n" + "=" * 60)
    print("  示例 3-2b: 不同传输方式的连接")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────┐
│  传输方式          Client 初始化方式               │
├─────────────────────────────────────────────────────┤
│  stdio（默认）    Client("server.py")           │
│  本地文件，自动启动子进程                         │
│                                                    │
│  SSE 远程         Client("http://localhost:8000/ │
│                    sse")                          │
│  连接远程 HTTP 服务（SSE 模式）                  │
│                                                    │
│  StreamableHTTP   Client("http://localhost:8000/ │
│                    mcp")                          │
│  连接远程 HTTP 服务（新标准）                     │
└─────────────────────────────────────────────────────┘

代码示例:

# 方式1: stdio（本地）
client = Client("mcp_patent_server.py")

# 方式2: SSE 远程
client = Client("http://localhost:8000/sse")

# 方式3: StreamableHTTP 远程
client = Client("http://localhost:8000/mcp")
""")

    print("  💡 本地开发推荐用 stdio（最简单）")
    print("     生产部署推荐用 StreamableHTTP（最灵活）")


# ============================================================
# Part 2: 实际运行的 Client 代码
# ============================================================


def get_patent_server_path() -> str:
    """获取专利 Server 文件路径（用于 Client 连接）"""
    server_file = Path(__file__).parent / "mcp_patent_server.py"
    return str(server_file)


def demo_sync_wrapper():
    """
    【示例 3-2c】同步包装器 —— 让同步代码也能用 async Client

    很多场景下你不想写 async/await，
    可以用 asyncio.run() 包装一下。
    """
    print("\n" + "=" * 60)
    print("  示例 3-2c: 同步包装器")
    print("=" * 60)

    wrapper_code = '''
"""同步包装：在同步函数中调用异步 Client"""

def call_mcp_tool_sync(server_path: str, tool_name: str, **kwargs):
    """同步调用 MCP 工具的包装函数"""
    import asyncio
    from fastmcp import Client

    async def _call():
        async with Client(server_path) as client:
            result = await client.call_tool(tool_name, kwargs)
            return result

    return asyncio.run(_call())


# 使用方式（同步！）
if __name__ == "__main__":
    result = call_mcp_tool_sync(
        "mcp_patent_server.py",
        "query_patent_type",
        patent_type="发明"
    )
    print(result)
'''

    print(wrapper_code)

    print("\n  ✅ 有了这个包装器，你可以在任何同步代码中调用 MCP！")
    print("     特别适合集成到 Flask/FastAPI 等 Web 框架中。")


def demo_error_handling():
    """
    【示例 3-2d】错误处理 —— 健壮的 Client

    MCP 调用可能失败的原因：
    1. Server 文件不存在
    2. 工具名称错误
    3. 参数类型不匹配
    4. Server 内部异常
    """
    print("\n" + "=" * 60)
    print("  示例 3-2d: 错误处理")
    print("=" * 60)

    robust_code = '''
"""健壮的 MCP Client — 带完整错误处理"""

import asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError, ConnectionError

async def safe_call_tool(server_path: str, tool: str, params: dict):
    """带错误处理的工具调用"""
    try:
        async with Client(server_path) as client:
            result = await client.call_tool(tool, params)
            return {"success": True, "data": result}

    except FileNotFoundError:
        return {"success": False, "error": f"Server 文件不存在: {server_path}"}

    except ToolError as e:
        return {"success": False, "error": f"工具调用失败: {e}"}

    except ConnectionError as e:
        return {"success": False, "error": f"连接失败: {e}"}

    except Exception as e:
        return {"success": False, "error": f"未知错误: {type(e).__name__}: {e}"}


# 使用
result = asyncio.run(safe_call_tool(
    "mcp_patent_server.py",
    "query_patent_type",
    {"patent_type": "发明"}
))
print(result)
'''

    for line in robust_code.strip().splitlines():
        print(f"  {line}")

    print("\n  🛡️ 生产环境中一定要加错误处理！")


# ============================================================
# Part 3: 保存可运行的 Client 文件
# ============================================================


def save_sample_client_file():
    """保存一个可运行的示例 Client 文件"""
    client_code = '''"""
MCP Client 示例 — 可运行
连接 mcp_patent_server.py 并调用工具
"""
import asyncio
from fastmcp import Client


async def main():
    # 连接到本地 Server（stdio 模式）
    # 注意：Server 文件需要在同一目录下
    server_path = "mcp_patent_server.py"

    print("🔗 正在连接 MCP Server...")
    async with Client(server_path) as client:
        print("✅ 连接成功！")

        # 列出所有工具
        tools = await client.list_tools()
        print(f"\\n📋 Server 提供了 {len(tools)} 个工具:")
        for t in tools:
            print(f"  - {t.name}")

        # 调用工具 1
        print("\\n" + "-" * 40)
        print("▶ 调用: query_patent_type('发明')")
        r1 = await client.call_tool(
            "query_patent_type",
            {"patent_type": "发明"}
        )
        print(f"  结果: {r1}")

        # 调用工具 2
        print("\\n▶ 调用: estimate_cost('实用新型', use_agency=True)")
        r2 = await client.call_tool(
            "estimate_cost",
            {"patent_type": "实用新型", "use_agency": True}
        )
        print(f"  结果: {r2}")


if __name__ == "__main__":
    asyncio.run(main())
'''

    output_path = Path(__file__).parent / "mcp_client_example.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(client_code)

    print(f"\\n✅ 已保存可运行文件: {output_path}")
    print(f"   运行前请确保 mcp_patent_server.py 在同一目录")
    print(f"   运行命令: python {output_path.name}")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  MCP 协议 — 3.2 FastMCP Client 教程       ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_client_code_explained()
        demo_transport_comparison()
        demo_sync_wrapper()
        demo_error_handling()
        save_sample_client_file()

        print("\\n✅ 全部示例讲解完毕！")
        print("\\n💡 下一步: 运行 python mcp_client_example.py 测试连接")
        print("   （需要先启动 mcp_patent_server.py 或在同目录）")

    except Exception as e:
        print(f"\\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
