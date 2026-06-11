"""
================================================================
  3.1 FastMCP Server 开发 — 写下你的第一个 MCP Server
================================================================

【学习目标】
  - 理解 MCP Server 的三大核心能力：Tools / Resources / Prompts
  - 掌握 FastMCP 的完整开发流程
  - 学会用 @mcp.tool / @mcp.resource / @mcp.prompt 装饰器

【前置知识】
  - Python 基础（函数、装饰器、类型注解）
  - 了解 HTTP 基本概念

【核心概念图解】

  MCP Server = 对外提供"能力"的程序

  ┌──────────────────────────────────────────────┐
  │              FastMCP Server                │
  │                                              │
  │  @mcp.tool         工具（可执行的函数）     │
  │  def add(a,b) → 注册为"加法工具"      │
  │                                              │
  │  @mcp.resource    资源（可读取的数据）     │
  │  def get_file() → 注册为"文件资源"      │
  │                                              │
  │  @mcp.prompt     提示词模板                │
  │  def summarize() → 注册为"总结提示词"   │
  │                                              │
  │  三种传输方式: stdio / SSE / StreamableHTTP │
  └──────────────────────────────────────────────┘

  客户端（Agent）通过 MCP 协议调用这些能力：
    client.call_tool("add", {"a": 1, "b": 2}) → 3
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Part 1: 最简单的 MCP Server
# ============================================================


def demo_minimal_server_code():
    """
    【示例 3-1a】最小化的 MCP Server 代码

    核心只有 3 行：
      1. 创建 FastMCP 实例
      2. 用 @mcp.tool 定义工具
      3. 调用 mcp.run()
    """
    print("\n" + "=" * 60)
    print("  示例 3-1a: 最小 MCP Server 代码解读")
    print("=" * 60)

    minimal_code = '''
"""最小 MCP Server"""
from fastmcp import FastMCP

# 1. 创建 Server 实例（第一个参数是名称）
mcp = FastMCP("My First Server")

# 2. 用 @mcp.tool 装饰器注册工具
@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"两个数字相加\"\"\"
    return a + b

# 3. 启动 Server（默认用 stdio 传输）
if __name__ == "__main__":
    mcp.run()
'''

    for i, line in enumerate(minimal_code.strip().splitlines(), 1):
        print(f"  {line}")

    print(f"\n  💡 运行方式:")
    print(f"     python mcp_server_demo.py          # stdio 模式（默认）")
    print(f"     mcp dev mcp_server_demo.py        # 用 Inspector 调试")
    print(f"     python mcp_server_demo.py --transport sse  # SSE 模式")


def demo_patent_server():
    """
    【示例 3-1b】专利信息服务 MCP Server

    实际可用的完整 Server，包含：
      - 3 个 Tool（工具）
      - 1 个 Resource（资源）
      - 1 个 Prompt（提示词模板）
    """
    print("\n" + "=" * 60)
    print("  示例 3-1b: 专利信息服务 MCP Server")
    print("=" * 60)

    full_code = '''
"""专利信息服务 MCP Server — 完整示例"""
from fastmcp import FastMCP
from typing import Literal

# 创建 Server（名称随意）
mcp = FastMCP("PatentInfoServer")

# =====================================================
# Part 1: Tools (工具) — Agent 可以调用的函数
# =====================================================

@mcp.tool()
def query_patent_type(patent_type: Literal["发明", "实用新型", "外观"]) -> dict:
    \"\"\"查询指定专利类型的信息

    Args:
        patent_type: 专利类型（发明/实用新型/外观）
    \"\"\"
    db = {
        "发明": {"期限": "20年", "费用": "~3400元", "周期": "18-24个月"},
        "实用新型": {"期限": "10年", "费用": "~500元", "周期": "6-8个月"},
        "外观": {"期限": "15年", "费用": "~500元", "周期": "4-6个月"},
    }
    return db.get(patent_type, {"error": "未知类型"})


@mcp.tool()
def estimate_total_cost(
    patent_type: Literal["发明", "实用新型", "外观"],
    use_agency: bool = True
) -> dict:
    \"\"\"估算专利申请总费用\"\"\"
    base = {"发明": 3400, "实用新型": 500, "外观": 500}
    agency = 2000 if use_agency else 0
    return {
        "官费": base.get(patent_type, 0),
        "代理费": agency,
        "合计": base.get(patent_type, 0) + agency,
    }


@mcp.tool()
def check_patent_timeline(patent_type: str, stage: str = "提交后") -> str:
    \"\"\"查询专利时间线\"\"\"
    return f"[{patent_type}] {stage}: 预计需要 6-24 个月"


# =====================================================
# Part 2: Resources (资源) — Agent 可以读取的数据
# =====================================================

@mcp.resource("patent://fee-schedule")
def get_fee_schedule() -> str:
    \"\"\"获取完整费用表（资源）\"\"\"
    return """中国专利费用表 (2026 年标准）
==================================
类型          官费       代理费      合计
----------------------------------
发明专利      900+2500   ~2000      ~5400
实用新型      500         ~2000      ~2500
外观设计      500         ~2000      ~2500
==================================
注: 费用可能因政策调整而变化，以国知局官网为准。
"""


# =====================================================
# Part 3: Prompts (提示词) — 供 Agent 使用的模板
# =====================================================

@mcp.prompt()
def patent_consultation_prompt(user_query: str) -> str:
    \"\"\"专利咨询提示词模板\"\"\"
    return f"""你是一个专业的中国专利咨询助手。

用户问题: {user_query}

请依次使用以下工具获取准确信息：
1. query_patent_type — 查询专利类型信息
2. estimate_total_cost — 估算费用
3. check_patent_timeline — 查询时间线

基于工具返回的结果，用简洁的中文回答用户。
"""


# =====================================================
# 启动 Server
# =====================================================
if __name__ == "__main__":
    # 默认 stdio 模式（本地进程通信）
    # 改为 SSE: mcp.run(transport="sse", host="0.0.0.0", port=8000)
    mcp.run()
'''

    # 打印完整代码（格式化的关键部分）
    print(f"\n  完整代码概览（核心部分）:\n")
    lines = full_code.strip().splitlines()
    for i, line in enumerate(lines[:40], 1):
        print(f"  {line}")
    print(f"  ...")
    print(f"\n  📋 完整代码已保存在这个文件的注释中，")
    print(f"     或者运行本文件查看: python {Path(__file__).name}")


def demo_transport_modes():
    """
    【示例 3-1c】三种传输方式对比

    Transport（传输层）决定了 Client 和 Server 如何通信。
    """
    print("\n" + "=" * 60)
    print("  示例 3-1c: 三种传输方式")
    print("=" * 60)

    print(f"\n  ┌────────────┬──────────────────────────────────────┐")
    print(f"  │ 传输方式    │ 使用场景                         │")
    print(f"  ├────────────┼──────────────────────────────────────┤")
    print(f"  │ stdio      │ 本地进程间通信（默认）             │")
    print(f"  │            │ Client 和 Server 在同一台机器       │")
    print(f"  │            │ 例如: Agent 进程启动 Server 子进程  │")
    print(f"  ├────────────┼──────────────────────────────────────┤")
    print(f"  │ SSE        │ 通过 HTTP Server-Sent Events      │")
    print(f"  │            │ 远程访问，Server 是 HTTP 服务      │")
    print(f"  │            │ 例如: 云端部署的 MCP Server       │")
    print(f"  ├────────────┼──────────────────────────────────────┤")
    print(f"  │ Streamable  │ 新标准（2025+），替代 SSE       │")
    print(f"  │ HTTP       │ 支持流式输出，更灵活              │")
    print(f"  └────────────┴──────────────────────────────────────┘")

    print(f"\n  💡 启动不同模式的命令:")
    print(f"     # stdio（默认）")
    print(f"     python mcp_server_demo.py")
    print(f"\n     # SSE 模式（监听 0.0.0.0:8000）")
    print(f"     python mcp_server_demo.py --transport sse --port 8000")
    print(f"\n     # Streamable HTTP 模式")
    print(f"     mcp run mcp_server_demo.py --transport streamable-http")


def demo_inspector_debugging():
    """
    【示例 3-1d】使用 MCP Inspector 调试

    MCP Inspector 是官方提供的调试工具，
    让你在浏览器中可视化的测试 MCP Server。
    """
    print("\n" + "=" * 60)
    print("  示例 3-1d: MCP Inspector 调试")
    print("=" * 60)

    print(f"\n  第 1 步: 启动 Inspector")
    print(f"    命令: mcp dev mcp_server_demo.py")
    print(f"    输出: Inspector URL (通常是 http://localhost:5173)\n")

    print(f"  第 2 步: 在浏览器中打开 Inspector")
    print(f"    - 左侧: 列出所有 Tools / Resources / Prompts")
    print(f"    - 中间: 调用工具的参数输入区")
    print(f"    - 右侧: 显示调用结果\n")

    print(f"  第 3 步: 测试工具调用")
    print(f"    1. 点击 'query_patent_type'")
    print(f"    2. 填入参数: {{\"patent_type\": \"发明\"}}")
    print(f"    3. 点击 'Call Tool'")
    print(f"    4. 查看右侧返回结果\n")

    print(f"  ✅ Inspector 是开发 MCP Server 的必备工具！")


# ============================================================
# 实际可运行的 Server 代码（写入单独文件）
# ============================================================

PATENT_SERVER_CODE = '''
"""专利信息服务 MCP Server — 可直接运行"""
from fastmcp import FastMCP
from typing import Literal

mcp = FastMCP("PatentInfoServer")


@mcp.tool()
def query_patent_type(patent_type: Literal["发明", "实用新型", "外观"]) -> dict:
    """查询指定专利类型的信息"""
    db = {
        "发明": {"保护期限": "20年", "官费": "900+2500元", "周期": "18-24个月"},
        "实用新型": {"保护期限": "10年", "官费": "500元", "周期": "6-8个月"},
        "外观": {"保护期限": "15年", "官费": "500元", "周期": "4-6个月"},
    }
    return db.get(patent_type, {"error": "未知类型"})


@mcp.tool()
def estimate_cost(patent_type: str, use_agency: bool = True) -> dict:
    """估算专利申请费用"""
    base = {"发明": 3400, "实用新型": 500, "外观": 500}
    agency = 2000 if use_agency else 0
    return {"官费": base.get(patent_type, 0), "代理费": agency,
            "合计": base.get(patent_type, 0) + agency}


@mcp.resource("patent://guide")
def patent_guide() -> str:
    """专利申请指南资源"""
    return "专利申请三步骤:\\n1. 撰写技术交底书\\n2. 检索现有技术\\n3. 提交申请文件"


@mcp.prompt()
def consultation_prompt(user_query: str) -> str:
    """专利咨询提示词"""
    return f"用户问题: {user_query}\\n请使用工具查询准确信息后回答。"


if __name__ == "__main__":
    mcp.run()
'''


def save_patent_server_file():
    """把专利 Server 代码保存为独立文件"""
    output_path = Path(__file__).parent / "mcp_patent_server.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(PATENT_SERVER_CODE))
    print(f"\n  ✅ 已保存到: {output_path}")
    print(f"     运行: python {output_path.name}")
    print(f"     调试: mcp dev {output_path.name}")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════╗")
    print("║  MCP 协议 — 3.1 FastMCP Server 教程        ║")
    print("╚════════════════════════════════════════════╝")

    try:
        demo_minimal_server_code()
        demo_patent_server()
        demo_transport_modes()
        demo_inspector_debugging()

        # 保存可直接运行的 Server 文件
        save_patent_server_file()

        print("\n✅ 全部示例运行完毕！")
        print("\n💡 下一步: 运行 python mcp_client_demo.py 学习如何调用 Server")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
