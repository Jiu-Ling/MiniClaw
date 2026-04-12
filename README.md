# MiniClaw

极简的 Multi-Agent 运行时，Powered by LangGraph。

> 单轮对话从 438K tokens 优化到 ~142K tokens（67% 降幅），支持 Subagent 动态分派、渐退记忆、Per-User 沙箱和端到端 Trace 可视化。

## 项目简介

MiniClaw 是基于 LangGraph 构建的智能体运行时，核心能力：

- **动态 Subagent 分派**：主 Agent 在 tool loop 中通过 `spawn_subagent` 按需拆分任务，多个 spawn 同一轮内并行执行（ThreadPoolExecutor），subagent 在 clean-room 上下文中独立运行
- **混合记忆检索**：FTS5 (BM25) + sqlite-vec (BGE-M3 1024 维) + RRF 融合，写入时摘要压缩，渐退式衰退（1d/7d/30d 三级）
- **Token 优化体系**：System Prompt 压缩 61%、Subagent 工具输出 2K 硬截断、Classify/Planner 输入截断、记忆注入 Budget 执行
- **Per-User 工作区沙箱**：`write_file` 工具限定用户沙箱目录，Shell 强制只读
- **端到端 Tracing**：统一 `trace_id`，per-tool span，graph 节点 state 快照，HTML 可视化（静态导出 + 实时 SSE 服务）
- **多渠道流式**：Telegram (edit_message 渐进式) + CLI (Rich 渲染)

## 架构概览

```
用户消息
    ↓
渠道层 (Telegram / CLI)
    ↓ receive()
MessageLoop (访问控制 → 命令拦截 → 附件处理 → user_id/sandbox 派生)
    ↓
LangGraph 状态图:
    ingest → classify (mini 模型) → load_context (混合记忆检索)
                                         ↓
                ┌── clarify ──────→ complete
                ├── simple ───────→ agent ──────→ complete
                └── planned ──→ planner → agent ──→ complete
                                         ↓ (任何节点 last_error)
                                    error_handler → complete

    agent 内部 (tool loop, max 16 rounds):
        provider.achat → [tool_calls?] → execute_tool_calls (并行)
            ├── tool.read_file / shell / web_search / memory_search / ...
            ├── tool.spawn_subagent → run_subagent (clean-room, 独立 tool loop)
            │       ├── subagent.tool_loop.round_N → provider.achat → tool.*
            │       └── 结果 >4K 自动写入 sandbox，返回摘要 + 路径
            └── tool.write_file (限定 user sandbox)
```

### 核心组件

| 组件 | 说明 |
|---|---|
| **意图分类** | Mini 模型路由 (clarify / simple / planned)，降级链：mini → main → 规则 |
| **Slim Planner** | LLM `submit_plan` 输出 advisory `subagent_briefs`；空 briefs = 直接执行 |
| **Subagent 分派** | `spawn_subagent` 工具，3 种预设角色 (researcher/executor/reviewer) + 自定义。同轮多 spawn 并行，递归深度 = 1 |
| **工具系统** | `worker_visible` 元数据控制 subagent 可见性。共享 `tool_loop.py` 执行层，per-tool `tool.<name>` span |
| **混合记忆** | FTS5 + sqlite-vec + RRF。写入时 `compress_memory_entry`（去表格/代码块/≤200 chars）。渐退：today→full, 1-7d→100ch, 7-30d→50ch, 30d+→删 |
| **Token 优化** | Capability 摘要仅名称（非 active）；subagent tool result 2K cap；assistant 4K truncation；spawn result persist+pointer；memory 4K hard cap |
| **流式输出** | Telegram EDIT 模式（send + edit_message_text，0.8s 节流）；CLI Rich 渲染 |
| **Workspace 沙箱** | `write_file` 仅限 `.miniclaw/users/<user_id>/`；shell 只读（禁 `>`, `|`, `;`, `&`, 写命令） |
| **Tracing** | 单轮单 trace_id（`resolve_turn_trace`）。graph 节点 state 快照 + round input/output + per-tool span。`tools/trace_view.py` HTML 可视化 |
| **命令系统** | `@command` 装饰器，渠道层拦截，零 Checkpoint 污染 |
| **访问控制** | 6 位配对码 → `miniclaw pair <code>` → 授权。CLI 免配对 |

## 技术栈

- **运行时**：Python 3.11+ / LangGraph / SQLite
- **记忆**：FTS5 + sqlite-vec (BGE-M3 via Ollama)
- **渠道**：python-telegram-bot / Rich CLI
- **模型**：OpenAI 兼容 API（通义千问 / Claude / GPT 等）
- **可视化**：`tools/trace_view.py`（纯 Python stdlib，零依赖）

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key、模型名称和 Base URL

# 初始化本地存储
miniclaw init

# 单轮对话
miniclaw chat "你好"

# 交互式 REPL（Rich Markdown 渲染）
miniclaw repl

# Telegram 机器人
miniclaw telegram polling
```

> `miniclaw` 通过 `pyproject.toml [project.scripts]` 注册，`uv sync` 后可直接使用。

## CLI 命令

### 运行时命令

| 命令 | 说明 |
|---|---|
| `miniclaw init` | 初始化本地存储 |
| `miniclaw chat "..."` | 单轮对话 |
| `miniclaw repl` | 交互式 REPL |
| `miniclaw telegram polling` | 启动 Telegram bot |
| `miniclaw pair <code>` | 授权远程渠道 |
| `miniclaw graph [--format mermaid\|ascii\|png]` | 导出 LangGraph 拓扑 |
| `miniclaw trace tail <path> [--no-follow]` | 实时查看 JSONL trace |

### 聊天命令（在对话中使用）

| 命令 | 说明 |
|---|---|
| `/help` | 显示可用命令 |
| `/status` | Checkpoint 状态 |
| `/model` | 模型配置 |
| `/clear` | 清除 Checkpoint |
| `/new` | 新会话 |
| `/stop` | 停止线程 |
| `/resume_run` | 从 Checkpoint 恢复 |
| `/retry` | 重试上轮 |

### Trace 可视化工具

```bash
# 静态 HTML 导出（可分享单文件）
python tools/trace_view.py ~/.miniclaw/traces/miniclaw.jsonl

# 实时服务模式（SSE 推送，浏览器自动刷新）
python tools/trace_view.py --serve --port 8787

# 示例数据
python tools/trace_view.py tools/sample_trace.jsonl
```

功能：折叠 Span 树、Flame Graph 时间轴、明暗主题切换、Metadata/Payload/Output 弹窗详情、搜索过滤。

## 环境配置

### 必填

```bash
MINICLAW_API_KEY=sk-your-key
MINICLAW_BASE_URL=https://api.openai.com/v1
MINICLAW_MODEL=gpt-4o
```

### 可选

```bash
# Telegram
MINICLAW_TELEGRAM_BOT_TOKEN=123:abc

# Mini 模型（意图分类）
MINICLAW_MINI_MODEL=qwen3.5-flash
MINICLAW_MINI_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Embedding（混合记忆检索）
MINICLAW_EMBEDDING_MODEL=bge-m3
MINICLAW_OLLAMA_BASE_URL=http://localhost:11434

# Tracing
MINICLAW_TRACE_MODE=local  # off / local / langsmith / both
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 图引擎 | LangGraph | 条件边、Checkpoint、状态管理 |
| Subagent 模型 | `spawn_subagent` 工具 + clean-room | 动态分派 > 静态 phase 编排；零上下文继承 |
| 记忆检索 | FTS5 + sqlite-vec + RRF | 零外部依赖，BM25 + 语义混合召回 |
| 记忆衰退 | 1d/7d/30d 三级压缩 | 老记忆越短 → 省 token + 降噪 |
| Token 优化 | P0-P5 系统性截断 | 438K → 142K (67%)，无质量损失 |
| 流式输出 | Telegram EDIT 模式 | 真实 API（send + edit），0.8s 节流 |
| 工具安全 | `worker_visible` + shell 只读 + sandbox | Subagent 不能 send/cron/spawn；文件写入限用户目录 |
| Tracing | 单 trace_id + per-tool span | 一轮对话一棵 trace 树，viewer 直接渲染 |
| Spawn 结果 | persist + pointer | >4K 写文件返路径，防 context 爆炸 |
| 渠道抽象 | Protocol + Meta | 声明式能力协商，自动降级 |
| Prompt 结构 | Static-first / Dynamic-last | `cache_control: ephemeral` 前缀命中 |

## Token 优化实测

基准：单轮 multi-agent 协作任务（6 subagent、8 轮主 agent、32 次 LLM 调用）

| 指标 | 优化前 | 优化后 | 降幅 |
|---|---|---|---|
| 总 prompt tokens | 413,521 | ~142,000 | **66%** |
| System prompt (chars) | 17,320 | 6,746 | **61%** |
| Subagent prompt 合计 | 181,568 | ~55,000 | **70%** |
| 单次 subagent 最大 prompt | 25,363 | ≤8,000 | **68%** |
| Cache hit rate | 0% | 待实测 | — |

## 项目结构

```
miniclaw/
├── bootstrap.py           # 依赖组装工厂
├── cli/app.py             # Typer CLI (chat/repl/telegram/graph/trace)
├── channels/
│   ├── loop.py            # 消息编排 + user sandbox 派生
│   ├── cli/channel.py     # Rich CLI 渠道
│   └── telegram/channel.py # Telegram 渠道 (EDIT streaming)
├── runtime/
│   ├── graph.py           # LangGraph 构建 + 节点 state 快照
│   ├── nodes.py           # classify / planner / agent / error_handler
│   ├── subagent.py        # SubagentBrief / run_subagent / ROLE_DEFAULTS
│   ├── tool_loop.py       # 共享工具执行 + trace helpers
│   ├── state.py           # RuntimeState TypedDict
│   └── service.py         # RuntimeService (turn / stream / memory / decay)
├── tools/builtin/
│   ├── filesystem.py      # read_file + write_file (sandbox scoped)
│   ├── shell.py           # 只读 shell
│   ├── spawn_subagent.py  # subagent 分派 + persist+pointer
│   └── ...                # web_search, send, cron, memory_search, skills
├── memory/
│   ├── retriever.py       # FTS5 + vec hybrid + RRF + score threshold
│   ├── indexer.py         # chunk embedding + pre-index cleaning
│   ├── files.py           # MEMORY.md 管理 + write-time compression
│   ├── decay.py           # 渐退式记忆衰退 (1d/7d/30d)
│   └── context.py         # memory context 构建 + budget 执行
├── prompting/
│   ├── context.py         # ContextBuilder (static/dynamic split + cache_control)
│   └── bootstrap.py       # SOUL.md / USER.md 加载
├── observability/
│   ├── contracts.py       # Tracer Protocol + TraceContext
│   ├── local.py           # JsonlTracer (JSONL 写入)
│   └── factory.py         # tracer 工厂 (local / langsmith / composite)
└── capabilities/
    ├── index.py           # CapabilityIndexBuilder
    └── render.py          # 名称-only 压缩渲染

tools/
├── trace_view.py          # HTML 可视化 (静态 + SSE 服务)
├── sample_trace.jsonl     # 85 条 demo trace
├── TRACE_SCHEMA.md        # JSONL schema 参考
└── README.md              # 工具使用文档
```

## License

MIT
