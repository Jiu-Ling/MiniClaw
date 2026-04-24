# MiniClaw

LangGraph-based agent runtime with prompt caching, hybrid memory, structured compression, and end-to-end observability.

> 极简的 multi-agent 运行时：4 breakpoint sliding window prompt cache、FTS5+sqlite-vec 混合记忆、A+B+C 压缩信息保留（pinned references + LLM 提取 + agent-driven remember）、tool 级 trace + LangSmith 可视化。

## 项目简介

MiniClaw 是基于 LangGraph 构建的智能体运行时，按四大子系统组织：

- **Prompt Cache 子系统**：4 个 breakpoint 的 sliding window 布局（tools / system / penultimate / latest），按 provider 自动分流（anthropic / openai_auto / none），易变信息（runtime metadata、retrieved memory）通过 `<system-reminder>` 注入到当轮 user message，不污染 cached system 段
- **Memory Write 子系统**：`MemoryFileStore` 是 `MEMORY.md` 唯一权威读写器；2 段 schema（`## Critical Preferences` + `## Long-term Facts`）；`add_fact` / `add_facts_batch` API 含 dedup + critical FIFO 控顶；per-thread 叙述写入 `daily/YYYY-MM-DD.md`，FTS5+vec 索引可搜索
- **Compression 信息保留三件套**：A. 同步正则提取 pinned references（path/URL/identifier 逐字保留）→ summary 段；B. 后台 LLM 异步提取 facts + narrative → MEMORY.md + daily MD；C. agent 主动调 `remember` 工具持久化发现的事实
- **Observability 子系统**：Tracer Protocol（run/span/event）+ JsonlTracer + LangSmithTracer + Composite；CLI `trace tail` 增强渲染（duration / 工具参数内联 / cache 命中率 / 多 filter）+ `trace summary` 聚合统计；HTML 可视化（`tools/trace_view.py`）
- **动态 Subagent 分派**：主 agent 通过 `spawn_subagent` 工具按需拆分任务，同一轮多 spawn 并行执行，subagent 在 clean-room 上下文运行，与主 agent 共享 BP2 cache namespace（同 role/thread/channel 的多次 spawn 复用 system 段）
- **混合记忆检索**：FTS5 (BM25) + sqlite-vec (BGE-M3 1024 维) + RRF 融合；rewrite + intent routing + critical/normal 分级 consolidation 后台流水线
- **多渠道流式**：Telegram (edit_message 渐进式) + CLI (Rich 渲染)；Per-user 工作区沙箱（`write_file` 限定，shell 只读）

## 架构概览

```
用户消息
    ↓
渠道层 (Telegram / CLI)
    ↓ receive()
MessageLoop（访问控制 → 命令拦截 → 附件处理 → user_id/sandbox 派生）
    ↓
LangGraph 状态图:
    ingest → classify (mini 模型) → load_context（混合检索 + rewrite）
                                         ↓
                ┌── clarify ──────→ complete
                ├── simple ───────→ agent ──────→ complete
                └── planned ──→ planner → agent ──→ complete
                                         ↓ (任何节点 last_error)
                                    error_handler → complete

    agent 内部 (tool loop, max 16 rounds):
        provider.achat (BP1-4 cache_control) → [tool_calls?] → execute_tool_calls (并行)
            ├── tool.read_file / shell / web_search / memory_search / remember / ...
            ├── tool.spawn_subagent → run_subagent (clean-room, 独立 tool loop)
            │       ├── subagent.tool_loop.round_N → provider.achat → tool.*
            │       └── 结果 >4K 自动写入 sandbox，返回摘要 + 路径
            └── tool.write_file (限定 user sandbox)

    history compression (每 30K chars / 20 messages 触发):
        sync:  extract_pinned_references → ## Pinned References 段进 summary
        async: BackgroundScheduler → llm_extract_facts → add_facts_batch + append_to_daily_journal
```

## 四大子系统

### Prompt Cache（`miniclaw/prompting/` + `miniclaw/providers/openai_compat.py`）

请求结构（`cache_strategy="anthropic"` 时）：

```
tools[..., {..., cache_control: ephemeral}]                  ← BP1 (工具列表稳定)
system: [{ static_text, cache_control: ephemeral }]          ← BP2 (system + Critical Preferences + Long-term Facts)
messages: [
  ...历史消息...,
  {role: assistant, content_parts: [{text, cache_control}]}  ← BP3 (penultimate, 借 lookback 命中上轮 BP4)
  {role: user, content_parts: [                              ← BP4 (latest, 当轮新写)
      {text: USER_INPUT_WITH_REMINDERS, cache_control}
  ]}
]
```

- **静态段**（system message）= system_prompt + bootstrap files + capability index + Critical Preferences + Long-term Facts；跨轮稳定 → cache 命中率高
- **易变信息**走 `<system-reminder>` 标签注入到当轮 user message：runtime metadata（thread_id/channel/clock）+ retrieved related_context
- Provider 分流：`anthropic` 透传 cache_control；`openai_auto` / `none` 剥离；自动检测 base_url（anthropic.com / dashscope → anthropic, openai.com → openai_auto）
- Subagent 系统消息单 part 静态（fleet_id/sub_id 进 user message reminder），同 (role, thread_id, channel) 的多次 spawn 共享 BP2 cache namespace

### Memory Write（`miniclaw/memory/files.py`）

`MEMORY.md` 唯一 schema：

```markdown
# Memory

## Critical Preferences
- [id:0] User prefers Chinese responses
- [id:1] Project uses BGE-M3

## Long-term Facts
- [id:2] API endpoint is https://api.example.com
- [id:3] PostgreSQL runs on port 5433
```

API：
- `MemoryFileStore.add_fact(fact, *, tier, source, dedup) -> bool` — 单 fact，幂等 dedup
- `MemoryFileStore.add_facts_batch(candidates, *, dedup_against_existing, critical_max) -> AddFactsResult` — 批量，跨批+本批 dedup，critical FIFO 控顶
- `MemoryFileStore.append_to_daily_journal(thread_id, narrative, source)` — 写入 `daily/YYYY-MM-DD.md`（被 indexer 自动索引）
- 并发保护：`threading.RLock`
- 旧版 `## Recent Work` 段在读取时静默跳过（向后兼容）

### Compression 信息保留（`miniclaw/memory/extract.py` + `compression_promote.py`）

每次历史压缩（>30K chars 或 >20 messages 触发）走三层防御：

```
sync 同步路径 (<100ms):
  extract_pinned_references(dropped_messages)
    → 正则提取: paths / URLs / function calls / SCREAMING_CONST / code blocks
    → ## Pinned References (verbatim) 段嵌入 compression summary 消息
    → 关键字面量 100% 保留

async 异步路径 (BackgroundScheduler):
  llm_extract_facts(dropped, pinned_references)
    → CompressionExtraction { narrative, facts_to_remember, discovered_facts }
    → memory_store.add_facts_batch(...)         # MEMORY.md 持久化
    → memory_store.append_to_daily_journal(...) # 索引到 FTS5+vec
    → indexer.mark_dirty(file)                  # 下轮 load_context 自动索引

agent 主动:
  remember(fact, tier="normal" | "critical", reason="") -> "Remembered (...) ..."
    → 在工具执行中发现关键事实时调用，立刻写入 MEMORY.md
    → dedup 自动；soft cap 50 calls / process
```

### Observability（`miniclaw/observability/` + `tools/trace_view.py`）

- **Tracer Protocol**：`start_run` / `finish_run` / `start_span` / `finish_span` / `record_event`
- **3 个实现 + Composite**：`JsonlTracer`（本地 JSONL）、`LangSmithTracer`（远端，按 span name 推断 `run_type=tool/llm/retriever/chain`）、`CompositeTracer`（多目标分发）
- **每个 tool 调用都有 `tool.<name>` span**：包含 `tool.name` + `tool.source`（builtin/skill/mcp）+ inputs (arguments) + outputs (truncated content)
- **每次 chat 完成发射 `prompt.cache.usage` 事件**：含 `prompt_tokens` / `cached_tokens` / `cache_hit_rate`
- **CLI `trace tail`** 增强渲染：duration、工具参数内联、状态着色、`--tool` / `--kind` / `--min-ms` / `--status` 过滤
- **CLI `trace summary`** 聚合统计：top tools、cache 命中率、error 分布
- **`tools/trace_view.py`** HTML 可视化：折叠 span 树、flame graph 时间轴、明暗主题、cache stats 面板、tool source 徽章

## 技术栈

- **运行时**：Python 3.12 / LangGraph / SQLite
- **记忆**：FTS5 + sqlite-vec (BGE-M3 via Ollama)
- **渠道**：python-telegram-bot / Rich CLI
- **模型**：OpenAI 兼容 API（Anthropic Claude / DashScope / OpenAI / 通义千问）
- **观测**：JSONL trace + LangSmith SDK + 自带 HTML 可视化（纯 stdlib）

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key、模型名称和 Base URL

# 初始化本地存储
uv run miniclaw init

# 单轮对话
uv run miniclaw chat "你好"

# 交互式 REPL（Rich Markdown 渲染）
uv run miniclaw repl

# Telegram 机器人
uv run miniclaw telegram polling
```

## CLI 命令

### 运行时

| 命令 | 说明 |
|---|---|
| `miniclaw init` | 初始化本地存储 |
| `miniclaw chat "..."` | 单轮对话 |
| `miniclaw repl` | 交互式 REPL |
| `miniclaw telegram polling` | 启动 Telegram bot |
| `miniclaw pair <code>` | 授权远程渠道 |
| `miniclaw graph [--format mermaid\|ascii\|png]` | 导出 LangGraph 拓扑 |

### 观测

| 命令 | 说明 |
|---|---|
| `miniclaw trace tail <path>` | 实时查看 JSONL trace（含 duration / 工具参数 / cache 命中率） |
| `miniclaw trace tail <path> --tool shell --min-ms 100` | 只看 shell 工具且 ≥100ms 的调用 |
| `miniclaw trace tail <path> --status error` | 只看错误 |
| `miniclaw trace summary <path>` | 聚合统计：top tools / cache hit rate / errors |
| `python tools/trace_view.py <path>` | 静态 HTML 导出（可分享单文件） |
| `python tools/trace_view.py --serve --port 8787` | 实时服务模式（SSE 推送，浏览器自动刷新） |

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

# Mini 模型（意图分类 / memory rewrite / compression 提取）
MINICLAW_MINI_MODEL=qwen-flash
MINICLAW_MINI_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Embedding（混合记忆检索）
MINICLAW_EMBEDDING_MODEL=bge-m3
MINICLAW_OLLAMA_BASE_URL=http://localhost:11434

# Prompt cache 策略：auto / anthropic / openai_auto / none
# auto = 根据 base_url 自动检测（anthropic/dashscope→anthropic, openai→openai_auto）
MINICLAW_CACHE_STRATEGY=auto

# Compression 信息保留
MINICLAW_COMPRESSION_PINNED_EXTRACT_ENABLED=true
MINICLAW_COMPRESSION_EXTRACT_ENABLED=true
MINICLAW_COMPRESSION_EXTRACT_MODEL_TIER=mini

# Remember 工具
MINICLAW_REMEMBER_TOOL_ENABLED=true
MINICLAW_REMEMBER_TOOL_MAX_CALLS=50

# Tracing
MINICLAW_TRACE_MODE=local  # off / local / langsmith / both
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=miniclaw
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 图引擎 | LangGraph | 条件边、Checkpoint、状态管理 |
| Prompt Cache | 4 BP sliding window（Claude Code 模式） | Anthropic 20-block lookback 让 BP4 of turn N 自动成为 BP3 of turn N+1，cache 写入永不浪费 |
| 易变信息注入 | `<system-reminder>` 包裹进 user message | 不污染 cached system 段；Anthropic 训练时见过该标签 |
| 长期事实存储 | MEMORY.md（2 段）+ daily/*.md | 人可读、git friendly、可手编；FTS5+vec 索引日 MD |
| Memory schema | `MemoryFileStore` 单一权威 | 消除"两个写者两个 schema"的旧 bug |
| 压缩信息保留 | 同步正则 + 异步 LLM + agent 主动 remember | 三层防御，failure 隔离 |
| Subagent 分派 | `spawn_subagent` 工具 + clean-room | 动态分派 > 静态 phase 编排；零上下文继承 |
| 记忆检索 | FTS5 + sqlite-vec + RRF | 零外部依赖，BM25 + 语义混合召回 |
| 工具安全 | `worker_visible` + shell 只读 + sandbox | Subagent 不能 send/cron/spawn；文件写入限用户目录 |
| Tracing | 单 trace_id + per-tool span + cache event | tool 级 + cache 级双重可观测，LangSmith 自动适配 run_type |
| Spawn 结果 | persist + pointer | >4K 写文件返路径，防 context 爆炸 |
| 渠道抽象 | Protocol + 能力声明 | 自动降级（NATIVE/EDIT/BUFFER） |

## 项目结构

```
miniclaw/
├── bootstrap.py                # 依赖组装工厂
├── cli/
│   ├── app.py                  # Typer CLI（chat/repl/telegram/graph/trace tail/trace summary）
│   └── trace_renderer.py       # 纯模块：duration / 工具参数内联 / 聚合统计
├── channels/
│   ├── loop.py                 # 消息编排 + user sandbox 派生
│   ├── cli/channel.py          # Rich CLI 渠道
│   └── telegram/channel.py     # Telegram 渠道（EDIT streaming）
├── runtime/
│   ├── graph.py                # LangGraph 构建
│   ├── nodes.py                # classify / planner / agent / error_handler
│   ├── subagent.py             # SubagentBrief / run_subagent / ROLE_DEFAULTS
│   ├── tool_loop.py            # 共享工具执行 + tool span 注入 source metadata
│   ├── state.py                # RuntimeState TypedDict
│   ├── service.py              # RuntimeService（turn / stream / 协调 compression callback）
│   ├── background.py           # BackgroundScheduler（单 worker FIFO）
│   └── checkpoint.py           # AsyncSQLiteCheckpointer
├── memory/
│   ├── files.py                # MemoryFileStore：唯一权威读写 + RLock + add_fact / add_facts_batch / append_to_daily_journal
│   ├── extract.py              # PinnedReferences + extract_pinned_references + CompressionExtraction + llm_extract_facts
│   ├── compression_promote.py  # CompressionEvent + schedule_compression_promotion（BackgroundScheduler 编排）
│   ├── consolidation.py        # 后台 LLM consolidation（critical/normal 分级）
│   ├── retriever.py            # FTS5 + vec hybrid + RRF
│   ├── indexer.py              # parent-child chunk + embedding + 索引日 MD
│   ├── rewrite.py              # query rewrite + intent routing
│   ├── chunker.py              # langchain RecursiveCharacterTextSplitter
│   ├── embedding.py            # Ollama BGE-M3 client
│   └── context.py              # MemoryContext dataclass + build_memory_context
├── prompting/
│   ├── context.py              # ContextBuilder：4 BP cache + system-reminder 注入 + on_compression callback
│   ├── runtime_metadata.py     # render_runtime_metadata_block → <system-reminder>
│   └── bootstrap.py            # SOUL.md / USER.md / TOOLS.md / AGENTS.md 加载
├── providers/
│   ├── contracts.py            # ChatProvider Protocol + ChatMessage / ChatResponse / ChatUsage
│   └── openai_compat.py        # OpenAICompatibleProvider + cache_strategy 分流 + BP1 tools 注入
├── tools/
│   ├── builtin/
│   │   ├── filesystem.py       # read_file + write_file（sandbox scoped）
│   │   ├── shell.py            # 只读 shell
│   │   ├── spawn_subagent.py   # subagent 分派 + persist+pointer
│   │   ├── memory_search.py    # FTS5+vec 召回
│   │   ├── remember.py         # 主动事实持久化（Phase 6）
│   │   └── ...                 # web_search, send, cron, skills, activation
│   └── registry.py             # ToolRegistry + worker_visible 过滤
├── observability/
│   ├── contracts.py            # Tracer Protocol + TraceContext + TraceRecord
│   ├── local.py                # JsonlTracer
│   ├── langsmith.py            # LangSmithTracer（按 name 推断 run_type，事件累积）
│   ├── composite.py            # CompositeTracer
│   ├── safe.py                 # safe_start_span / safe_finish_span / safe_record_event
│   ├── cache_event.py          # emit_cache_usage 辅助
│   └── factory.py              # tracer 工厂
└── capabilities/
    ├── index.py                # CapabilityIndexBuilder
    └── render.py               # 渐进式激活渲染

tools/
├── trace_view.py               # HTML 可视化（静态 + SSE，含 cache stats + tool source 徽章）
├── sample_trace.jsonl          # demo trace
├── TRACE_SCHEMA.md             # JSONL schema 参考
└── README.md                   # 工具使用文档

docs/
├── PROJECT_ARCHITECTURE.md     # 完整架构
├── PROMPT_ENGINE_LESSONS.md    # Prompt 工程经验总结
├── INTERVIEW_QA.md             # FAQ
└── superpowers/
    ├── specs/                  # 设计文档（gitignored，本地）
    └── plans/                  # 实施计划（gitignored，本地）
```

## License

MIT
