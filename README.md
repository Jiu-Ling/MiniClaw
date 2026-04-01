# MiniClaw

MiniClaw - 极简的OpenClaw复现，Powered by LangGraph。

> 🚧 持续打磨中……

## 项目简介

MiniClaw 是基于 LangGraph 构建的极简智能体运行时，集成了 LLM 驱动的任务规划、混合记忆检索、多渠道流式输出和工具编排等核心能力。支持 OpenAI 兼容模型接入、线程级 Checkpoint 持久化、Skill/MCP 渐进式激活及 Worker 并发执行。

## 架构概览

```
用户消息
    ↓
渠道层 (Telegram / CLI)
    ↓ receive()
消息编排层 MessageLoop (访问控制 → 命令拦截 → 附件处理 → 运行时调用)
    ↓
LangGraph 状态图:
    ingest → classify (小模型意图识别) → load_context (记忆检索)
                ↓ clarify (反问)              ↓
                ↓                      planner (LLM 规划) → validate → executor
                ↓                                                         ↓
                └────────────────────── complete ←────────────────────────┘
    ↓
渠道层 → 渐进式流式输出
```

### 核心组件

| 组件 | 说明 |
|---|---|
| **条件路由** | 小模型意图分类 (clarify / simple / planned)，mini_model → 主模型兜底 → 规则降级 |
| **自主规划器** | LLM 通过 `submit_plan` tool call 生成 1-8 个结构化任务 |
| **混合记忆检索** | FTS5 (BM25) + sqlite-vec (BGE-M3 1024 维) + RRF 融合，按需加载 |
| **渐进式流式输出** | 5 阶段：thinking → 模型思考 → 工具调用中 → 工具完成 → 最终回复 |
| **渠道抽象** | 统一 Protocol + 声明式 Meta 能力；Telegram (sendMessageDraft 原生流式) + CLI (Rich 渲染) |
| **命令系统** | `@command` 装饰器自动注册；渠道层拦截，零 Checkpoint 污染 |
| **Worker 编排** | Research → Execution → Review 三阶段批量调度，并行执行 |
| **访问控制** | 配对系统：6 位随机码 → `miniclaw pair <code>` → 授权。CLI 免配对 |
| **心跳与定时任务** | 周期性任务管理，可配置小模型做 skip/run 判断 |

## 技术栈

- **运行时：** Python 3.11+ / LangGraph / SQLite
- **记忆：** FTS5 + sqlite-vec (BGE-M3 via Ollama)
- **渠道：** python-telegram-bot / Rich CLI
- **模型：** OpenAI 兼容 API（任意提供商）

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

# 渠道配对（授权远程渠道访问）
miniclaw pair --list    # 查看待配对请求
miniclaw pair <code>    # 授权配对
miniclaw unpair <id>    # 撤销授权
```

> `miniclaw` 命令通过 `pyproject.toml` 的 `[project.scripts]` 注册，`uv sync` 后即可直接使用。也可以用 `uv run miniclaw` 或 `python -m miniclaw` 运行。

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

# 小模型（意图分类 + 心跳判断）
MINICLAW_MINI_MODEL=qwen3:0.6b
MINICLAW_MINI_MODEL_BASE_URL=http://localhost:11434/v1

# Embedding（混合记忆检索）
MINICLAW_EMBEDDING_MODEL=bge-m3
MINICLAW_OLLAMA_BASE_URL=http://localhost:11434

# 调试模式（流式输出中显示工具参数和结果）
MINICLAW_DEBUG=false
```

## 命令列表

| 命令 | 说明 |
|---|---|
| `/help` | 显示可用命令 |
| `/status` | 显示线程 Checkpoint 状态 |
| `/model` | 显示模型配置 |
| `/ping` | 检查机器人状态，显示线程 ID |
| `/clear` | 清除线程 Checkpoint |
| `/stop` | 停止当前线程 |
| `/new` | 开始新会话 |
| `/resume_run` | 从 Checkpoint 恢复 |
| `/retry` | 重试上一轮 |

## 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 图引擎 | LangGraph | 条件边、Checkpoint、状态管理 |
| 记忆检索 | FTS5 + sqlite-vec + RRF | 零外部依赖，BM25 + 语义混合召回 |
| 流式输出 | sendMessageDraft (Telegram) | 原生 UX，无闪烁，无速率限制 |
| 渠道抽象 | Protocol + Meta | 声明式能力协商，自动降级 |
| 意图分类 | 小模型 + 降级链 | 成本低、速度快、准确率高 |
| 命令系统 | @command 装饰器 + 渠道层拦截 | 零 Checkpoint 污染，help 自动生成 |
| 上下文压缩 | 双触发（30K 字符 + 20 条消息） | 长短消息场景均高效 |
| Markdown | HTML parse_mode | 比 MarkdownV2 转换可靠性高一个数量级 |
