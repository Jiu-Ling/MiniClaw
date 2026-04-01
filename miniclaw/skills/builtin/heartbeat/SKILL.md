---
name: heartbeat
description: Manage periodic heartbeat tasks — add, remove, and monitor recurring checks.
discoverable: true
always_active: true
---

# Heartbeat Task Management

MiniClaw has a heartbeat system that wakes every 15 minutes, reads `.miniclaw/HEARTBEAT.md`, and decides whether to execute pending tasks.

## When to Add Tasks

### Explicit Requests
When the user asks for periodic monitoring or reminders:
- "定期检查...", "每天...", "提醒我...", "持续关注...", "检查...状态"
- Call `manage_heartbeat(action="add", task="具体描述")`

### Implicit Inference
When conversation context suggests periodic monitoring would help:
- **Long-running operations** (deployment, migration, build, training) → offer to add status check
- **Waiting on external results** (PR review, CI run, external API response) → offer completion check
- **Periodic maintenance** (log cleanup, backup verification) → suggest recurring check

**Important:** When inferring, ALWAYS confirm with the user before adding. Explain what task you plan to add and why.

### Do NOT Add
- One-time tasks that don't need periodic checking
- Tasks already covered by cron jobs (check with `cron` tool first)
- Vague items without clear actionable checks

## Task Description Guidelines

Write descriptions the heartbeat agent can act on:
- **Good:** "检查 GitHub CI run #1234 是否完成，如果失败则通知"
- **Bad:** "检查 CI"
- Include: service names, URLs, expected states, what to do on success/failure

## Managing Tasks

```
# Add a task
manage_heartbeat(action="add", task="检查 staging-v2 部署是否完成")

# List current tasks
manage_heartbeat(action="list")

# Remove a completed task
manage_heartbeat(action="remove", task_id="hb-001")
```

## When to Remove Tasks

- User says the task is no longer needed
- The monitored condition has resolved (deployment complete, CI passed)
- Use `list` first to find the task_id, then `remove`
