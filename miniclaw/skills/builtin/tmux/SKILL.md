---
name: tmux
description: Run interactive CLI programs (ssh, python REPL, top, etc.) via tmux sessions. Load this when a command requires a TTY or persistent session.
discoverable: true
---

# tmux

Use tmux when a normal non-interactive command is not enough.

## When to Use

- Interactive REPLs (python, node, psql, etc.)
- Long-running terminal sessions
- Monitoring output while sending keystrokes
- Tasks that need a live TTY

Prefer non-interactive `shell` execution first. Only reach for tmux when the task truly needs a live terminal.

## Session Naming

Use predictable names: `miniclaw-<purpose>` (e.g., `miniclaw-repl`, `miniclaw-monitor`).

## Targeting a Session

```bash
# Create a new detached session
tmux new-session -d -s miniclaw-repl

# Send a command to a session
tmux send-keys -t miniclaw-repl 'python3' Enter

# Read the current pane content
tmux capture-pane -t miniclaw-repl -p

# Send Ctrl-C to interrupt
tmux send-keys -t miniclaw-repl C-c
```

## Input Patterns

```bash
# Type text and press Enter
tmux send-keys -t <session> 'command here' Enter

# Send special keys
tmux send-keys -t <session> C-c      # Ctrl-C
tmux send-keys -t <session> C-d      # Ctrl-D / EOF
tmux send-keys -t <session> Escape   # Escape key
tmux send-keys -t <session> Tab      # Tab completion

# Multi-line input
tmux send-keys -t <session> 'line one' Enter
tmux send-keys -t <session> 'line two' Enter
```

## Reading Output

```bash
# Capture visible pane content
tmux capture-pane -t <session> -p

# Capture with scrollback (last 200 lines)
tmux capture-pane -t <session> -p -S -200
```

## Waiting for Output

Use a poll loop when you need to wait for specific text:

```bash
for i in $(seq 1 30); do
  output=$(tmux capture-pane -t <session> -p)
  if echo "$output" | grep -q "expected text"; then
    break
  fi
  sleep 1
done
```

## Cleanup

```bash
# Kill a session when done
tmux kill-session -t miniclaw-repl

# List active sessions
tmux list-sessions
```

## Multi-Agent Orchestration

When running parallel tasks, each agent should use its own session:

```bash
tmux new-session -d -s miniclaw-worker-1
tmux new-session -d -s miniclaw-worker-2
```

## Notes

- Keep session names short and predictable
- Always clean up sessions when done
- Capture pane output to verify command results before proceeding
- Use `-d` (detached) when creating sessions to avoid blocking
