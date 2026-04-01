---
name: clawhub
description: Search, install, and manage skills from the public ClawHub registry (vector search). Load this when user asks to find or install new skills.
discoverable: true
---

# ClawHub

Public skill registry for AI agents. Search by natural language (vector search).

## When to Use

Use this skill when the user asks any of:
- "find a skill for …"
- "search for skills"
- "install a skill"
- "what skills are available?"
- "update my skills"

## Search

```bash
npx --yes clawhub@latest search "web scraping" --limit 5
```

## Install

```bash
npx --yes clawhub@latest install <slug> --workdir <workspace>/.miniclaw
```

Replace `<slug>` with the skill name from search results. This places the skill into `.miniclaw/skills/`, where MiniClaw loads workspace skills from. Always include `--workdir`.

## Update

```bash
npx --yes clawhub@latest update --all --workdir <workspace>/.miniclaw
```

## List Installed

```bash
npx --yes clawhub@latest list --workdir <workspace>/.miniclaw
```

## Path Rules

- Builtin skills ship with MiniClaw in code under `miniclaw/skills/builtin/`
- User skills live under `.miniclaw/skills/`
- User skills can override builtin skills by name

## Notes

- Requires Node.js (`npx` comes with it).
- No API key needed for search and install.
- Login (`npx --yes clawhub@latest login`) is only required for publishing.
- `--workdir` is critical — without it, skills install to the current directory instead of the MiniClaw workspace.
- After install, remind the user to start a new session to load the skill.
