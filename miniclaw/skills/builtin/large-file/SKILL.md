---
name: large-file
description: Read large files via shell (head, tail, grep, awk, split). Load this when read_file fails with size limit error.
discoverable: true
---

# Large File Handling

When `read_file` rejects a file as too large, use the `shell` tool with the strategies below.

## 1. Assess first

```bash
wc -l large_file.txt            # total line count
ls -lh large_file.txt           # human-readable size
file large_file.txt             # encoding / type
head -n 5 large_file.txt        # peek at structure
```

## 2. Read a slice

```bash
head -n 100 large_file.txt              # first 100 lines
tail -n 100 large_file.txt              # last 100 lines
sed -n '500,600p' large_file.txt        # lines 500-600
```

## 3. Pattern search

```bash
grep -n 'error' large_file.txt                    # lines containing "error"
grep -n -i 'keyword' large_file.txt               # case-insensitive
grep -n -C 3 'pattern' large_file.txt             # with 3 lines of context
grep -c 'pattern' large_file.txt                  # count matches
```

## 4. Structured data (JSON / JSONL / CSV)

### JSONL (one JSON object per line)

```bash
# Count entries
wc -l data.jsonl

# Last 5 entries
tail -n 5 data.jsonl

# Search by field value
grep '"name":"error"' data.jsonl

# Pretty-print one entry
tail -n 1 data.jsonl | python3 -m json.tool

# Extract specific fields with jq (if available)
tail -n 10 data.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    ts = obj.get('timestamp', '')
    name = obj.get('name', '')
    print(f'{ts} {name}')
"
```

### JSON

```bash
# Pretty-print with truncation
python3 -c "
import json
with open('data.json') as f:
    data = json.load(f)
if isinstance(data, list):
    print(f'Array with {len(data)} items')
    print(json.dumps(data[:3], indent=2, ensure_ascii=False))
elif isinstance(data, dict):
    print(f'Object with keys: {list(data.keys())}')
"
```

### CSV

```bash
# Header + first 5 rows
head -n 6 data.csv

# Row count
wc -l data.csv

# Search rows
grep 'keyword' data.csv | head -n 10

# Column stats with Python
python3 -c "
import csv
with open('data.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'Rows: {len(rows)}')
print(f'Columns: {reader.fieldnames}')
"
```

## 5. Log files

```bash
# Recent entries
tail -n 50 app.log

# Errors only
grep -n -i 'error\|exception\|traceback' app.log

# Time-range filter (ISO timestamps)
grep '^2026-03-28T18' app.log

# Unique error types
grep -oP '(?<=ERROR: ).*' app.log | sort | uniq -c | sort -rn | head -20
```

## 6. Binary or non-UTF-8 files

```bash
file unknown_file                        # detect type
xxd unknown_file | head -n 20           # hex dump
strings unknown_file | head -n 50       # printable strings
```

## Guidelines

- Always assess size and structure before reading content.
- Prefer targeted queries (grep, sed, tail) over reading everything.
- When extracting data, pipe through `head` to limit output size.
- For repeated analysis, write a small Python script via shell instead of many individual commands.
- Keep shell output under ~8KB to avoid bloating the conversation context.
