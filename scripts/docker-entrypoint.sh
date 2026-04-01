#!/bin/bash
set -e

if [ ! -f "/app/.miniclaw/miniclaw.sqlite3" ]; then
    echo "🛠️ 未检测到初始化数据，正在自动执行 miniclaw init..."
    cd /app
    miniclaw init
    echo "✅ 初始化完成！"
else
    echo "✅ 已检测到初始化数据，跳过 init。"
fi

exec "$@"