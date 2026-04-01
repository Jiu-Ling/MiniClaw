#!/bin/bash

if [ $# -eq 0 ]; then
    set -- "--help"
fi

if docker compose ps -q miniclaw | grep -q .; then
    docker compose exec -it miniclaw miniclaw "$@"
else
    echo "⚠️ 请先启动miniclaw容器，再执行命令"
fi