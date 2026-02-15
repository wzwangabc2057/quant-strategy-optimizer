#!/bin/bash
#
# 启动量化策略 API 服务
#
# 用法:
#   bash run_api.sh              # 默认端口 8000
#   bash run_api.sh 8080         # 指定端口
#

PORT=${1:-8000}

echo "=========================================="
echo "量化策略 API 服务"
echo "端口: $PORT"
echo "=========================================="

cd "$(dirname "$0")"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 启动服务
uvicorn api.server:app --host 0.0.0.0 --port $PORT --reload

# 访问地址:
# - API 文档: http://localhost:$PORT/docs
# - 健康检查: http://localhost:$PORT/health
# - 获取持仓: http://localhost:$PORT/portfolio
