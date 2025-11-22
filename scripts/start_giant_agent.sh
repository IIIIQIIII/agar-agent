#!/bin/bash

# Giant Agent 启动脚本

echo "========================================"
echo "🎮 Giant Agent - Agar.io AI System"
echo "========================================"
echo ""

# 切换到项目目录
cd "$(dirname "$0")"

# 检查 Python 虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    pip install flask flask-cors numpy
else
    source venv/bin/activate
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import flask, flask_cors, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖，正在安装..."
    pip install flask flask-cors numpy
fi

echo "✅ 依赖检查完成"
echo ""

# 启动 Giant Agent 服务器
echo "🚀 启动 Giant Agent 服务器 (端口 5002)..."
echo ""

python giant_server.py &
SERVER_PID=$!

# 等待服务器启动
echo "⏳ 等待服务器启动..."
sleep 3

# 检查服务器是否成功启动
curl -s http://localhost:5002/api/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ 服务器启动成功！"
    echo ""
    echo "📡 API 端点:"
    echo "  - Health Check: http://localhost:5002/api/health"
    echo "  - Action:       http://localhost:5002/api/action"
    echo "  - Stats:        http://localhost:5002/api/stats"
    echo ""
    echo "🌐 打开游戏界面..."

    # 在默认浏览器中打开游戏
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open index.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open index.html
    fi

    echo ""
    echo "🎮 游戏已在浏览器中打开！"
    echo "   1. 点击 'Play Game' 开始游戏"
    echo "   2. 启用 'AI Mode' 开关"
    echo "   3. 选择 'Giant Agent (Math)'"
    echo "   4. 观看 AI 自动游戏！"
    echo ""
    echo "💡 按 Ctrl+C 停止服务器"
    echo "========================================"

    # 等待用户中断
    wait $SERVER_PID
else
    echo "❌ 服务器启动失败！"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# 清理
echo ""
echo "👋 服务器已停止"
