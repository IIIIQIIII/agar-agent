#!/bin/bash
# 启动 SFT Agent 服务器

echo "======================================"
echo "🤖 Starting SFT Agent Server"
echo "======================================"
echo ""

# 检查Python环境
if [ -d "venv" ]; then
    echo "✅ Found virtual environment"
    PYTHON="venv/bin/python"
elif [ -d ".venv" ]; then
    echo "✅ Found virtual environment"
    PYTHON=".venv/bin/python"
else
    echo "⚠️  No virtual environment found, using system Python"
    PYTHON="python3"
fi

# 检查必要的文件
if [ ! -f "sft_server.py" ]; then
    echo "❌ Error: sft_server.py not found!"
    exit 1
fi

if [ ! -f "train_sft_from_giant.py" ]; then
    echo "❌ Error: train_sft_from_giant.py not found!"
    exit 1
fi

# 检查模型文件
if [ -f "models/sft_giant/best_model.pt" ]; then
    echo "✅ Found trained SFT model: models/sft_giant/best_model.pt"
elif [ -f "models/sft_giant/final_model.pt" ]; then
    echo "✅ Found trained SFT model: models/sft_giant/final_model.pt"
else
    echo "⚠️  Warning: No trained SFT model found in models/sft_giant/"
    echo "   You will need to train a model first or load one manually"
    echo ""
    echo "   To train a model, run:"
    echo "   ./run_giant_sft_pipeline.sh"
    echo ""
fi

# 检查依赖
echo "Checking dependencies..."
$PYTHON -c "import torch, flask, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing"
    echo "Installing required packages..."
    $PYTHON -m pip install torch flask flask-cors numpy
fi

echo ""
echo "======================================"
echo "🚀 Starting SFT Server on port 5003"
echo "======================================"
echo ""

# 启动服务器
$PYTHON sft_server.py
