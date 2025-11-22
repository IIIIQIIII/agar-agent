#!/bin/bash
# 测试SFT Agent集成

echo "======================================"
echo "🧪 Testing SFT Agent Integration"
echo "======================================"
echo ""

# 使用虚拟环境中的Python
if [ -d "venv" ]; then
    PYTHON="venv/bin/python"
elif [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "📋 Checking files..."
echo ""

# 检查必要文件
files=(
    "sft_server.py"
    "train_sft_from_giant.py"
    "index.html"
    "game.js"
    "style.css"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - NOT FOUND"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "❌ Some files are missing!"
    exit 1
fi

echo ""
echo "📦 Checking model files..."
echo ""

if [ -f "models/sft_giant/best_model.pt" ]; then
    echo "  ✅ models/sft_giant/best_model.pt"
    model_size=$(du -h "models/sft_giant/best_model.pt" | cut -f1)
    echo "     Size: $model_size"
elif [ -f "models/sft_giant/final_model.pt" ]; then
    echo "  ✅ models/sft_giant/final_model.pt"
    model_size=$(du -h "models/sft_giant/final_model.pt" | cut -f1)
    echo "     Size: $model_size"
else
    echo "  ⚠️  No trained model found"
    echo "     Run: ./run_giant_sft_pipeline.sh to train a model"
fi

echo ""
echo "🔧 Testing SFT server startup (dry run)..."
echo ""

# 测试导入
$PYTHON -c "
import sys
try:
    from flask import Flask
    from flask_cors import CORS
    import torch
    import numpy as np
    print('  ✅ All Python dependencies available')
except ImportError as e:
    print(f'  ❌ Missing dependency: {e}')
    sys.exit(1)

try:
    from train_sft_from_giant import BehaviorCloner
    print('  ✅ BehaviorCloner class can be imported')
except Exception as e:
    print(f'  ❌ Failed to import BehaviorCloner: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dependency check failed!"
    echo "   Run: pip install flask flask-cors torch numpy"
    exit 1
fi

echo ""
echo "🌐 Checking if ports are available..."
echo ""

check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "  ⚠️  Port $port ($name) is already in use"
        return 1
    else
        echo "  ✅ Port $port ($name) is available"
        return 0
    fi
}

check_port 5001 "RL Server"
check_port 5002 "Giant Server"
check_port 5003 "SFT Server"

echo ""
echo "✅ Integration check complete!"
echo ""
echo "======================================"
echo "🚀 Next Steps:"
echo "======================================"
echo ""
echo "1. Start all servers in separate terminals:"
echo "   Terminal 1: ./start_rl_agent.sh     (or python rl_server.py)"
echo "   Terminal 2: ./start_giant_agent.sh  (or python giant_server.py)"
echo "   Terminal 3: ./start_sft_agent.sh    (or python sft_server.py)"
echo ""
echo "2. Open the game:"
echo "   Open index.html in your browser"
echo ""
echo "3. In the game:"
echo "   - Toggle AI Mode ON"
echo "   - Select 'SFT Agent' from AI Type dropdown"
echo "   - Click 'Load SFT Model'"
echo "   - Click 'Play Game'"
echo ""
echo "4. View statistics:"
echo "   - Click 'View Stats' button while playing"
echo ""
echo "📚 For detailed instructions, see:"
echo "   README_SFT_INTEGRATION.md"
echo ""
