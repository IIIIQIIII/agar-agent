#!/bin/bash
# 测试多智能体环境

echo "======================================"
echo "🧪 Testing Multi-Agent Environment"
echo "======================================"
echo ""

# 使用虚拟环境
if [ -d "venv" ]; then
    PYTHON="venv/bin/python"
elif [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 检查文件
echo "1. Checking files..."
if [ -f "multi_agent_env.py" ]; then
    echo "   ✅ multi_agent_env.py"
else
    echo "   ❌ multi_agent_env.py not found!"
    exit 1
fi

if [ -f "models/sft_giant/best_model.pt" ] || [ -f "models/sft_giant/final_model.pt" ]; then
    echo "   ✅ SFT model found"
else
    echo "   ⚠️  No SFT model (will use random opponents)"
fi

echo ""
echo "2. Testing environment..."
echo ""

# 运行测试
$PYTHON multi_agent_env.py

echo ""
echo "======================================"
echo "✅ Test Complete!"
echo "======================================"
echo ""
echo "如果测试成功，你可以:"
echo "  1. 开始后训练: ./run_sft_post_training.sh"
echo "  2. 自定义参数: ./run_sft_post_training.sh [步数] [环境数] [难度]"
echo ""
echo "示例:"
echo "  ./run_sft_post_training.sh 1000000 8 hard"
echo ""
