#!/bin/bash
# SFT后训练启动脚本 - 竞速模式 - 优化M3 Mac Studio

set -e

echo "=========================================="
echo "🏁 SFT Post-Training - Race Mode"
echo "   First to 300 Points Wins!"
echo "=========================================="
echo ""

# 使用虚拟环境
if [ -d "venv" ]; then
    PYTHON="venv/bin/python"
elif [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 解析参数
TIMESTEPS=${1:-500000}
NUM_ENVS=${2:-8}
DIFFICULTY=${3:-medium}
TARGET_SCORE=${4:-300}

echo "配置:"
echo "  训练步数: $TIMESTEPS"
echo "  并行环境: $NUM_ENVS (M3 Mac Studio推荐: 6-8)"
echo "  难度: $DIFFICULTY"
echo "  🎯 目标分数: $TARGET_SCORE"
echo ""

# 检查SFT模型
if [ ! -f "models/sft_giant/best_model.pt" ]; then
    echo "❌ SFT模型未找到！"
    echo "   请先训练SFT模型: ./run_giant_sft_pipeline.sh"
    exit 1
fi

echo "✅ SFT模型已找到"
echo ""

# 检查依赖
echo "检查依赖..."
$PYTHON -c "import torch, stable_baselines3, gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装缺失的依赖..."
    $PYTHON -m pip install stable-baselines3[extra] gymnasium torch
fi

echo ""
echo "=========================================="
echo "🏁 开始竞速训练..."
echo "=========================================="
echo ""
echo "🎯 竞速规则:"
echo "  - 玩家 vs 5个SFT对手"
echo "  - 吃食物 +1 分，击杀对手 +30 分"
echo "  - 最快达到 $TARGET_SCORE 分获胜"
echo "  - 获胜奖励 +500，失败惩罚 -200"
echo ""
echo "提示:"
echo "  - 按 Ctrl+C 可以随时停止"
echo "  - 模型会定期保存到 models/sft_post_training/"
echo "  - 使用 TensorBoard 查看训练进度:"
echo "    tensorboard --logdir models/sft_post_training/tensorboard"
echo ""

# 启动训练
$PYTHON training/sft_post/sft_post_training.py \
    --sft-model models/sft_giant/best_model.pt \
    --output-dir models/sft_post_training \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --difficulty $DIFFICULTY \
    --lr 1e-4 \
    --target-score $TARGET_SCORE

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  1. 最佳模型: models/sft_post_training/best_model.zip"
echo "  2. TensorBoard: tensorboard --logdir models/sft_post_training/tensorboard"
echo "  3. 评估模型: $PYTHON sft_post_training.py --eval models/sft_post_training/best_model"
echo ""
