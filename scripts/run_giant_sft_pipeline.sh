#!/bin/bash
# Giant Agent SFT 训练完整流程
# 自动收集数据、处理数据、训练模型

set -e  # 遇到错误立即退出

echo "=================================="
echo "Giant Agent SFT 训练流程"
echo "=================================="
echo ""

# 配置参数
EPISODES=${1:-10}           # 需要收集的成功对局数 (默认10)
SCORE_THRESHOLD=${2:-300}   # 分数阈值 (默认300)
EPOCHS=${3:-50}             # 训练轮数 (默认50)

echo "配置:"
echo "  目标对局数: $EPISODES"
echo "  分数阈值: $SCORE_THRESHOLD"
echo "  训练轮数: $EPOCHS"
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装"
    exit 1
fi

# 检查必需的包
echo "检查依赖..."
python -c "import numpy, torch, stable_baselines3, gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少必需的包，正在安装..."
    pip install numpy torch stable_baselines3 gymnasium matplotlib
fi

echo "✅ 依赖检查完成"
echo ""

# ===== 步骤 1: 收集数据 =====
echo "=================================="
echo "步骤 1/3: 收集 Giant Agent 数据"
echo "=================================="
echo ""

python training/sft/collect_giant_data.py \
    --episodes $EPISODES \
    --score-threshold $SCORE_THRESHOLD \
    --data-dir giant_data

if [ $? -ne 0 ]; then
    echo "❌ 数据收集失败"
    exit 1
fi

echo ""
echo "✅ 数据收集完成"
echo ""

# ===== 步骤 2: 处理数据 =====
echo "=================================="
echo "步骤 2/3: 处理和分析数据"
echo "=================================="
echo ""

python training/sft/prepare_sft_dataset.py \
    --data-dir giant_data \
    --output-dir giant_data/processed \
    --min-score $SCORE_THRESHOLD \
    --val-ratio 0.15 \
    --analyze

if [ $? -ne 0 ]; then
    echo "❌ 数据处理失败"
    exit 1
fi

echo ""
echo "✅ 数据处理完成"
echo ""

# ===== 步骤 3: 训练模型 =====
echo "=================================="
echo "步骤 3/3: 训练 SFT 模型"
echo "=================================="
echo ""

python training/sft/train_sft_from_giant.py \
    --train-data giant_data/processed/train_data.pkl \
    --val-data giant_data/processed/val_data.pkl \
    --output-dir models/sft_giant \
    --epochs $EPOCHS \
    --batch-size 256 \
    --lr 3e-4 \
    --patience 10

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败"
    exit 1
fi

echo ""
echo "✅ 模型训练完成"
echo ""

# ===== 完成 =====
echo "=================================="
echo "🎉 完整流程成功完成！"
echo "=================================="
echo ""
echo "生成的文件:"
echo "  数据: giant_data/episodes/"
echo "  处理后数据: giant_data/processed/"
echo "  模型: models/sft_giant/"
echo ""
echo "下一步:"
echo "  1. 查看训练曲线: models/sft_giant/training_curves.png"
echo "  2. 使用模型进行测试"
echo "  3. 或继续进行 RL fine-tuning"
echo ""
echo "测试命令:"
echo "  python test_sft_model.py --model-path models/sft_giant/best_model.pt"
echo ""
