#!/bin/bash
# 快速演示：收集少量数据并训练
# 适合测试和演示，约10-15分钟完成

echo "=================================="
echo "Giant Agent SFT 快速演示"
echo "=================================="
echo ""
echo "这个演示将："
echo "  1. 收集 3 个成功对局（分数 ≥ 300）"
echo "  2. 处理数据"
echo "  3. 训练 20 轮"
echo "  4. 测试模型"
echo ""
echo "预计耗时: 10-15 分钟"
echo ""
read -p "按 Enter 继续..."

# 步骤1: 收集数据
echo ""
echo "==== 步骤 1/4: 收集数据 ===="
python collect_giant_data.py \
    --episodes 3 \
    --score-threshold 300 \
    --data-dir giant_data_demo

if [ $? -ne 0 ]; then
    echo "❌ 数据收集失败"
    exit 1
fi

# 步骤2: 处理数据
echo ""
echo "==== 步骤 2/4: 处理数据 ===="
python prepare_sft_dataset.py \
    --data-dir giant_data_demo \
    --output-dir giant_data_demo/processed \
    --min-score 300

if [ $? -ne 0 ]; then
    echo "❌ 数据处理失败"
    exit 1
fi

# 步骤3: 训练模型
echo ""
echo "==== 步骤 3/4: 训练模型 ===="
python train_sft_from_giant.py \
    --train-data giant_data_demo/processed/train_data.pkl \
    --val-data giant_data_demo/processed/val_data.pkl \
    --output-dir models/sft_giant_demo \
    --epochs 20 \
    --batch-size 128

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

# 步骤4: 测试模型
echo ""
echo "==== 步骤 4/4: 测试模型 ===="
python test_sft_model.py \
    --model-path models/sft_giant_demo/best_model.pt \
    --episodes 3

echo ""
echo "=================================="
echo "✅ 演示完成！"
echo "=================================="
echo ""
echo "生成的文件："
echo "  - 数据: giant_data_demo/"
echo "  - 模型: models/sft_giant_demo/"
echo "  - 训练曲线: models/sft_giant_demo/training_curves.png"
echo ""
echo "查看训练曲线："
echo "  open models/sft_giant_demo/training_curves.png"
echo ""
echo "运行完整训练："
echo "  ./run_giant_sft_pipeline.sh 10 2000 50"
echo ""
