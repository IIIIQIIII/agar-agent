#!/bin/bash
# 启动 Giant Agent 全局视野服务器

echo "======================================================================"
echo "🎯 Giant Agent 全局视野服务器"
echo "======================================================================"
echo ""
echo "🚀 特性："
echo "  ✅ 全局视野 (视野范围: 800-1200)"
echo "  ✅ 无限制食物/敌人数量"
echo "  ✅ 智能威胁和猎物分类"
echo "  ✅ 优化的分裂决策"
echo "  ✅ 增强的势场参数"
echo ""
echo "======================================================================"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 错误: 找不到虚拟环境 venv/"
    echo "请先运行: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# 启动服务器
echo "🌐 启动服务器在 http://localhost:5002 ..."
echo ""
./venv/bin/python giant_server.py
