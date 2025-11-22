#!/usr/bin/env python3
"""
自动化 SFT 训练流程
监控数据收集进度，完成后自动处理数据和训练模型
"""

import os
import time
import subprocess
import json
from datetime import datetime


def count_collected_episodes(data_dir: str = "giant_data") -> int:
    """统计已收集的对局数"""
    episodes_dir = f"{data_dir}/episodes"
    if not os.path.exists(episodes_dir):
        return 0
    pkl_files = [f for f in os.listdir(episodes_dir) if f.endswith('.pkl')]
    return len(pkl_files)


def wait_for_collection(target_episodes: int = 300,
                       data_dir: str = "giant_data",
                       check_interval: int = 30):
    """
    等待数据收集完成

    Args:
        target_episodes: 目标对局数
        data_dir: 数据目录
        check_interval: 检查间隔（秒）
    """
    print(f"\n{'='*70}")
    print(f"🔍 监控数据收集进度")
    print(f"{'='*70}")
    print(f"目标: {target_episodes} 个对局")
    print(f"检查间隔: {check_interval} 秒")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    last_count = 0

    while True:
        current_count = count_collected_episodes(data_dir)

        if current_count != last_count:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = current_count / (elapsed / 60) if elapsed > 0 else 0
            eta_minutes = (target_episodes - current_count) / rate if rate > 0 else 0

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"进度: {current_count}/{target_episodes} "
                  f"({current_count/target_episodes*100:.1f}%) | "
                  f"速度: {rate:.1f} 对局/分钟 | "
                  f"预计剩余: {eta_minutes:.1f} 分钟")

            last_count = current_count

        if current_count >= target_episodes:
            print(f"\n✅ 数据收集完成！共收集 {current_count} 个对局")
            print(f"总耗时: {(datetime.now() - start_time).total_seconds()/60:.1f} 分钟\n")
            break

        time.sleep(check_interval)


def run_command(command: str, description: str) -> bool:
    """
    运行命令并实时显示输出

    Args:
        command: 要运行的命令
        description: 命令描述

    Returns:
        是否成功
    """
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"命令: {command}\n")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时输出
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            print(f"\n✅ {description} 完成！\n")
            return True
        else:
            print(f"\n❌ {description} 失败（退出码: {process.returncode}）\n")
            return False

    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        return False


def main():
    """主流程"""
    import argparse

    parser = argparse.ArgumentParser(description='自动化 SFT 训练流程')
    parser.add_argument('--target-episodes', type=int, default=300,
                       help='等待收集的对局数 (默认: 300)')
    parser.add_argument('--data-dir', type=str, default='giant_data',
                       help='数据目录 (默认: giant_data)')
    parser.add_argument('--score-threshold', type=int, default=300,
                       help='分数阈值 (默认: 300)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='批次大小 (默认: 256)')
    parser.add_argument('--skip-wait', action='store_true',
                       help='跳过等待，直接处理和训练')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🚀 自动化 SFT 训练流程")
    print(f"{'='*70}")
    print(f"配置:")
    print(f"  目标对局数: {args.target_episodes}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  分数阈值: {args.score_threshold}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    # ===== 步骤 1: 等待数据收集完成 =====
    if not args.skip_wait:
        try:
            wait_for_collection(
                target_episodes=args.target_episodes,
                data_dir=args.data_dir,
                check_interval=30
            )
        except KeyboardInterrupt:
            print("\n⚠️  用户中断等待")
            current_count = count_collected_episodes(args.data_dir)
            if current_count < 10:
                print(f"❌ 数据不足（{current_count} < 10），无法继续")
                return
            print(f"将使用已收集的 {current_count} 个对局继续")
    else:
        current_count = count_collected_episodes(args.data_dir)
        print(f"跳过等待，当前已有 {current_count} 个对局")

    # ===== 步骤 2: 处理数据 =====
    success = run_command(
        f"venv/bin/python prepare_sft_dataset.py "
        f"--data-dir {args.data_dir} "
        f"--output-dir {args.data_dir}/processed "
        f"--min-score {args.score_threshold} "
        f"--val-ratio 0.15 "
        f"--analyze",
        "步骤 2/3: 处理和分析数据"
    )

    if not success:
        print("❌ 数据处理失败，终止流程")
        return

    # ===== 步骤 3: 训练模型 =====
    success = run_command(
        f"venv/bin/python train_sft_from_giant.py "
        f"--train-data {args.data_dir}/processed/train_data.pkl "
        f"--val-data {args.data_dir}/processed/val_data.pkl "
        f"--output-dir models/sft_giant "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--lr 3e-4 "
        f"--patience 10",
        "步骤 3/3: 训练 SFT 模型"
    )

    if not success:
        print("❌ 模型训练失败")
        return

    # ===== 完成 =====
    total_time = (datetime.now() - start_time).total_seconds() / 60

    print(f"\n{'='*70}")
    print(f"🎉 完整流程成功完成！")
    print(f"{'='*70}")
    print(f"总耗时: {total_time:.1f} 分钟")
    print(f"\n生成的文件:")
    print(f"  数据: {args.data_dir}/")
    print(f"  模型: models/sft_giant/")
    print(f"\n下一步:")
    print(f"  1. 查看训练曲线: open models/sft_giant/training_curves.png")
    print(f"  2. 测试模型: venv/bin/python test_sft_model.py --episodes 10")
    print(f"  3. 对比测试: venv/bin/python test_sft_model.py --compare --episodes 5")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
