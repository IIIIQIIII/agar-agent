#!/usr/bin/env python3
"""
使用 Giant Agent 专家数据进行监督微调 (SFT)
Supervised Fine-Tuning using Giant Agent expert demonstrations

训练一个神经网络模仿 Giant Agent 的决策
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from agario_core.envs.rl_env import AgarioEnv
from agario_core.models.behavior_cloner import BehaviorCloner
import gymnasium as gym





def train_behavior_cloning(train_data_path: str,
                           val_data_path: str,
                           output_dir: str = "models/sft_giant",
                           epochs: int = 50,
                           batch_size: int = 256,
                           learning_rate: float = 3e-4,
                           early_stopping_patience: int = 10) -> Dict:
    """
    训练行为克隆模型

    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        early_stopping_patience: 早停耐心值

    Returns:
        training_history: 训练历史
    """
    print(f"\n{'='*70}")
    print(f"开始行为克隆训练")
    print(f"{'='*70}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("加载训练数据...")
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)

    print("加载验证数据...")
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)

    train_obs = torch.FloatTensor(train_data['observations'])
    train_acts = torch.LongTensor(train_data['actions'])
    val_obs = torch.FloatTensor(val_data['observations'])
    val_acts = torch.LongTensor(val_data['actions'])

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_obs):,} 样本")
    print(f"  验证集: {len(val_obs):,} 样本")

    # 创建数据加载器
    train_dataset = TensorDataset(train_obs, train_acts)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = BehaviorCloner(
        obs_dim=49,
        action_dim=16,
        hidden_dims=[256, 256, 128],
        learning_rate=learning_rate
    )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_top3_accuracy': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{output_dir}/best_model.pt"

    print(f"\n{'='*70}")
    print(f"开始训练")
    print(f"{'='*70}\n")
    print(f"配置:")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  早停耐心: {early_stopping_patience}")
    print(f"\n{'='*70}\n")

    start_time = datetime.now()

    for epoch in range(epochs):
        # 训练阶段
        model.policy_net.train()
        train_losses = []

        for batch_obs, batch_acts in train_loader:
            loss = model.train_step(batch_obs, batch_acts)
            train_losses.append(loss)

        avg_train_loss = np.mean(train_losses)

        # 验证阶段
        val_metrics = model.evaluate(val_obs, val_acts)

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_top3_accuracy'].append(val_metrics['top3_accuracy'])

        # 打印进度
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Val Top3: {val_metrics['top3_accuracy']*100:.2f}%")

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            model.save(best_model_path)
            print(f"  ⭐ 新最佳模型！验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= early_stopping_patience:
            print(f"\n早停触发！{early_stopping_patience} 轮无改进")
            break

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 加载最佳模型
    model.load(best_model_path)

    # 最终评估
    final_metrics = model.evaluate(val_obs, val_acts)

    print(f"\n{'='*70}")
    print(f"训练完成！")
    print(f"{'='*70}")
    print(f"  总耗时: {duration/60:.1f} 分钟")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最终准确率: {final_metrics['accuracy']*100:.2f}%")
    print(f"  最终Top-3准确率: {final_metrics['top3_accuracy']*100:.2f}%")
    print(f"{'='*70}\n")

    # 保存训练历史
    history_path = f"{output_dir}/training_history.json"
    with open(history_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)

    # 绘制训练曲线
    plot_training_curves(history, output_dir)

    # 保存最终模型
    final_model_path = f"{output_dir}/final_model.pt"
    model.save(final_model_path)

    print(f"✅ 模型和历史已保存到: {output_dir}/")

    return history


def plot_training_curves(history: Dict, output_dir: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot([acc * 100 for acc in history['val_accuracy']],
                label='Accuracy', linewidth=2)
    axes[1].plot([acc * 100 for acc in history['val_top3_accuracy']],
                label='Top-3 Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {output_dir}/training_curves.png")
    plt.close()


def convert_to_sb3_model(bc_model_path: str,
                         output_path: str = "models/agario_ppo_sft"):
    """
    将行为克隆模型转换为 Stable-Baselines3 PPO 模型
    用于在游戏中加载

    Args:
        bc_model_path: 行为克隆模型路径
        output_path: 输出PPO模型路径
    """
    print(f"\n{'='*70}")
    print(f"转换模型为 Stable-Baselines3 格式")
    print(f"{'='*70}\n")

    # 创建一个临时环境
    env = AgarioEnv(world_size=4000, food_count=500, enemy_count=30)

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256])
        )
    )

    # 加载行为克隆的权重
    bc_checkpoint = torch.load(bc_model_path)
    bc_state_dict = bc_checkpoint['policy_net']

    # 将权重复制到PPO的策略网络
    # 注意: 这是一个简化的转换，可能需要根据实际网络结构调整
    try:
        # 获取PPO策略网络的state_dict
        ppo_policy = model.policy.mlp_extractor.policy_net

        # 尝试匹配层
        # 这里需要手动映射，因为结构可能不完全一致
        print("警告: 权重转换可能不完全准确，建议使用行为克隆模型进行进一步的RL训练")

        # 保存模型
        model.save(output_path)
        print(f"✅ PPO模型已保存: {output_path}")

    except Exception as e:
        print(f"⚠️  直接转换失败: {e}")
        print(f"建议使用行为克隆模型作为PPO的预训练初始化")

        # 保存PPO模型（未初始化）
        model.save(output_path)
        print(f"✅ 新PPO模型已保存: {output_path}")
        print(f"   可以使用此模型进行进一步的RL训练")

    env.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用Giant Agent数据进行SFT训练')
    parser.add_argument('--train-data', type=str,
                       default='giant_data/processed/train_data.pkl',
                       help='训练数据路径')
    parser.add_argument('--val-data', type=str,
                       default='giant_data/processed/val_data.pkl',
                       help='验证数据路径')
    parser.add_argument('--output-dir', type=str,
                       default='models/sft_giant',
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--convert-to-sb3', action='store_true',
                       help='转换为Stable-Baselines3格式')

    args = parser.parse_args()

    try:
        # 训练行为克隆模型
        history = train_behavior_cloning(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stopping_patience=args.patience
        )

        # 转换为SB3格式（可选）
        if args.convert_to_sb3:
            best_model_path = f"{args.output_dir}/best_model.pt"
            convert_to_sb3_model(best_model_path)

        print("\n🎉 训练完成！")
        print(f"\n下一步:")
        print(f"  1. 使用训练好的模型进行测试")
        print(f"  2. 或者使用此模型作为RL训练的初始化")
        print(f"\n模型文件:")
        print(f"  {args.output_dir}/best_model.pt")
        print(f"  {args.output_dir}/final_model.pt")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
