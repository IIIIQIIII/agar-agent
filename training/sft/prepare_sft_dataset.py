#!/usr/bin/env python3
"""
准备SFT训练数据集
将收集的 Giant Agent 对局数据转换为适合监督学习的格式
"""

import numpy as np
import pickle
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset


class GiantAgentDataset(Dataset):
    """
    PyTorch Dataset for Giant Agent expert demonstrations
    用于从专家演示中学习
    """

    def __init__(self,
                 data_dir: str = "giant_data",
                 min_score: int = 2000,
                 augment: bool = True,
                 normalize: bool = True):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            min_score: 最低分数阈值
            augment: 是否使用数据增强
            normalize: 是否归一化观察值
        """
        self.data_dir = data_dir
        self.min_score = min_score
        self.augment = augment
        self.normalize = normalize

        self.observations = []
        self.actions = []
        self.episode_ids = []
        self.rewards = []

        self._load_data()

    def _load_data(self):
        """加载所有成功对局的数据"""
        episodes_dir = Path(self.data_dir) / "episodes"

        if not episodes_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {episodes_dir}")

        # 找到所有对局文件
        episode_files = list(episodes_dir.glob("episode_*_score*.pkl"))

        if not episode_files:
            raise FileNotFoundError(f"没有找到对局数据文件在: {episodes_dir}")

        print(f"找到 {len(episode_files)} 个对局文件")

        loaded_episodes = 0
        total_data_points = 0

        for episode_file in sorted(episode_files):
            # 从文件名提取分数
            filename = episode_file.stem
            try:
                score_str = filename.split('score')[1]
                score = int(score_str)
            except:
                print(f"警告: 无法解析文件名中的分数: {filename}")
                continue

            # 只加载达标的对局
            if score < self.min_score:
                continue

            # 加载数据
            with open(episode_file, 'rb') as f:
                episode_data = pickle.load(f)

            # 验证数据
            if not episode_data.get('success', False):
                continue

            # 添加数据
            obs_list = episode_data['observations']
            act_list = episode_data['actions']
            reward_list = episode_data['rewards']

            for i in range(len(obs_list)):
                self.observations.append(obs_list[i])
                self.actions.append(act_list[i])
                self.episode_ids.append(episode_data['episode_id'])
                self.rewards.append(reward_list[i])

            loaded_episodes += 1
            total_data_points += len(obs_list)

            print(f"  加载对局 {episode_data['episode_id']}: "
                  f"分数={score}, 步数={len(obs_list)}")

        print(f"\n✅ 成功加载 {loaded_episodes} 个对局")
        print(f"   总数据点数: {total_data_points:,}")
        print(f"   平均每对局: {total_data_points/max(loaded_episodes, 1):.0f} 步\n")

        # 转换为numpy数组
        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        self.episode_ids = np.array(self.episode_ids, dtype=np.int32)
        self.rewards = np.array(self.rewards, dtype=np.float32)

        print(f"数据形状:")
        print(f"  observations: {self.observations.shape}")
        print(f"  actions: {self.actions.shape}")
        print(f"  rewards: {self.rewards.shape}\n")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        """
        返回一个训练样本

        Returns:
            observation: 状态观察 (49,)
            action: 动作标签 (标量 0-15)
            reward: 奖励 (可选，用于加权)
        """
        obs = self.observations[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        # 数据增强（可选）
        if self.augment and np.random.random() < 0.3:
            # 随机添加小噪声
            noise = np.random.normal(0, 0.01, obs.shape).astype(np.float32)
            obs = obs + noise
            obs = np.clip(obs, -1.0, 1.0)

        return {
            'observation': torch.FloatTensor(obs),
            'action': torch.LongTensor([action]),
            'reward': torch.FloatTensor([reward])
        }

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        action_counts = np.bincount(self.actions, minlength=16)
        action_distribution = action_counts / len(self.actions)

        return {
            'total_samples': len(self),
            'unique_episodes': len(np.unique(self.episode_ids)),
            'action_distribution': action_distribution.tolist(),
            'observation_mean': self.observations.mean(axis=0).tolist(),
            'observation_std': self.observations.std(axis=0).tolist(),
            'reward_mean': float(self.rewards.mean()),
            'reward_std': float(self.rewards.std()),
            'reward_min': float(self.rewards.min()),
            'reward_max': float(self.rewards.max()),
        }


def prepare_train_val_split(data_dir: str = "giant_data",
                            val_ratio: float = 0.15,
                            min_score: int = 300,
                            output_dir: str = "giant_data/processed") -> Tuple[str, str]:
    """
    准备训练集和验证集

    Args:
        data_dir: 原始数据目录
        val_ratio: 验证集比例
        min_score: 最低分数阈值
        output_dir: 处理后数据的输出目录

    Returns:
        (train_path, val_path): 训练集和验证集的路径
    """
    print(f"\n{'='*70}")
    print(f"准备训练/验证数据集")
    print(f"{'='*70}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载完整数据集
    print("加载数据...")
    dataset = GiantAgentDataset(data_dir=data_dir, min_score=min_score, augment=False)

    # 获取统计信息
    stats = dataset.get_statistics()
    print(f"\n数据集统计:")
    print(f"  总样本数: {stats['total_samples']:,}")
    print(f"  对局数: {stats['unique_episodes']}")
    print(f"  平均奖励: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
    print(f"  奖励范围: [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")

    print(f"\n动作分布:")
    for i, prob in enumerate(stats['action_distribution']):
        direction = i % 8
        split = " + Split" if i >= 8 else ""
        print(f"  动作 {i:2d} (方向{direction}{split}): {prob*100:5.2f}%")

    # 按对局划分训练集和验证集（避免数据泄漏）
    unique_episodes = np.unique(dataset.episode_ids)
    np.random.shuffle(unique_episodes)

    n_val_episodes = max(1, int(len(unique_episodes) * val_ratio))
    val_episodes = set(unique_episodes[:n_val_episodes])
    train_episodes = set(unique_episodes[n_val_episodes:])

    print(f"\n划分数据集:")
    print(f"  训练对局: {len(train_episodes)}")
    print(f"  验证对局: {len(val_episodes)}")

    # 分离数据
    train_mask = np.array([ep_id in train_episodes for ep_id in dataset.episode_ids])
    val_mask = ~train_mask

    train_data = {
        'observations': dataset.observations[train_mask],
        'actions': dataset.actions[train_mask],
        'rewards': dataset.rewards[train_mask],
        'episode_ids': dataset.episode_ids[train_mask],
    }

    val_data = {
        'observations': dataset.observations[val_mask],
        'actions': dataset.actions[val_mask],
        'rewards': dataset.rewards[val_mask],
        'episode_ids': dataset.episode_ids[val_mask],
    }

    print(f"  训练样本: {len(train_data['observations']):,}")
    print(f"  验证样本: {len(val_data['observations']):,}")

    # 保存处理后的数据
    train_path = f"{output_dir}/train_data.pkl"
    val_path = f"{output_dir}/val_data.pkl"
    stats_path = f"{output_dir}/dataset_stats.json"

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)

    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ 数据集已准备完成!")
    print(f"   训练集: {train_path}")
    print(f"   验证集: {val_path}")
    print(f"   统计信息: {stats_path}")
    print(f"{'='*70}\n")

    return train_path, val_path


def analyze_dataset(data_dir: str = "giant_data",
                   min_score: int = 300):
    """
    分析数据集质量

    Args:
        data_dir: 数据目录
        min_score: 最低分数阈值
    """
    print(f"\n{'='*70}")
    print(f"数据集质量分析")
    print(f"{'='*70}\n")

    dataset = GiantAgentDataset(data_dir=data_dir, min_score=min_score, augment=False)
    stats = dataset.get_statistics()

    # 分析动作多样性
    action_entropy = -np.sum([p * np.log(p + 1e-10) for p in stats['action_distribution']])
    max_entropy = np.log(16)  # 16个动作的最大熵

    print(f"动作多样性:")
    print(f"  熵: {action_entropy:.3f} / {max_entropy:.3f}")
    print(f"  归一化熵: {action_entropy/max_entropy*100:.1f}%")

    # 检查是否有偏向某些动作
    max_action_prob = max(stats['action_distribution'])
    if max_action_prob > 0.5:
        dominant_action = np.argmax(stats['action_distribution'])
        print(f"  ⚠️  警告: 动作 {dominant_action} 占主导 ({max_action_prob*100:.1f}%)")
    else:
        print(f"  ✅ 动作分布较均匀")

    # 分析分裂行为
    split_ratio = sum(stats['action_distribution'][8:])
    print(f"\n分裂行为:")
    print(f"  分裂动作比例: {split_ratio*100:.2f}%")

    # 分析奖励分布
    print(f"\n奖励分布:")
    print(f"  均值: {stats['reward_mean']:.2f}")
    print(f"  标准差: {stats['reward_std']:.2f}")
    print(f"  范围: [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")

    # 检查观察值分布
    obs_mean = np.array(stats['observation_mean'])
    obs_std = np.array(stats['observation_std'])

    print(f"\n观察值分布:")
    print(f"  均值范围: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"  标准差范围: [{obs_std.min():.3f}, {obs_std.max():.3f}]")

    # 检查是否有常数特征
    constant_features = np.where(obs_std < 1e-6)[0]
    if len(constant_features) > 0:
        print(f"  ⚠️  警告: {len(constant_features)} 个特征几乎为常数: {constant_features.tolist()}")
    else:
        print(f"  ✅ 所有特征都有足够变化")

    print(f"\n{'='*70}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='准备SFT训练数据集')
    parser.add_argument('--data-dir', type=str, default='giant_data',
                       help='原始数据目录 (默认: giant_data)')
    parser.add_argument('--output-dir', type=str, default='giant_data/processed',
                       help='输出目录 (默认: giant_data/processed)')
    parser.add_argument('--min-score', type=int, default=300,
                       help='最低分数阈值 (默认: 300)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例 (默认: 0.15)')
    parser.add_argument('--analyze', action='store_true',
                       help='运行数据集质量分析')

    args = parser.parse_args()

    try:
        # 分析数据集
        if args.analyze:
            analyze_dataset(data_dir=args.data_dir, min_score=args.min_score)

        # 准备训练/验证集
        train_path, val_path = prepare_train_val_split(
            data_dir=args.data_dir,
            val_ratio=args.val_ratio,
            min_score=args.min_score,
            output_dir=args.output_dir
        )

        print("✅ 数据准备完成！可以开始训练。")
        print(f"\n运行训练命令:")
        print(f"  python train_sft_from_giant.py --train-data {train_path} --val-data {val_path}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
