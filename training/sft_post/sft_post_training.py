#!/usr/bin/env python3
"""
SFT Post-Training Script
Initializes PPO policy with SFT weights and fine-tunes with RL
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn # This was not in the original, but is in the provided snippet. I will add it.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from agario_core.envs.multi_agent_env import MultiAgentAgarioEnv
from training.sft.train_sft_from_giant import BehaviorCloner # Path changed from train_sft_from_giant to training.sft.train_sft_from_giant
from datetime import datetime, time
from typing import Dict, List, Optional
import json


class SFTInitPolicy:
    """从SFT模型初始化PPO策略"""

    @staticmethod
    def load_sft_weights_to_ppo(
        ppo_model: PPO,
        sft_model_path: str
    ) -> PPO:
        """
        将SFT模型的权重迁移到PPO模型

        Args:
            ppo_model: PPO模型
            sft_model_path: SFT模型路径

        Returns:
            初始化后的PPO模型
        """
        print(f"\n🔄 Loading SFT weights to initialize PPO policy...")

        # 加载SFT模型
        sft_model = BehaviorCloner(
            obs_dim=49,
            action_dim=16,
            hidden_dims=[256, 256, 128]
        )
        sft_model.load(sft_model_path)

        # 获取SFT模型的state_dict
        sft_state_dict = sft_model.policy_net.state_dict()

        # 获取PPO策略网络
        ppo_policy = ppo_model.policy

        # 尝试迁移权重到PPO的actor网络
        # PPO的mlp_extractor包含policy和value网络
        try:
            # 映射SFT层到PPO层
            # SFT: 0.weight/bias (Linear 49->256)
            #      3.weight/bias (Linear 256->256)
            #      6.weight/bias (Linear 256->128)
            #      9.weight/bias (Linear 128->16)

            # PPO mlp_extractor.policy_net结构类似，尝试匹配
            ppo_state = ppo_policy.mlp_extractor.policy_net.state_dict()

            # 手动映射权重
            weight_mapping = [
                ('0.weight', '0.weight'),
                ('0.bias', '0.bias'),
                ('3.weight', '2.weight'),
                ('3.bias', '2.bias'),
                ('6.weight', '4.weight'),
                ('6.bias', '4.bias'),
            ]

            for sft_key, ppo_key in weight_mapping:
                if sft_key in sft_state_dict and ppo_key in ppo_state:
                    ppo_state[ppo_key] = sft_state_dict[sft_key].clone()
                    print(f"  ✅ Transferred: {sft_key} -> {ppo_key}")

            ppo_policy.mlp_extractor.policy_net.load_state_dict(ppo_state)

            # 也初始化action网络的最后一层
            if '9.weight' in sft_state_dict:
                action_net_state = ppo_policy.action_net.state_dict()
                action_net_state['weight'] = sft_state_dict['9.weight'].clone()
                action_net_state['bias'] = sft_state_dict['9.bias'].clone()
                ppo_policy.action_net.load_state_dict(action_net_state)
                print(f"  ✅ Transferred final layer to action_net")

            print(f"✅ SFT weights successfully transferred to PPO!")

        except Exception as e:
            print(f"⚠️  Partial weight transfer: {e}")
            print(f"   PPO will use partially initialized weights")

        return ppo_model


class TensorboardCallback(BaseCallback):
    """自定义回调：记录竞速模式的额外指标到Tensorboard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_kills = []
        self.episode_max_radius = []
        self.episode_wins = 0
        self.episode_losses = 0
        self.episode_scores = []

    def _on_step(self) -> bool:
        # 获取环境信息
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'kills' in info:
                    self.logger.record('custom/kills', info['kills'])
                    self.logger.record('custom/max_radius', info['max_radius'])
                    self.logger.record('custom/alive_opponents', info['alive_opponents'])

                # 竞速模式新增指标
                if 'player_score' in info:
                    self.logger.record('race/player_score', info['player_score'])
                    self.logger.record('race/max_opponent_score', info['max_opponent_score'])
                    self.logger.record('race/score_gap', info['score_gap'])
                    self.logger.record('race/progress', info['player_score'] / info['target_score'])

                # 记录胜负
                if 'race_finished' in info and info['race_finished']:
                    if info['winner'] == 'player':
                        self.episode_wins += 1
                        self.logger.record('race/total_wins', self.episode_wins)
                    else:
                        self.episode_losses += 1
                        self.logger.record('race/total_losses', self.episode_losses)

                    win_rate = self.episode_wins / max(1, self.episode_wins + self.episode_losses)
                    self.logger.record('race/win_rate', win_rate)
                    self.logger.record('race/completion_steps', info['step'])

        return True


def make_env(rank: int, sft_model_path: str, config: dict):
    """创建环境的工厂函数（用于并行）- 竞速模式"""
    def _init():
        env = MultiAgentAgarioEnv(
            world_size=config['world_size'],
            food_count=config['food_count'],
            num_opponents=config['num_opponents'],
            sft_model_path=sft_model_path,
            max_steps=config['max_steps'],
            difficulty=config['difficulty'],
            target_score=config['target_score']  # 竞速目标分数
        )
        env = Monitor(env)
        return env
    return _init


def train_sft_post_training(
    sft_model_path: str = "models/sft_giant/best_model.pt",
    output_dir: str = "models/sft_post_training",
    total_timesteps: int = 500000,
    num_envs: int = 8,  # M3 Mac Studio可以轻松运行8个环境
    difficulty: str = "medium",
    learning_rate: float = 1e-4,
    save_freq: int = 10000,
    eval_freq: int = 5000,
    device: str = "auto",
    target_score: int = 300  # 竞速目标分数
):
    """
    SFT后训练主函数 - 竞速模式

    Args:
        sft_model_path: SFT模型路径
        output_dir: 输出目录
        total_timesteps: 总训练步数
        num_envs: 并行环境数（建议M3 Mac: 6-8个）
        difficulty: 难度
        learning_rate: 学习率（比初始训练小）
        save_freq: 保存频率
        eval_freq: 评估频率
        device: 设备（M3会自动使用MPS）
        target_score: 竞速目标分数（默认300）
    """
    print("=" * 80)
    print("🏁 SFT Post-Training - Race Mode: First to 300 Points!")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  SFT模型: {sft_model_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  并行环境数: {num_envs}")
    print(f"  难度: {difficulty}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    print(f"  🎯 竞速目标: {target_score} 分")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    # 环境配置 - 竞速模式
    env_config = {
        'world_size': 6000,
        'food_count': 1000,
        'num_opponents': 5,  # 5个SFT对手
        'max_steps': 5000,
        'difficulty': difficulty,
        'target_score': target_score  # 竞速目标分数
    }

    # 保存配置
    config = {
        'sft_model_path': sft_model_path,
        'total_timesteps': total_timesteps,
        'num_envs': num_envs,
        'env_config': env_config,
        'learning_rate': learning_rate,
        'device': device,
        'start_time': datetime.now().isoformat()
    }

    with open(f"{output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # 创建并行环境
    print(f"\n📦 Creating {num_envs} parallel environments...")

    # 使用DummyVecEnv（单进程但可以运行多个环境）
    # 96GB内存完全够用，即使单进程也能高效运行
    print(f"   Using DummyVecEnv (single process, {num_envs} environments)")
    print(f"   With 96GB RAM, this is efficient and stable!")
    env = DummyVecEnv([
        make_env(i, sft_model_path, env_config)
        for i in range(num_envs)
    ])

    print(f"✅ Environments created!")

    # 创建评估环境
    print(f"\n📊 Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(0, sft_model_path, env_config)
    ])

    # 创建PPO模型
    print(f"\n🤖 Creating PPO model...")

    # M3 Mac优化的PPO配置
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,  # 增加n_steps以更好地利用并行环境
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 鼓励探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],  # 匹配SFT架构
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=None,  # 暂时禁用tensorboard
        device=device
    )

    print(f"✅ PPO model created!")

    # 从SFT初始化权重
    print(f"\n🎯 Initializing PPO from SFT weights...")
    model = SFTInitPolicy.load_sft_weights_to_ppo(model, sft_model_path)

    # 创建回调
    callbacks = []

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,  # 考虑并行环境
        save_path=f"{output_dir}/checkpoints",
        name_prefix="sft_post_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=eval_freq // num_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Tensorboard回调
    tb_callback = TensorboardCallback()
    callbacks.append(tb_callback)

    # 开始训练
    print(f"\n" + "=" * 80)
    print(f"🏁 Starting Race Training...")
    print(f"=" * 80)
    print(f"\n🎯 竞速规则:")
    print(f"  - 玩家 vs {env_config['num_opponents']}个SFT对手")
    print(f"  - 吃食物 +1 分，击杀对手 +30 分")
    print(f"  - 最快达到 {target_score} 分获胜")
    print(f"  - 获胜奖励 +500，失败惩罚 -200")
    print(f"\n预计训练时间: {total_timesteps / (num_envs * 1000):.0f}-{total_timesteps / (num_envs * 500):.0f} 分钟\n")

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 保存最终模型
    final_model_path = f"{output_dir}/final_model"
    model.save(final_model_path)

    print(f"\n" + "=" * 80)
    print(f"✅ Post-Training Complete!")
    print(f"=" * 80)
    print(f"  训练时长: {duration / 60:.1f} 分钟")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  吞吐量: {total_timesteps / duration:.0f} steps/sec")
    print(f"\n模型保存位置:")
    print(f"  最佳模型: {output_dir}/best_model.zip")
    print(f"  最终模型: {final_model_path}.zip")
    print(f"  检查点: {output_dir}/checkpoints/")
    print(f"\nTensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/tensorboard")
    print(f"=" * 80)

    env.close()
    eval_env.close()

    return model


def evaluate_model(
    model_path: str,
    sft_model_path: str = "models/sft_giant/best_model.pt",
    num_episodes: int = 10,
    difficulty: str = "hard",
    target_score: int = 300
):
    """评估训练后的模型 - 竞速模式"""
    print("=" * 80)
    print("🏁 Evaluating Post-Trained Model - Race Mode")
    print("=" * 80)

    # 创建评估环境（高难度）
    env = MultiAgentAgarioEnv(
        world_size=6000,
        food_count=1000,
        num_opponents=7,  # 更多对手
        sft_model_path=sft_model_path,
        difficulty=difficulty,
        target_score=target_score
    )

    # 加载模型
    model = PPO.load(model_path)

    results = {
        'scores': [],
        'kills': [],
        'deaths': [],
        'max_radius': [],
        'survived': 0,
        'wins': 0,
        'losses': 0,
        'player_scores': [],
        'completion_steps': []
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        results['scores'].append(episode_reward)
        results['kills'].append(info['kills'])
        results['deaths'].append(info['deaths'])
        results['max_radius'].append(info['max_radius'])
        results['player_scores'].append(info['player_score'])

        if not info['deaths']:
            results['survived'] += 1

        # 竞速模式结果
        if info['race_finished']:
            if info['winner'] == 'player':
                results['wins'] += 1
                results['completion_steps'].append(info['step'])

                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"🏆 WIN! Score={info['player_score']}, "
                      f"Steps={info['step']}, "
                      f"Kills={info['kills']}")
            else:
                results['losses'] += 1
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"❌ LOSS. Score={info['player_score']}, "
                      f"Winner={info['winner']}, "
                      f"Kills={info['kills']}")
        else:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"⏱️  TIMEOUT. Score={info['player_score']}/{target_score}, "
                  f"Kills={info['kills']}")

    print(f"\n" + "=" * 80)
    print(f"🏁 Race Mode Evaluation Results ({num_episodes} episodes):")
    print(f"=" * 80)
    print(f"  🏆 Wins: {results['wins']}/{num_episodes} ({results['wins']/num_episodes*100:.1f}%)")
    print(f"  ❌ Losses: {results['losses']}/{num_episodes}")
    print(f"  ⏱️  Timeouts: {num_episodes - results['wins'] - results['losses']}")
    if results['completion_steps']:
        print(f"  ⚡ Avg Steps to Win: {np.mean(results['completion_steps']):.0f}")
    print(f"\n  📊 Performance:")
    print(f"  Average Player Score: {np.mean(results['player_scores']):.1f} ± {np.std(results['player_scores']):.1f}")
    print(f"  Average Reward: {np.mean(results['scores']):.1f} ± {np.std(results['scores']):.1f}")
    print(f"  Average Kills: {np.mean(results['kills']):.1f}")
    print(f"  Average Max Radius: {np.mean(results['max_radius']):.1f}")
    print(f"  Survival Rate: {results['survived'] / num_episodes * 100:.1f}%")
    print(f"=" * 80)

    env.close()
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SFT Post-Training')
    parser.add_argument('--sft-model', type=str,
                       default='models/sft_giant/best_model.pt',
                       help='SFT model path')
    parser.add_argument('--output-dir', type=str,
                       default='models/sft_post_training',
                       help='Output directory')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total timesteps')
    parser.add_argument('--num-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--eval', type=str, default=None,
                       help='Evaluate model instead of training')
    parser.add_argument('--target-score', type=int, default=300,
                       help='Target score to win the race')

    args = parser.parse_args()

    if args.eval:
        # 评估模式
        evaluate_model(
            model_path=args.eval,
            num_episodes=10,
            difficulty='hard',
            target_score=args.target_score
        )
    else:
        # 训练模式
        train_sft_post_training(
            sft_model_path=args.sft_model,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            difficulty=args.difficulty,
            learning_rate=args.lr,
            target_score=args.target_score
        )
