#!/usr/bin/env python3
"""
测试 SFT 训练的模型
在 Agar.io 环境中评估模型性能
"""

import numpy as np
import torch
import argparse
from agario_core.envs.rl_env import AgarioEnv
from agario_core.models.behavior_cloner import BehaviorCloner


def test_model(model_path: str,
               num_episodes: int = 10,
               render: bool = False,
               max_steps: int = 5000) -> dict:
    """
    测试训练好的模型

    Args:
        model_path: 模型路径
        num_episodes: 测试对局数
        render: 是否渲染
        max_steps: 每局最大步数

    Returns:
        results: 测试结果统计
    """
    print(f"\n{'='*70}")
    print(f"测试 SFT 模型")
    print(f"{'='*70}\n")

    # 加载模型
    print(f"加载模型: {model_path}")
    model = BehaviorCloner(obs_dim=49, action_dim=16)
    model.load(model_path)
    print("✅ 模型加载成功\n")

    # 创建环境
    env = AgarioEnv(world_size=4000, food_count=500, enemy_count=30)

    # 测试统计
    episode_scores = []
    episode_steps = []
    episode_max_radius = []
    success_count = 0  # 分数 >= 2000

    print(f"开始测试 {num_episodes} 个对局...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        max_radius = env.player['radius']

        while not done and step_count < max_steps:
            # 使用模型预测动作
            action = model.predict(obs)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 更新统计
            max_radius = max(max_radius, env.player['radius'])
            step_count += 1

        # 记录结果
        final_score = env.score
        episode_scores.append(final_score)
        episode_steps.append(step_count)
        episode_max_radius.append(max_radius)

        if final_score >= 2000:
            success_count += 1

        # 打印单局结果
        status = "✅" if final_score >= 2000 else "  "
        print(f"{status} Episode {episode+1:2d}: "
              f"Score={final_score:5d}, "
              f"Steps={step_count:4d}, "
              f"Max Radius={max_radius:5.1f}")

    env.close()

    # 计算统计
    results = {
        'num_episodes': num_episodes,
        'mean_score': np.mean(episode_scores),
        'std_score': np.std(episode_scores),
        'max_score': np.max(episode_scores),
        'min_score': np.min(episode_scores),
        'mean_steps': np.mean(episode_steps),
        'mean_max_radius': np.mean(episode_max_radius),
        'success_rate': success_count / num_episodes,
        'success_count': success_count,
        'scores': episode_scores
    }

    # 打印总结
    print(f"\n{'='*70}")
    print(f"测试结果总结")
    print(f"{'='*70}")
    print(f"  测试对局数: {num_episodes}")
    print(f"  平均分数: {results['mean_score']:.1f} ± {results['std_score']:.1f}")
    print(f"  最高分数: {results['max_score']:.0f}")
    print(f"  最低分数: {results['min_score']:.0f}")
    print(f"  平均步数: {results['mean_steps']:.0f}")
    print(f"  平均最大半径: {results['mean_max_radius']:.1f}")
    print(f"  成功率 (≥2000): {results['success_rate']*100:.1f}% ({success_count}/{num_episodes})")
    print(f"{'='*70}\n")

    return results


def compare_with_giant(model_path: str,
                      num_episodes: int = 5) -> dict:
    """
    比较 SFT 模型和 Giant Agent 的性能

    Args:
        model_path: SFT模型路径
        num_episodes: 对比对局数

    Returns:
        comparison: 对比结果
    """
    from agario_core.agents.giant_agent import GiantAgent
    from training.sft.collect_giant_data import GiantDataCollector

    print(f"\n{'='*70}")
    print(f"SFT 模型 vs Giant Agent 性能对比")
    print(f"{'='*70}\n")

    # 测试 SFT 模型
    print("测试 SFT 模型...")
    sft_results = test_model(model_path, num_episodes=num_episodes)

    # 测试 Giant Agent
    print("\n测试 Giant Agent...")
    collector = GiantDataCollector(data_dir="temp_comparison")
    giant_scores = []

    for i in range(num_episodes):
        success, episode_data = collector.collect_episode(i+1, verbose=False)
        giant_scores.append(episode_data['max_score'])
        print(f"  Episode {i+1:2d}: Score={episode_data['max_score']:5d}")

    # 比较
    comparison = {
        'sft_mean_score': sft_results['mean_score'],
        'giant_mean_score': np.mean(giant_scores),
        'sft_max_score': sft_results['max_score'],
        'giant_max_score': np.max(giant_scores),
        'sft_success_rate': sft_results['success_rate'],
        'giant_success_rate': sum(s >= 2000 for s in giant_scores) / num_episodes,
    }

    print(f"\n{'='*70}")
    print(f"对比结果")
    print(f"{'='*70}")
    print(f"  平均分数:")
    print(f"    SFT 模型: {comparison['sft_mean_score']:.1f}")
    print(f"    Giant Agent: {comparison['giant_mean_score']:.1f}")
    print(f"    差距: {comparison['sft_mean_score'] - comparison['giant_mean_score']:.1f}")
    print(f"\n  最高分数:")
    print(f"    SFT 模型: {comparison['sft_max_score']:.0f}")
    print(f"    Giant Agent: {comparison['giant_max_score']:.0f}")
    print(f"\n  成功率 (≥2000):")
    print(f"    SFT 模型: {comparison['sft_success_rate']*100:.1f}%")
    print(f"    Giant Agent: {comparison['giant_success_rate']*100:.1f}%")
    print(f"{'='*70}\n")

    return comparison


def main():
    parser = argparse.ArgumentParser(description='测试SFT模型')
    parser.add_argument('--model-path', type=str,
                       default='models/sft_giant/best_model.pt',
                       help='模型路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='测试对局数')
    parser.add_argument('--compare', action='store_true',
                       help='与Giant Agent对比')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='每局最大步数')

    args = parser.parse_args()

    try:
        if args.compare:
            # 对比测试
            comparison = compare_with_giant(args.model_path, num_episodes=args.episodes)
        else:
            # 单独测试
            results = test_model(args.model_path,
                               num_episodes=args.episodes,
                               max_steps=args.max_steps)

        print("✅ 测试完成！")

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print(f"请确保模型文件存在: {args.model_path}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
