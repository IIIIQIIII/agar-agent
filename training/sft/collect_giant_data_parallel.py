#!/usr/bin/env python3
"""
Giant Agent 并行数据收集模块
使用多进程加速数据收集
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from agario_core.agents.giant_agent import GiantAgent
from agario_core.envs.rl_env import AgarioEnv
import pickle
from multiprocessing import Pool, cpu_count, Manager
import argparse


def collect_single_episode(args):
    """
    收集单个对局（用于多进程）

    Args:
        args: (episode_idx, score_threshold, data_dir, max_steps)

    Returns:
        (success, episode_data)
    """
    episode_idx, score_threshold, data_dir, max_steps = args

    giant_agent = GiantAgent(world_size=4000)
    env = AgarioEnv(world_size=4000, food_count=500, enemy_count=30)

    episode_data = {
        'episode_id': episode_idx,
        'timestamp': datetime.now().isoformat(),
        'observations': [],
        'actions': [],
        'rewards': [],
        'giant_decisions': [],
        'score_history': [],
        'radius_history': [],
        'final_score': 0,
        'final_radius': 0,
        'steps': 0,
        'success': False
    }

    obs, info = env.reset()
    done = False
    step_count = 0
    max_score = 0

    def extract_game_state_from_env(env):
        return {
            'player_x': env.player['x'],
            'player_y': env.player['y'],
            'player_radius': env.player['radius'],
            'foods': env.foods.copy(),
            'enemies': env.enemies.copy()
        }

    def convert_giant_action_to_discrete(giant_action):
        angle = giant_action['angle']
        should_split = giant_action['split']
        direction_idx = int((angle + 22.5) % 360 / 45) % 8
        if should_split:
            return direction_idx + 8
        else:
            return direction_idx

    while not done and step_count < max_steps:
        game_state = extract_game_state_from_env(env)
        giant_action = giant_agent.get_action(game_state)
        discrete_action = convert_giant_action_to_discrete(giant_action)

        next_obs, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated

        episode_data['observations'].append(obs.copy())
        episode_data['actions'].append(discrete_action)
        episode_data['rewards'].append(reward)
        episode_data['giant_decisions'].append(giant_action)
        episode_data['score_history'].append(env.score)
        episode_data['radius_history'].append(env.player['radius'])

        if env.score > max_score:
            max_score = env.score

        obs = next_obs
        step_count += 1

    episode_data['final_score'] = env.score
    episode_data['final_radius'] = env.player['radius']
    episode_data['max_score'] = max_score
    episode_data['steps'] = step_count
    episode_data['success'] = max_score >= score_threshold

    env.close()

    # 如果成功，保存数据
    if episode_data['success']:
        os.makedirs(f"{data_dir}/episodes", exist_ok=True)
        filename = f"{data_dir}/episodes/episode_{episode_idx:04d}_score{episode_data['final_score']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(episode_data, f)

        metadata = {
            'episode_id': episode_data['episode_id'],
            'timestamp': episode_data['timestamp'],
            'final_score': episode_data['final_score'],
            'max_score': episode_data['max_score'],
            'final_radius': episode_data['final_radius'],
            'steps': episode_data['steps'],
            'success': episode_data['success'],
            'data_points': len(episode_data['observations'])
        }
        metadata_file = f"{data_dir}/episodes/episode_{episode_idx:04d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    return (episode_data['success'], episode_data['max_score'], step_count)


def collect_parallel(num_episodes: int = 300,
                     score_threshold: int = 300,
                     data_dir: str = "giant_data",
                     num_workers: int = None,
                     max_steps: int = 2000) -> Dict:
    """
    并行收集数据

    Args:
        num_episodes: 需要收集的成功对局数
        score_threshold: 分数阈值
        data_dir: 数据保存目录
        num_workers: 工作进程数（默认为CPU核心数-2）
        max_steps: 每局最大步数

    Returns:
        collection_summary: 收集统计
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # 保留2个核心给系统

    print(f"\n{'='*70}")
    print(f"🚀 Giant Agent 并行数据收集")
    print(f"{'='*70}")
    print(f"目标: 收集 {num_episodes} 个分数 ≥ {score_threshold} 的对局")
    print(f"并行进程数: {num_workers}")
    print(f"每局最大步数: {max_steps}")
    print(f"{'='*70}\n")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/episodes", exist_ok=True)

    start_time = datetime.now()

    successful_count = 0
    total_attempts = 0
    batch_size = num_workers * 2  # 每批运行的对局数

    # 使用进度管理器
    with Manager() as manager:
        while successful_count < num_episodes:
            # 准备这一批的参数
            tasks = []
            for i in range(batch_size):
                episode_idx = total_attempts + i + 1
                tasks.append((episode_idx, score_threshold, data_dir, max_steps))

            # 并行执行
            print(f"\n🔄 批次 {total_attempts//batch_size + 1}: 运行 {batch_size} 个对局...")
            with Pool(processes=num_workers) as pool:
                results = pool.map(collect_single_episode, tasks)

            # 统计结果
            batch_successful = sum(1 for success, _, _ in results if success)
            successful_count += batch_successful
            total_attempts += batch_size

            # 打印批次结果
            for i, (success, score, steps) in enumerate(results):
                status = "✅" if success else "❌"
                print(f"  {status} Episode {total_attempts - batch_size + i + 1}: "
                      f"Score={score:5d}, Steps={steps:4d}")

            print(f"\n📊 进度: {successful_count}/{num_episodes} 成功 "
                  f"({total_attempts} 次尝试, 成功率 {successful_count/total_attempts*100:.1f}%)")

            if successful_count >= num_episodes:
                break

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 读取所有成功对局的统计
    episode_files = [f for f in os.listdir(f"{data_dir}/episodes") if f.endswith('.pkl')]
    total_data_points = 0
    all_scores = []
    all_steps = []

    for episode_file in episode_files[:num_episodes]:
        with open(f"{data_dir}/episodes/{episode_file}", 'rb') as f:
            episode_data = pickle.load(f)
            total_data_points += len(episode_data['observations'])
            all_scores.append(episode_data['max_score'])
            all_steps.append(episode_data['steps'])

    # 统计信息
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'duration_minutes': duration / 60,
        'total_attempts': total_attempts,
        'successful_episodes': successful_count,
        'success_rate': successful_count / total_attempts,
        'score_threshold': score_threshold,
        'total_data_points': total_data_points,
        'avg_episode_length': np.mean(all_steps) if all_steps else 0,
        'avg_score': np.mean(all_scores) if all_scores else 0,
        'max_score_achieved': max(all_scores) if all_scores else 0,
        'num_workers': num_workers,
        'episodes_per_minute': successful_count / (duration / 60) if duration > 0 else 0,
    }

    # 保存统计
    summary_file = f"{data_dir}/collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印总结
    print(f"\n{'='*70}")
    print(f"🎉 数据收集完成！")
    print(f"{'='*70}")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功对局数: {successful_count}")
    print(f"  成功率: {summary['success_rate']*100:.1f}%")
    print(f"  总数据点数: {summary['total_data_points']:,}")
    print(f"  平均对局长度: {summary['avg_episode_length']:.0f} 步")
    print(f"  平均分数: {summary['avg_score']:.0f}")
    print(f"  最高分数: {summary['max_score_achieved']:.0f}")
    print(f"  总耗时: {summary['duration_minutes']:.1f} 分钟")
    print(f"  收集速度: {summary['episodes_per_minute']:.1f} 对局/分钟")
    print(f"  并行加速: {num_workers}x 进程")
    print(f"{'='*70}")
    print(f"\n✅ 数据已保存到: {data_dir}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description='并行收集 Giant Agent 游戏数据')
    parser.add_argument('--episodes', type=int, default=300,
                       help='需要收集的成功对局数 (默认: 300)')
    parser.add_argument('--score-threshold', type=int, default=300,
                       help='分数阈值 (默认: 300)')
    parser.add_argument('--data-dir', type=str, default='giant_data',
                       help='数据保存目录 (默认: giant_data)')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数 (默认: CPU核心数-2)')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='每局最大步数 (默认: 2000)')

    args = parser.parse_args()

    try:
        summary = collect_parallel(
            num_episodes=args.episodes,
            score_threshold=args.score_threshold,
            data_dir=args.data_dir,
            num_workers=args.workers,
            max_steps=args.max_steps
        )

        print("\n🎉 数据收集任务完成！")
        print(f"可以使用 prepare_sft_dataset.py 处理数据用于训练。")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        print("可以使用已收集的数据进行训练")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
