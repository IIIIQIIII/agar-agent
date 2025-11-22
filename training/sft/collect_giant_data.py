#!/usr/bin/env python3
"""
Giant Agent 数据收集模块
收集突破2000分的对局数据，用于SFT训练

数据格式：
- 状态 (observation)
- 动作 (action)
- 回合信息
"""

import numpy as np
import json
import pickle
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
from typing import List, Dict, Tuple
from agario_core.agents.giant_agent import GiantAgent
from agario_core.envs.rl_env import AgarioEnv


class GiantDataCollector:
    """收集 Giant Agent 的游戏数据"""

    def __init__(self,
                 data_dir: str = "giant_data",
                 score_threshold: int = 300,
                 max_episodes: int = 100):
        """
        初始化数据收集器

        Args:
            data_dir: 数据保存目录
            score_threshold: 分数阈值（只保存超过此分数的对局）
            max_episodes: 最大收集对局数
        """
        self.data_dir = data_dir
        self.score_threshold = score_threshold
        self.max_episodes = max_episodes

        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/episodes", exist_ok=True)
        os.makedirs(f"{data_dir}/raw", exist_ok=True)

        self.giant_agent = GiantAgent(world_size=4000)
        self.successful_episodes = []

    def convert_giant_state_to_obs(self,
                                   game_state: Dict,
                                   env_obs: np.ndarray) -> np.ndarray:
        """
        将游戏状态转换为与RL环境兼容的观察向量

        这里我们使用env的observation作为基准，因为它已经是标准格式
        """
        return env_obs

    def convert_giant_action_to_discrete(self,
                                        giant_action: Dict) -> int:
        """
        将 Giant Agent 的连续动作转换为离散动作

        Giant Agent 输出: {'angle': 0-360, 'split': bool}
        RL 环境动作: 0-15 (0-7: 8个方向, 8-15: 8个方向+分裂)

        Args:
            giant_action: Giant Agent的动作字典

        Returns:
            离散动作 (0-15)
        """
        angle = giant_action['angle']
        should_split = giant_action['split']

        # 将360度角度映射到8个方向 (0, 45, 90, 135, 180, 225, 270, 315)
        # 每个方向覆盖45度范围
        direction_idx = int((angle + 22.5) % 360 / 45) % 8

        # 如果需要分裂，动作值 +8
        if should_split:
            return direction_idx + 8
        else:
            return direction_idx

    def extract_game_state_from_env(self, env: AgarioEnv) -> Dict:
        """
        从环境中提取游戏状态用于 Giant Agent 决策

        Args:
            env: AgarioEnv 环境实例

        Returns:
            game_state: Giant Agent 需要的状态字典
        """
        return {
            'player_x': env.player['x'],
            'player_y': env.player['y'],
            'player_radius': env.player['radius'],
            'foods': env.foods.copy(),
            'enemies': env.enemies.copy()
        }

    def collect_episode(self,
                       episode_idx: int,
                       verbose: bool = True) -> Tuple[bool, Dict]:
        """
        收集一个完整对局的数据

        Args:
            episode_idx: 对局编号
            verbose: 是否打印详细信息

        Returns:
            (success, episode_data): 是否成功达标，对局数据
        """
        env = AgarioEnv(world_size=4000, food_count=500, enemy_count=30)

        episode_data = {
            'episode_id': episode_idx,
            'timestamp': datetime.now().isoformat(),
            'observations': [],
            'actions': [],
            'rewards': [],
            'giant_decisions': [],  # Giant Agent的原始决策
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

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode_idx} - 开始收集数据")
            print(f"{'='*60}")

        while not done and step_count < env.max_steps:
            # 提取游戏状态
            game_state = self.extract_game_state_from_env(env)

            # Giant Agent 决策
            giant_action = self.giant_agent.get_action(game_state)

            # 转换为离散动作
            discrete_action = self.convert_giant_action_to_discrete(giant_action)

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(discrete_action)
            done = terminated or truncated

            # 记录数据
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(discrete_action)
            episode_data['rewards'].append(reward)
            episode_data['giant_decisions'].append(giant_action)
            episode_data['score_history'].append(env.score)
            episode_data['radius_history'].append(env.player['radius'])

            # 更新最高分
            if env.score > max_score:
                max_score = env.score

            # 打印进度
            if verbose and step_count % 500 == 0:
                print(f"Step {step_count:4d} | "
                      f"Score: {env.score:5d} | "
                      f"Radius: {env.player['radius']:5.1f} | "
                      f"Phase: {giant_action['debug_info']['phase']:5s} | "
                      f"Risk: {giant_action['debug_info']['risk_score']:6.2f}")

            obs = next_obs
            step_count += 1

        # 记录最终结果
        episode_data['final_score'] = env.score
        episode_data['final_radius'] = env.player['radius']
        episode_data['max_score'] = max_score
        episode_data['steps'] = step_count
        episode_data['success'] = max_score >= self.score_threshold

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode_idx} 结束")
            print(f"{'='*60}")
            print(f"  最终分数: {episode_data['final_score']}")
            print(f"  最高分数: {max_score}")
            print(f"  最终半径: {episode_data['final_radius']:.1f}")
            print(f"  总步数: {step_count}")
            print(f"  是否达标 (≥{self.score_threshold}): {'✅ 是' if episode_data['success'] else '❌ 否'}")
            print(f"{'='*60}\n")

        env.close()
        return episode_data['success'], episode_data

    def save_episode_data(self, episode_data: Dict, episode_idx: int):
        """
        保存单个对局数据

        Args:
            episode_data: 对局数据
            episode_idx: 对局编号
        """
        # 保存为pickle格式（高效）
        filename = f"{self.data_dir}/episodes/episode_{episode_idx:04d}_score{episode_data['final_score']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(episode_data, f)

        # 同时保存元数据为JSON（可读）
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

        metadata_file = f"{self.data_dir}/episodes/episode_{episode_idx:04d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def collect_dataset(self,
                       min_successful_episodes: int = 10,
                       max_attempts: int = None,
                       verbose: bool = True) -> Dict:
        """
        收集多个对局的数据，直到获得足够的成功对局

        Args:
            min_successful_episodes: 最少需要的成功对局数
            max_attempts: 最大尝试次数（None表示不限制）
            verbose: 是否打印详细信息

        Returns:
            collection_summary: 收集统计信息
        """
        if max_attempts is None:
            max_attempts = min_successful_episodes * 10  # 默认最多尝试10倍

        successful_count = 0
        total_attempts = 0
        all_episodes = []

        print(f"\n{'='*70}")
        print(f"🎯 开始收集 Giant Agent 数据集")
        print(f"{'='*70}")
        print(f"目标: 收集至少 {min_successful_episodes} 个分数 ≥ {self.score_threshold} 的对局")
        print(f"最大尝试次数: {max_attempts}")
        print(f"{'='*70}\n")

        start_time = datetime.now()

        while successful_count < min_successful_episodes and total_attempts < max_attempts:
            total_attempts += 1

            # 收集一个对局
            success, episode_data = self.collect_episode(total_attempts, verbose=verbose)
            all_episodes.append(episode_data)

            # 如果成功达标，保存数据
            if success:
                successful_count += 1
                self.successful_episodes.append(episode_data)
                self.save_episode_data(episode_data, total_attempts)

                print(f"\n✅ 成功对局 {successful_count}/{min_successful_episodes} 已保存！")
                print(f"   分数: {episode_data['max_score']}, 步数: {episode_data['steps']}\n")
            else:
                print(f"\n❌ 对局未达标 (分数: {episode_data['final_score']} < {self.score_threshold})")
                print(f"   进度: {successful_count}/{min_successful_episodes} 成功, "
                      f"{total_attempts}/{max_attempts} 尝试\n")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 统计信息
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_attempts': total_attempts,
            'successful_episodes': successful_count,
            'success_rate': successful_count / total_attempts if total_attempts > 0 else 0,
            'score_threshold': self.score_threshold,
            'total_data_points': sum(len(ep['observations']) for ep in self.successful_episodes),
            'avg_episode_length': np.mean([ep['steps'] for ep in self.successful_episodes]) if self.successful_episodes else 0,
            'avg_score': np.mean([ep['max_score'] for ep in self.successful_episodes]) if self.successful_episodes else 0,
            'max_score_achieved': max([ep['max_score'] for ep in self.successful_episodes]) if self.successful_episodes else 0,
        }

        # 保存统计信息
        summary_file = f"{self.data_dir}/collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 打印总结
        print(f"\n{'='*70}")
        print(f"📊 数据收集完成！")
        print(f"{'='*70}")
        print(f"  总尝试次数: {total_attempts}")
        print(f"  成功对局数: {successful_count}")
        print(f"  成功率: {summary['success_rate']*100:.1f}%")
        print(f"  总数据点数: {summary['total_data_points']:,}")
        print(f"  平均对局长度: {summary['avg_episode_length']:.0f} 步")
        print(f"  平均分数: {summary['avg_score']:.0f}")
        print(f"  最高分数: {summary['max_score_achieved']:.0f}")
        print(f"  总耗时: {duration/60:.1f} 分钟")
        print(f"{'='*70}")
        print(f"\n✅ 数据已保存到: {self.data_dir}/")
        print(f"   - episodes/: 对局数据文件")
        print(f"   - collection_summary.json: 收集统计\n")

        return summary


def main():
    """主函数：运行数据收集"""
    import argparse

    parser = argparse.ArgumentParser(description='收集 Giant Agent 游戏数据')
    parser.add_argument('--episodes', type=int, default=10,
                       help='需要收集的成功对局数 (默认: 10)')
    parser.add_argument('--score-threshold', type=int, default=300,
                       help='分数阈值 (默认: 300)')
    parser.add_argument('--max-attempts', type=int, default=None,
                       help='最大尝试次数 (默认: episodes * 10)')
    parser.add_argument('--data-dir', type=str, default='giant_data',
                       help='数据保存目录 (默认: giant_data)')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式（减少输出）')

    args = parser.parse_args()

    # 创建收集器
    collector = GiantDataCollector(
        data_dir=args.data_dir,
        score_threshold=args.score_threshold
    )

    # 开始收集
    try:
        summary = collector.collect_dataset(
            min_successful_episodes=args.episodes,
            max_attempts=args.max_attempts,
            verbose=not args.quiet
        )

        print("\n🎉 数据收集任务完成！")
        print(f"可以使用 prepare_sft_dataset.py 处理数据用于训练。")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        print(f"已收集 {len(collector.successful_episodes)} 个成功对局")
        if collector.successful_episodes:
            print("可以使用已收集的数据进行训练")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
