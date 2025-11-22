"""
巨无霸智能体评估工具
Giant Agent Evaluation Tool

用于测试和评估数学建模智能体的性能
"""

import numpy as np
import json
from agario_core.agents.giant_agent import GiantAgent
from typing import List, Dict
import time


class GameSimulator:
    """简化的游戏模拟器，用于评估智能体"""

    def __init__(self, world_size=4000, food_count=500, enemy_count=30):
        self.world_size = world_size
        self.food_count = food_count
        self.enemy_count = enemy_count
        self.reset()

    def reset(self):
        """重置游戏状态"""
        # 玩家初始状态
        self.player = {
            'x': self.world_size / 2,
            'y': self.world_size / 2,
            'radius': 20,
            'score': 0,
            'alive': True
        }

        # 生成食物
        self.foods = []
        for _ in range(self.food_count):
            self.foods.append({
                'x': np.random.uniform(0, self.world_size),
                'y': np.random.uniform(0, self.world_size),
                'radius': np.random.uniform(3, 6),
            })

        # 生成敌人
        self.enemies = []
        for _ in range(self.enemy_count):
            radius = np.random.uniform(15, 50)
            self.enemies.append({
                'x': np.random.uniform(0, self.world_size),
                'y': np.random.uniform(0, self.world_size),
                'radius': radius,
                'dx': np.random.uniform(-2, 2),
                'dy': np.random.uniform(-2, 2),
            })

        self.steps = 0
        self.max_steps = 5000

        return self.get_state()

    def get_state(self) -> Dict:
        """获取当前游戏状态"""
        return {
            'player_x': self.player['x'],
            'player_y': self.player['y'],
            'player_radius': self.player['radius'],
            'foods': self.foods.copy(),
            'enemies': self.enemies.copy()
        }

    def step(self, action: Dict) -> tuple:
        """
        执行一步游戏

        Args:
            action: {'angle': float, 'split': bool}

        Returns:
            (state, reward, done, info)
        """
        self.steps += 1

        # 移动玩家
        angle_rad = np.radians(action['angle'])
        speed = 30 / (self.player['radius'] ** 0.4)
        self.player['x'] += np.cos(angle_rad) * speed
        self.player['y'] += np.sin(angle_rad) * speed

        # 边界限制
        self.player['x'] = np.clip(
            self.player['x'],
            self.player['radius'],
            self.world_size - self.player['radius']
        )
        self.player['y'] = np.clip(
            self.player['y'],
            self.player['radius'],
            self.world_size - self.player['radius']
        )

        # 移动敌人
        for enemy in self.enemies:
            enemy['x'] += enemy['dx']
            enemy['y'] += enemy['dy']

            # 边界反弹
            if enemy['x'] - enemy['radius'] < 0 or enemy['x'] + enemy['radius'] > self.world_size:
                enemy['dx'] *= -1
            if enemy['y'] - enemy['radius'] < 0 or enemy['y'] + enemy['radius'] > self.world_size:
                enemy['dy'] *= -1

        # 计算奖励
        reward = 0
        initial_radius = self.player['radius']

        # 吃食物
        foods_eaten = 0
        for i in range(len(self.foods) - 1, -1, -1):
            food = self.foods[i]
            dist = np.hypot(food['x'] - self.player['x'], food['y'] - self.player['y'])
            if dist < self.player['radius'] + food['radius']:
                area = np.pi * self.player['radius'] ** 2 + np.pi * food['radius'] ** 2
                self.player['radius'] = np.sqrt(area / np.pi)
                self.player['score'] += 1
                foods_eaten += 1
                reward += 1

                # 重新生成食物
                self.foods[i] = {
                    'x': np.random.uniform(0, self.world_size),
                    'y': np.random.uniform(0, self.world_size),
                    'radius': np.random.uniform(3, 6),
                }

        # 与敌人交互
        enemies_eaten = 0
        for enemy in self.enemies:
            dist = np.hypot(enemy['x'] - self.player['x'], enemy['y'] - self.player['y'])
            if dist < self.player['radius'] + enemy['radius']:
                if self.player['radius'] > enemy['radius'] * 1.1:
                    # 吃掉敌人
                    area = np.pi * self.player['radius'] ** 2 + np.pi * enemy['radius'] ** 2
                    self.player['radius'] = np.sqrt(area / np.pi)
                    enemy_score = int(enemy['radius'])
                    self.player['score'] += enemy_score
                    enemies_eaten += 1
                    reward += 10 + enemy_score * 0.2

                    # 重新生成敌人
                    enemy['radius'] = np.random.uniform(15, 50)
                    enemy['x'] = np.random.uniform(0, self.world_size)
                    enemy['y'] = np.random.uniform(0, self.world_size)
                elif enemy['radius'] > self.player['radius'] * 1.1:
                    # 被吃掉
                    self.player['alive'] = False
                    reward -= 100

        # 成长奖励
        growth = self.player['radius'] - initial_radius
        if growth > 0:
            reward += growth * 5

        # 存活奖励
        reward += 0.01

        # 检查游戏结束
        done = not self.player['alive'] or self.steps >= self.max_steps

        info = {
            'score': self.player['score'],
            'radius': self.player['radius'],
            'steps': self.steps,
            'foods_eaten': foods_eaten,
            'enemies_eaten': enemies_eaten
        }

        return self.get_state(), reward, done, info


class GiantAgentEvaluator:
    """巨无霸智能体评估器"""

    def __init__(self):
        self.agent = GiantAgent()
        self.simulator = GameSimulator()

    def run_episode(self, max_steps=5000, verbose=False) -> Dict:
        """运行一个完整的游戏回合"""
        state = self.simulator.reset()
        total_reward = 0
        episode_info = {
            'steps': 0,
            'total_reward': 0,
            'final_score': 0,
            'final_radius': 0,
            'max_radius': 0,
            'foods_eaten': 0,
            'enemies_eaten': 0,
            'phases': {'early': 0, 'mid': 0, 'late': 0},
            'split_count': 0,
            'survived': False
        }

        for step in range(max_steps):
            # 获取智能体决策
            action = self.agent.get_action(state)

            # 记录阶段分布
            phase = action['debug_info']['phase']
            episode_info['phases'][phase] += 1

            # 记录分裂次数
            if action['split']:
                episode_info['split_count'] += 1

            # 执行动作
            state, reward, done, info = self.simulator.step(action)

            total_reward += reward
            episode_info['steps'] = step + 1
            episode_info['foods_eaten'] += info.get('foods_eaten', 0)
            episode_info['enemies_eaten'] += info.get('enemies_eaten', 0)
            episode_info['max_radius'] = max(episode_info['max_radius'], info['radius'])

            if verbose and step % 100 == 0:
                print(f"Step {step}: Radius={info['radius']:.1f}, Score={info['score']}, Phase={phase}")

            if done:
                episode_info['final_score'] = info['score']
                episode_info['final_radius'] = info['radius']
                episode_info['survived'] = self.simulator.player['alive']
                episode_info['total_reward'] = total_reward
                break

        return episode_info

    def evaluate(self, num_episodes=10, verbose=True) -> Dict:
        """评估智能体性能"""
        print("=" * 70)
        print("🎯 开始评估巨无霸智能体")
        print("=" * 70)
        print(f"\n评估回合数: {num_episodes}\n")

        all_results = []

        for episode in range(num_episodes):
            if verbose:
                print(f"\n{'─' * 70}")
                print(f"📊 回合 {episode + 1}/{num_episodes}")
                print('─' * 70)

            start_time = time.time()
            result = self.run_episode(verbose=False)
            elapsed_time = time.time() - start_time

            all_results.append(result)

            if verbose:
                print(f"✅ 完成 (耗时: {elapsed_time:.1f}秒)")
                print(f"  - 最终分数: {result['final_score']}")
                print(f"  - 最终半径: {result['final_radius']:.1f}")
                print(f"  - 最大半径: {result['max_radius']:.1f}")
                print(f"  - 存活步数: {result['steps']}")
                print(f"  - 是否存活: {'✓' if result['survived'] else '✗'}")
                print(f"  - 吃掉食物: {result['foods_eaten']}")
                print(f"  - 吃掉敌人: {result['enemies_eaten']}")
                print(f"  - 分裂次数: {result['split_count']}")

        # 统计分析
        print("\n" + "=" * 70)
        print("📈 评估结果统计")
        print("=" * 70)

        final_scores = [r['final_score'] for r in all_results]
        final_radii = [r['final_radius'] for r in all_results]
        max_radii = [r['max_radius'] for r in all_results]
        survival_rate = sum(1 for r in all_results if r['survived']) / num_episodes
        total_foods = sum(r['foods_eaten'] for r in all_results)
        total_enemies = sum(r['enemies_eaten'] for r in all_results)
        total_splits = sum(r['split_count'] for r in all_results)

        stats = {
            'num_episodes': num_episodes,
            'mean_score': np.mean(final_scores),
            'std_score': np.std(final_scores),
            'max_score': np.max(final_scores),
            'min_score': np.min(final_scores),
            'mean_final_radius': np.mean(final_radii),
            'mean_max_radius': np.mean(max_radii),
            'std_max_radius': np.std(max_radii),
            'survival_rate': survival_rate,
            'total_foods_eaten': total_foods,
            'total_enemies_eaten': total_enemies,
            'total_splits': total_splits,
            'avg_foods_per_episode': total_foods / num_episodes,
            'avg_enemies_per_episode': total_enemies / num_episodes,
            'episodes': all_results
        }

        print(f"\n🏆 平均分数: {stats['mean_score']:.1f} ± {stats['std_score']:.1f}")
        print(f"📏 平均最终半径: {stats['mean_final_radius']:.1f}")
        print(f"📏 平均最大半径: {stats['mean_max_radius']:.1f} ± {stats['std_max_radius']:.1f}")
        print(f"💚 存活率: {stats['survival_rate'] * 100:.1f}%")
        print(f"🍎 平均每局吃食物: {stats['avg_foods_per_episode']:.1f}")
        print(f"👾 平均每局吃敌人: {stats['avg_enemies_per_episode']:.1f}")
        print(f"✂️  总分裂次数: {stats['total_splits']}")

        # 成长为巨无霸的统计
        giant_threshold = 60  # 定义巨无霸的阈值
        giant_count = sum(1 for r in max_radii if r >= giant_threshold)
        print(f"\n🦖 成长为巨无霸次数 (半径≥{giant_threshold}): {giant_count}/{num_episodes} ({giant_count/num_episodes*100:.1f}%)")

        print("\n" + "=" * 70)

        return stats


def save_results(stats: Dict, filename='evaluation_results.json'):
    """保存评估结果到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存到: {filename}")


if __name__ == '__main__':
    evaluator = GiantAgentEvaluator()

    # 运行评估
    results = evaluator.evaluate(num_episodes=10, verbose=True)

    # 保存结果
    save_results(results)

    # 额外统计
    print("\n" + "=" * 70)
    print("🎮 巨无霸智能体评估完成！")
    print("=" * 70)
