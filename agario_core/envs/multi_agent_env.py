"""
多智能体Agar.io环境 - 用于SFT后训练（竞速模式）
Multi-Agent Agario Environment for SFT Post-Training (Race Mode)

特性:
- 1个可训练的主智能体（从SFT初始化）
- 多个固定的SFT对手智能体
- 竞速模式：最快达到300分获胜
- 优化M3 Mac性能
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Optional
import torch
from agario_core.models.behavior_cloner import BehaviorCloner


class MultiAgentAgarioEnv(gym.Env):
    """
    多智能体Agar.io环境

    主智能体与多个SFT对手竞争
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        world_size: int = 6000,
        food_count: int = 1000,
        num_opponents: int = 5,
        sft_model_path: str = "models/sft_giant/best_model.pt",
        max_steps: int = 5000,
        difficulty: str = "medium",  # easy, medium, hard
        target_score: int = 300  # 竞速目标分数
    ):
        """
        初始化多智能体环境（竞速模式）

        Args:
            world_size: 世界大小（更大的世界）
            food_count: 食物数量（更多食物）
            num_opponents: 对手数量（建议3-8个）
            sft_model_path: SFT模型路径
            max_steps: 最大步数
            difficulty: 难度级别
            target_score: 竞速目标分数（默认300）
        """
        super().__init__()

        self.world_size = world_size
        self.food_count = food_count
        self.num_opponents = num_opponents
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.target_score = target_score  # 竞速目标分数

        # 观察和动作空间（与原环境相同）
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(49,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(16)

        # 加载SFT对手模型
        print(f"Loading SFT model for {num_opponents} opponents...")
        self.sft_model = BehaviorCloner(
            obs_dim=49,
            action_dim=16,
            hidden_dims=[256, 256, 128]
        )
        try:
            self.sft_model.load(sft_model_path)
            print(f"✅ SFT model loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load SFT model: {e}")
            print(f"   Opponents will use random actions")
            self.sft_model = None

        # 难度配置
        self.difficulty_configs = {
            'easy': {
                'opponent_spawn_radius': (15, 25),
                'opponent_skill': 0.7,  # 70% 使用SFT，30%随机
                'food_spawn_rate': 1.2
            },
            'medium': {
                'opponent_spawn_radius': (20, 35),
                'opponent_skill': 0.9,
                'food_spawn_rate': 1.0
            },
            'hard': {
                'opponent_spawn_radius': (25, 45),
                'opponent_skill': 1.0,
                'food_spawn_rate': 0.8
            }
        }
        self.config = self.difficulty_configs[difficulty]

        # 游戏状态
        self.current_step = 0
        self.player = None
        self.opponents = []
        self.foods = []

        # 统计信息
        self.stats = {
            'kills': 0,
            'deaths': 0,
            'food_eaten': 0,
            'max_radius': 0,
            'total_reward': 0,
            'opponent_kills': [0] * num_opponents,
            'player_score': 0,  # 玩家分数
            'opponent_scores': [0] * num_opponents,  # 对手分数
            'winner': None,  # 获胜者 ('player' 或 opponent id)
            'race_finished': False  # 竞速是否结束
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        self.current_step = 0

        # 重置统计
        self.stats = {
            'kills': 0,
            'deaths': 0,
            'food_eaten': 0,
            'max_radius': 0,
            'total_reward': 0,
            'opponent_kills': [0] * self.num_opponents,
            'player_score': 0,
            'opponent_scores': [0] * self.num_opponents,
            'winner': None,
            'race_finished': False
        }

        # 初始化玩家（中心位置，小半径）
        self.player = {
            'x': self.world_size / 2,
            'y': self.world_size / 2,
            'radius': 20,
            'alive': True,
            'dx': 0,
            'dy': 0,
            'score': 0  # 玩家分数
        }

        # 初始化对手（分散在地图上）
        self.opponents = []
        for i in range(self.num_opponents):
            # 在地图上均匀分布
            angle = (2 * np.pi * i) / self.num_opponents
            distance = self.world_size * 0.3

            min_r, max_r = self.config['opponent_spawn_radius']

            self.opponents.append({
                'x': self.world_size / 2 + np.cos(angle) * distance,
                'y': self.world_size / 2 + np.sin(angle) * distance,
                'radius': np.random.uniform(min_r, max_r),
                'alive': True,
                'dx': 0,
                'dy': 0,
                'id': i,
                'score': 0  # 对手分数
            })

        # 初始化食物
        self._spawn_foods(self.food_count)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        self.current_step += 1

        # 1. 玩家动作
        self._apply_player_action(action)

        # 2. 对手动作（使用SFT模型）
        self._update_opponents()

        # 3. 物理更新
        self._update_physics()

        # 4. 碰撞检测
        reward = self._check_collisions()

        # 5. 食物补充
        self._replenish_foods()

        # 6. 检查竞速胜利条件
        race_result = self._check_race_winner()
        if race_result['race_finished']:
            self.stats['race_finished'] = True
            self.stats['winner'] = race_result['winner']

            # 如果玩家获胜，给予巨大奖励
            if race_result['winner'] == 'player':
                reward += 500.0
            # 如果对手获胜，给予惩罚
            else:
                reward -= 200.0

        # 7. 检查终止条件
        terminated = not self.player['alive'] or self.stats['race_finished']
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        self.stats['total_reward'] += reward

        return obs, reward, terminated, truncated, info

    def _apply_player_action(self, action: int):
        """应用玩家动作"""
        if not self.player['alive']:
            return

        direction_idx = action % 8
        angle = direction_idx * (np.pi / 4)

        # 移动速度（基于半径）
        base_speed = 30 / np.power(self.player['radius'], 0.4)

        self.player['dx'] = np.cos(angle) * base_speed
        self.player['dy'] = np.sin(angle) * base_speed

        # 分裂（动作8-15）
        if action >= 8 and self.player['radius'] >= 30:
            # 简化版本：减小半径代替实际分裂
            self.player['radius'] *= 0.9

    def _update_opponents(self):
        """更新对手行为（使用SFT模型）"""
        for opponent in self.opponents:
            if not opponent['alive']:
                continue

            # 使用SFT模型决策
            if self.sft_model and np.random.random() < self.config['opponent_skill']:
                # 获取对手的观察
                obs = self._get_opponent_observation(opponent)
                action = self.sft_model.predict(obs)
            else:
                # 随机动作（降低难度）
                action = np.random.randint(0, 16)

            # 应用动作
            direction_idx = action % 8
            angle = direction_idx * (np.pi / 4)

            base_speed = 30 / np.power(opponent['radius'], 0.4)
            opponent['dx'] = np.cos(angle) * base_speed
            opponent['dy'] = np.sin(angle) * base_speed

            # 对手也可能分裂
            if action >= 8 and opponent['radius'] >= 30 and np.random.random() < 0.1:
                opponent['radius'] *= 0.9

    def _get_opponent_observation(self, opponent: Dict) -> np.ndarray:
        """获取对手的观察向量（从对手视角）"""
        obs = np.zeros(49, dtype=np.float32)

        # 对手状态
        obs[0] = opponent['x'] / self.world_size
        obs[1] = opponent['y'] / self.world_size
        obs[2] = opponent['radius'] / 100.0
        obs[3] = 1.0 / 16.0  # 单个blob

        # 最近5个食物
        food_distances = [
            (self._distance(opponent['x'], opponent['y'], f['x'], f['y']), f)
            for f in self.foods
        ]
        food_distances.sort(key=lambda x: x[0])

        for i in range(min(5, len(food_distances))):
            idx = 4 + i * 4
            dist, food = food_distances[i]
            dx = food['x'] - opponent['x']
            dy = food['y'] - opponent['y']
            angle = np.arctan2(dy, dx)

            obs[idx] = np.cos(angle)
            obs[idx + 1] = np.sin(angle)
            obs[idx + 2] = food['radius'] / 10.0
            obs[idx + 3] = dist / self.world_size

        # 最近5个威胁（包括玩家和其他对手）
        threats = []

        # 玩家
        if self.player['alive']:
            dist = self._distance(opponent['x'], opponent['y'],
                                self.player['x'], self.player['y'])
            threats.append((dist, self.player))

        # 其他对手
        for other in self.opponents:
            if other['alive'] and other['id'] != opponent['id']:
                dist = self._distance(opponent['x'], opponent['y'],
                                    other['x'], other['y'])
                threats.append((dist, other))

        threats.sort(key=lambda x: x[0])

        for i in range(min(5, len(threats))):
            idx = 24 + i * 5
            dist, threat = threats[i]
            dx = threat['x'] - opponent['x']
            dy = threat['y'] - opponent['y']
            angle = np.arctan2(dy, dx)

            obs[idx] = np.cos(angle)
            obs[idx + 1] = np.sin(angle)
            obs[idx + 2] = threat['radius'] / 100.0
            obs[idx + 3] = dist / self.world_size
            obs[idx + 4] = opponent['radius'] / threat['radius']

        return obs

    def _update_physics(self):
        """更新物理状态"""
        # 更新玩家位置
        if self.player['alive']:
            self.player['x'] += self.player['dx']
            self.player['y'] += self.player['dy']
            self.player['x'] = np.clip(self.player['x'],
                                      self.player['radius'],
                                      self.world_size - self.player['radius'])
            self.player['y'] = np.clip(self.player['y'],
                                      self.player['radius'],
                                      self.world_size - self.player['radius'])

            # 衰减速度
            self.player['dx'] *= 0.9
            self.player['dy'] *= 0.9

        # 更新对手位置
        for opponent in self.opponents:
            if not opponent['alive']:
                continue

            opponent['x'] += opponent['dx']
            opponent['y'] += opponent['dy']
            opponent['x'] = np.clip(opponent['x'],
                                   opponent['radius'],
                                   self.world_size - opponent['radius'])
            opponent['y'] = np.clip(opponent['y'],
                                   opponent['radius'],
                                   self.world_size - opponent['radius'])

            opponent['dx'] *= 0.9
            opponent['dy'] *= 0.9

    def _check_collisions(self) -> float:
        """检查碰撞并返回奖励（竞速模式）"""
        reward = 0.0

        if not self.player['alive']:
            return reward

        # 玩家吃食物
        foods_to_remove = []
        for i, food in enumerate(self.foods):
            dist = self._distance(self.player['x'], self.player['y'],
                                food['x'], food['y'])
            if dist < self.player['radius'] + food['radius']:
                area = np.pi * (self.player['radius'] ** 2 + food['radius'] ** 2)
                self.player['radius'] = np.sqrt(area / np.pi)
                foods_to_remove.append(i)

                # 竞速模式：吃食物获得分数
                points = 1  # 基础分数
                self.player['score'] += points
                self.stats['player_score'] = self.player['score']

                # 奖励与目标分数的距离成反比（越接近越奖励越大）
                progress = self.player['score'] / self.target_score
                reward += 2.0 + progress * 3.0  # 1-4分奖励

                self.stats['food_eaten'] += 1

        for i in reversed(foods_to_remove):
            self.foods.pop(i)

        # 对手吃食物（更新对手分数）
        self._update_opponent_scores()

        # 与对手碰撞
        for opponent in self.opponents:
            if not opponent['alive']:
                continue

            dist = self._distance(self.player['x'], self.player['y'],
                                opponent['x'], opponent['y'])

            if dist < self.player['radius'] + opponent['radius']:
                # 吃掉对手
                if self.player['radius'] > opponent['radius'] * 1.1:
                    area = np.pi * (self.player['radius'] ** 2 + opponent['radius'] ** 2)
                    self.player['radius'] = np.sqrt(area / np.pi)
                    opponent['alive'] = False

                    # 竞速模式：击杀对手获得大量分数和奖励
                    kill_points = 30  # 击杀获得30分
                    self.player['score'] += kill_points
                    self.stats['player_score'] = self.player['score']

                    reward += 80.0  # 击杀对手大奖励！
                    self.stats['kills'] += 1
                    self.stats['opponent_kills'][opponent['id']] += 1

                # 被对手吃掉
                elif opponent['radius'] > self.player['radius'] * 1.1:
                    self.player['alive'] = False
                    reward = -100.0  # 死亡惩罚
                    self.stats['deaths'] += 1

        # 竞速模式：生存奖励减少，更注重分数
        reward += 0.05

        # 成长奖励
        if self.player['radius'] > self.stats['max_radius']:
            reward += (self.player['radius'] - self.stats['max_radius']) * 0.3
            self.stats['max_radius'] = self.player['radius']

        return reward

    def _update_opponent_scores(self):
        """更新对手分数（吃食物）"""
        for opponent in self.opponents:
            if not opponent['alive']:
                continue

            # 检查对手是否吃到食物
            foods_to_remove = []
            for i, food in enumerate(self.foods):
                dist = self._distance(opponent['x'], opponent['y'],
                                    food['x'], food['y'])
                if dist < opponent['radius'] + food['radius']:
                    area = np.pi * (opponent['radius'] ** 2 + food['radius'] ** 2)
                    opponent['radius'] = np.sqrt(area / np.pi)
                    foods_to_remove.append(i)

                    # 对手吃食物获得分数
                    points = 1
                    opponent['score'] += points
                    self.stats['opponent_scores'][opponent['id']] = opponent['score']

            for i in reversed(foods_to_remove):
                if i < len(self.foods):  # 防止重复删除
                    self.foods.pop(i)

    def _check_race_winner(self) -> Dict:
        """检查是否有获胜者（达到目标分数）"""
        result = {
            'race_finished': False,
            'winner': None,
            'winner_score': 0,
            'winner_steps': 0
        }

        # 检查玩家是否达到目标
        if self.player['alive'] and self.player['score'] >= self.target_score:
            result['race_finished'] = True
            result['winner'] = 'player'
            result['winner_score'] = self.player['score']
            result['winner_steps'] = self.current_step
            return result

        # 检查对手是否达到目标
        for opponent in self.opponents:
            if opponent['alive'] and opponent['score'] >= self.target_score:
                result['race_finished'] = True
                result['winner'] = f"opponent_{opponent['id']}"
                result['winner_score'] = opponent['score']
                result['winner_steps'] = self.current_step
                return result

        return result

    def _spawn_foods(self, count: int):
        """生成食物"""
        self.foods = []
        for _ in range(count):
            self.foods.append({
                'x': np.random.uniform(0, self.world_size),
                'y': np.random.uniform(0, self.world_size),
                'radius': np.random.uniform(3, 6)
            })

    def _replenish_foods(self):
        """补充食物"""
        target_food = int(self.food_count * self.config['food_spawn_rate'])
        needed = target_food - len(self.foods)

        for _ in range(needed):
            self.foods.append({
                'x': np.random.uniform(0, self.world_size),
                'y': np.random.uniform(0, self.world_size),
                'radius': np.random.uniform(3, 6)
            })

    def _get_observation(self) -> np.ndarray:
        """获取玩家观察（与原环境相同格式）"""
        obs = np.zeros(49, dtype=np.float32)

        if not self.player['alive']:
            return obs

        # 玩家状态
        obs[0] = self.player['x'] / self.world_size
        obs[1] = self.player['y'] / self.world_size
        obs[2] = self.player['radius'] / 100.0
        obs[3] = 1.0 / 16.0

        # 最近5个食物
        food_distances = [
            (self._distance(self.player['x'], self.player['y'], f['x'], f['y']), f)
            for f in self.foods
        ]
        food_distances.sort(key=lambda x: x[0])

        for i in range(min(5, len(food_distances))):
            idx = 4 + i * 4
            dist, food = food_distances[i]
            dx = food['x'] - self.player['x']
            dy = food['y'] - self.player['y']
            angle = np.arctan2(dy, dx)

            obs[idx] = np.cos(angle)
            obs[idx + 1] = np.sin(angle)
            obs[idx + 2] = food['radius'] / 10.0
            obs[idx + 3] = dist / self.world_size

        # 最近5个对手
        opponent_distances = [
            (self._distance(self.player['x'], self.player['y'], o['x'], o['y']), o)
            for o in self.opponents if o['alive']
        ]
        opponent_distances.sort(key=lambda x: x[0])

        for i in range(min(5, len(opponent_distances))):
            idx = 24 + i * 5
            dist, opp = opponent_distances[i]
            dx = opp['x'] - self.player['x']
            dy = opp['y'] - self.player['y']
            angle = np.arctan2(dy, dx)

            obs[idx] = np.cos(angle)
            obs[idx + 1] = np.sin(angle)
            obs[idx + 2] = opp['radius'] / 100.0
            obs[idx + 3] = dist / self.world_size
            obs[idx + 4] = self.player['radius'] / opp['radius']

        return obs

    def _get_info(self) -> Dict:
        """获取额外信息（竞速模式）"""
        alive_opponents = sum(1 for o in self.opponents if o['alive'])
        max_opponent_score = max(self.stats['opponent_scores']) if self.stats['opponent_scores'] else 0

        return {
            'player_radius': self.player['radius'],
            'player_alive': self.player['alive'],
            'step': self.current_step,
            'kills': self.stats['kills'],
            'deaths': self.stats['deaths'],
            'food_eaten': self.stats['food_eaten'],
            'max_radius': self.stats['max_radius'],
            'alive_opponents': alive_opponents,
            'total_reward': self.stats['total_reward'],
            # 竞速模式新增信息
            'player_score': self.stats['player_score'],
            'max_opponent_score': max_opponent_score,
            'score_gap': self.stats['player_score'] - max_opponent_score,
            'target_score': self.target_score,
            'race_finished': self.stats['race_finished'],
            'winner': self.stats['winner']
        }

    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """计算距离"""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def render(self):
        """渲染（可选）"""
        pass

    def close(self):
        """关闭环境"""
        pass


if __name__ == '__main__':
    # 测试环境
    print("Testing Multi-Agent Agario Environment...")

    env = MultiAgentAgarioEnv(
        world_size=6000,
        food_count=1000,
        num_opponents=5,
        difficulty='medium'
    )

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # 运行几步
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Reward: {reward:.2f}")
            print(f"  Player radius: {info['player_radius']:.1f}")
            print(f"  Kills: {info['kills']}")
            print(f"  Alive opponents: {info['alive_opponents']}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Final info: {info}")
            break

    env.close()
    print("\n✅ Environment test completed!")
