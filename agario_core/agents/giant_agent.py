"""
🎯 巨无霸智能体 - 全局视野优化版
Giant Agent - Advanced Mathematical Modeling System with Global Vision

基于以下数学理论：
1. 势场理论 (Potential Field Theory)
2. 多目标优化 (Multi-Objective Optimization)
3. 博弈论 (Game Theory)
4. 空间分析 (Spatial Analysis)
5. 动态规划 (Dynamic Programming)
6. 贝叶斯风险评估 (Bayesian Risk Assessment)

🎯 全局视野优化特性：
- 不再限制可见食物/敌人数量
- 扩大视野范围 (早期1000, 中期800, 后期1200)
- 全局威胁评估和猎物优先级排序
- 更激进的分裂策略
- 优化的势场参数

目标：以作弊的方式达到最佳状态，快速成长为游戏中的巨无霸
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class GiantAgent:
    """
    高级数学建模智能体 - 巨无霸版本

    核心策略：
    - 早期(r<25): 极度谨慎 + 高效觅食
    - 中期(25≤r<60): 平衡成长 + 机会捕猎
    - 后期(r≥60): 主动进攻 + 区域控制
    """

    def __init__(self, world_size: float = 4000.0):
        self.world_size = world_size

        # ========== 成长阶段定义 ==========
        self.EARLY_PHASE_THRESHOLD = 25.0
        self.MID_PHASE_THRESHOLD = 60.0

        # ========== 🎯 势场参数 (平衡版 - 快速成长+生存) ==========
        self.params = {
            'early': {
                'food_attraction': 200.0,    # 🚀 大幅提高食物吸引力（早期要快速吃豆）
                'danger_repulsion': 1000.0,  # 🛡️ 保持强危险排斥
                'prey_attraction': 15.0,     # ↓ 降低猎物吸引（早期别贪）
                'danger_distance': 800.0,    # 🛡️ 扩大危险感知距离
                'risk_aversion': 3.5,        # 🛡️ 适度降低风险规避（允许冒险吃豆）
            },
            'mid': {
                'food_attraction': 80.0,     # 食物吸引力
                'danger_repulsion': 500.0,   # 🛡️ 大幅增强危险排斥
                'prey_attraction': 100.0,    # 适中猎物吸引
                'danger_distance': 600.0,    # 🛡️ 扩大危险感知距离
                'risk_aversion': 2.5,        # 🛡️ 提高风险规避
            },
            'late': {
                'food_attraction': 50.0,     # 食物吸引力
                'danger_repulsion': 200.0,   # 🛡️ 增强危险排斥
                'prey_attraction': 300.0,    # 高猎物吸引
                'danger_distance': 500.0,    # 🛡️ 扩大危险感知距离
                'risk_aversion': 0.8,        # 🛡️ 保持一定风险规避
            }
        }

        # ========== 空间分析参数 ==========
        self.GRID_SIZE = 500  # 将世界划分为格子
        self.SECTOR_COUNT = 8  # 8个方向扇区

        # ========== 轨迹预测 ==========
        self.enemy_history = defaultdict(list)  # 跟踪敌人历史位置
        self.prediction_horizon = 5  # 预测未来5步

    def get_phase(self, radius: float) -> str:
        """确定当前成长阶段"""
        if radius < self.EARLY_PHASE_THRESHOLD:
            return 'early'
        elif radius < self.MID_PHASE_THRESHOLD:
            return 'mid'
        else:
            return 'late'

    def analyze_food_density(
        self,
        foods: List[Dict],
        scan_radius: float = 600.0
    ) -> Dict[str, any]:
        """
        食物密度分析 - 寻找食物富集区域

        使用高斯核密度估计:
        ρ(p) = Σ exp(-||p - p_i||² / (2σ²)) × r_i²

        Returns:
            {
                'hotspots': [(x, y, density), ...],  # 热点区域
                'grid': np.array,  # 密度网格
                'best_region': (x, y)  # 最佳觅食区域
            }
        """
        if len(foods) == 0:
            return {
                'hotspots': [],
                'grid': None,
                'best_region': (self.world_size / 2, self.world_size / 2)
            }

        # 创建密度网格
        grid_res = 10  # 网格分辨率
        grid_size = int(self.world_size / grid_res)
        density_grid = np.zeros((grid_size, grid_size))

        # 计算每个格子的食物密度
        for food in foods:
            fx, fy = food['x'], food['y']
            fr = food['radius']

            # 食物影响范围
            gx = int(fx / grid_res)
            gy = int(fy / grid_res)

            # 高斯核影响周围格子
            sigma = scan_radius / 3.0
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        dist = np.hypot(dx * grid_res, dy * grid_res)
                        weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                        density_grid[ny, nx] += weight * (fr ** 2)

        # 找到密度最高的区域
        hotspots = []
        for i in range(0, grid_size, 5):
            for j in range(0, grid_size, 5):
                local_density = density_grid[i:i+5, j:j+5].sum()
                if local_density > 0:
                    hotspots.append((
                        j * grid_res + grid_res * 2.5,
                        i * grid_res + grid_res * 2.5,
                        local_density
                    ))

        # 按密度排序
        hotspots.sort(key=lambda x: x[2], reverse=True)

        # 最佳区域
        best_region = hotspots[0][:2] if hotspots else (self.world_size / 2, self.world_size / 2)

        return {
            'hotspots': hotspots[:5],  # 前5个热点
            'grid': density_grid,
            'best_region': best_region
        }

    def analyze_sector_values(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        foods: List[Dict],
        enemies: List[Dict]
    ) -> List[Dict]:
        """
        8方向扇区价值分析

        每个扇区计算：
        V_i = w_f × f_i - w_e × e_i - w_d × d_i

        f_i: 食物密度
        e_i: 威胁度
        d_i: 距离惩罚

        Returns: 按价值排序的扇区列表
        """
        px, py = player_pos
        sectors = []

        for i in range(self.SECTOR_COUNT):
            angle_start = i * (360 / self.SECTOR_COUNT)
            angle_end = (i + 1) * (360 / self.SECTOR_COUNT)
            angle_mid = (angle_start + angle_end) / 2

            # 扇区方向
            dir_x = np.cos(np.radians(angle_mid))
            dir_y = np.sin(np.radians(angle_mid))

            # 分析这个方向的食物
            food_value = 0.0
            for food in foods:
                fx, fy = food['x'] - px, food['y'] - py
                dist = np.hypot(fx, fy)
                if dist < 1:
                    continue

                # 检查是否在扇区内
                food_angle = np.degrees(np.arctan2(fy, fx))
                if food_angle < 0:
                    food_angle += 360

                angle_diff = abs(food_angle - angle_mid)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff < (360 / self.SECTOR_COUNT / 2):
                    # 食物价值 = 质量 / 距离
                    food_value += (food['radius'] ** 2) / max(dist, 1.0)

            # 分析这个方向的威胁
            threat_value = 0.0
            for enemy in enemies:
                ex, ey = enemy['x'] - px, enemy['y'] - py
                dist = np.hypot(ex, ey)
                if dist < 1:
                    continue

                enemy_angle = np.degrees(np.arctan2(ey, ex))
                if enemy_angle < 0:
                    enemy_angle += 360

                angle_diff = abs(enemy_angle - angle_mid)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff < (360 / self.SECTOR_COUNT / 2):
                    size_ratio = player_radius / enemy['radius']
                    if size_ratio < 0.9:  # 危险的敌人
                        # 威胁 = (大小差异) / 距离²
                        threat_value += (0.9 - size_ratio) * 10.0 / max(dist ** 2, 1.0)

            # 综合价值
            phase = self.get_phase(player_radius)
            params = self.params[phase]

            value = food_value * 10.0 - threat_value * params['risk_aversion']

            sectors.append({
                'angle': angle_mid,
                'direction': (dir_x, dir_y),
                'food_value': food_value,
                'threat_value': threat_value,
                'total_value': value
            })

        # 按价值排序
        sectors.sort(key=lambda x: x['total_value'], reverse=True)
        return sectors

    def predict_enemy_position(
        self,
        enemy: Dict,
        steps: int = 5
    ) -> Tuple[float, float]:
        """
        预测敌人未来位置（简单线性预测）

        假设敌人保持当前速度移动
        """
        if 'dx' in enemy and 'dy' in enemy:
            pred_x = enemy['x'] + enemy['dx'] * steps
            pred_y = enemy['y'] + enemy['dy'] * steps
            return (pred_x, pred_y)
        else:
            return (enemy['x'], enemy['y'])

    def calculate_advanced_potential_field(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        foods: List[Dict],
        enemies: List[Dict],
        food_density_info: Dict
    ) -> Tuple[float, float]:
        """
        高级势场计算 - 结合多种因素

        1. 食物吸引力（考虑密度热点）
        2. 敌人排斥/吸引（预测轨迹）
        3. 边界排斥
        4. 区域控制引导

        Returns: (dx, dy) 归一化方向向量
        """
        px, py = player_pos
        phase = self.get_phase(player_radius)
        params = self.params[phase]

        total_fx = 0.0
        total_fy = 0.0

        # ========== 1. 食物吸引 (快速成长优化) ==========
        # 早期：快速吃豆成长（躲避+吃豆并重）
        # 后期：追求食物密集区域
        if phase == 'early':
            # 🚀 早期策略：优先吃近处安全的豆子
            food_forces = []
            vision_range = 600  # 🚀 缩小视野到600，专注近处安全食物

            # 评估每个食物的安全性
            for food in foods:
                dx = food['x'] - px
                dy = food['y'] - py
                dist = np.hypot(dx, dy)
                if dist < 1 or dist > vision_range:
                    continue

                # 🚀 检查这个食物附近是否安全
                food_safety = 1.0
                for enemy in enemies:
                    enemy_to_food_dist = np.hypot(enemy['x'] - food['x'], enemy['y'] - food['y'])
                    size_ratio = player_radius / enemy['radius']

                    # 如果敌人在食物附近且比我们大，降低安全性
                    if enemy_to_food_dist < 300 and size_ratio < 0.85:
                        food_safety *= 0.3  # 危险食物，大幅降低吸引力

                # 🚀 基础吸引力 (距离越近越强)
                base_force = params['food_attraction'] / (dist ** 1.2)  # 改为1.2次方，更平缓

                # 🚀 食物大小奖励（大食物更有价值）
                size_bonus = (food['radius'] / 5.0) ** 0.5

                # 🚀 综合吸引力 = 基础力 × 安全性 × 大小奖励
                force_mag = base_force * food_safety * size_bonus

                food_forces.append((dx / dist * force_mag, dy / dist * force_mag, dist, food_safety))

            # 🚀 优先追最近最安全的食物（增加到25个）
            food_forces.sort(key=lambda x: x[2] / max(x[3], 0.1))  # 距离/安全性排序
            for fx, fy, _, _ in food_forces[:25]:
                total_fx += fx
                total_fy += fy
        else:
            # 🎯 中后期：全局食物密集区吸引 + 路径优化
            best_region = food_density_info['best_region']
            dx = best_region[0] - px
            dy = best_region[1] - py
            dist = np.hypot(dx, dy)

            if dist > 100:  # 不在最佳区域内
                force_mag = params['food_attraction'] * 2.5 / (dist ** 1.2)  # 增强吸引力
                total_fx += (dx / dist) * force_mag
                total_fy += (dy / dist) * force_mag

            # 🎯 全局视野：利用所有可达范围内的食物
            vision_range = 800 if phase == 'mid' else 1200  # 后期视野更广
            nearby_foods = []
            for food in foods:  # 不再限制数量
                dx = food['x'] - px
                dy = food['y'] - py
                dist = np.hypot(dx, dy)
                if dist < 1 or dist > vision_range:
                    continue
                nearby_foods.append((food, dist))

            # 按距离排序，优先吃近处的
            nearby_foods.sort(key=lambda x: x[1])
            for food, dist in nearby_foods[:30]:  # 增加到30个
                dx = food['x'] - px
                dy = food['y'] - py
                force_mag = params['food_attraction'] * 0.5 / (dist ** 1.5)
                total_fx += (dx / dist) * force_mag
                total_fy += (dy / dist) * force_mag

        # ========== 2. 敌人交互（全局预测性分析） ==========
        # 🎯 全局视野：分析所有敌人的威胁和机会
        threats = []  # 危险的敌人
        prey = []     # 可捕猎的敌人

        for enemy in enemies:
            # 预测敌人位置
            pred_x, pred_y = self.predict_enemy_position(enemy, steps=3)

            dx = pred_x - px
            dy = pred_y - py
            dist = np.hypot(dx, dy)

            if dist < 1:
                continue

            size_ratio = player_radius / enemy['radius']

            if size_ratio > 1.2:  # 猎物
                prey.append((enemy, pred_x, pred_y, dist, size_ratio))
            elif size_ratio < 0.85:  # 危险
                threats.append((enemy, pred_x, pred_y, dist, size_ratio))

        # 处理猎物：优先追击最优目标
        if prey:
            # 按收益排序：距离近且大小差异大的优先
            prey.sort(key=lambda x: x[3] / (x[4] - 1.0))  # dist / advantage

            for enemy, pred_x, pred_y, dist, size_ratio in prey[:20]:  # 考虑前20个猎物
                dx = pred_x - px
                dy = pred_y - py

                # 吸引力（后期更强）
                hunt_strength = params['prey_attraction']
                advantage = (size_ratio - 1.0) * 2.0

                # 🎯 作弊优化：对大猎物提高吸引力
                size_bonus = (enemy['radius'] / 10.0) ** 0.5
                force_mag = hunt_strength * advantage * size_bonus / (dist ** 1.3)

                total_fx += (dx / dist) * force_mag
                total_fy += (dy / dist) * force_mag

        # 处理威胁：按危险程度避开
        if threats:
            # 按危险程度排序：近且大的最危险
            threats.sort(key=lambda x: x[3] / max(0.85 - x[4], 0.1))

            # 🛡️ 考虑所有可见威胁，不限制数量
            for enemy, pred_x, pred_y, dist, size_ratio in threats[:30]:
                dx = pred_x - px
                dy = pred_y - py

                # 排斥力（早期更强）
                danger_strength = params['danger_repulsion']
                danger_level = (0.85 - size_ratio) * 5.0  # 🛡️ 提高危险等级系数

                # 🚀 优化距离分级：近距离超强，中等距离适中，远距离很弱
                if dist < 100:
                    # 极近距离：超强排斥
                    force_mag = danger_strength * danger_level * 10.0 / max(dist ** 3.0, 1.0)
                elif dist < 200:
                    # 很近距离：强排斥
                    force_mag = danger_strength * danger_level * 5.0 / (dist ** 2.8)
                elif dist < 400:
                    # 中等距离：正常排斥
                    force_mag = danger_strength * danger_level / (dist ** 2.5)
                elif dist < params['danger_distance']:
                    # 较远距离：轻度排斥
                    force_mag = danger_strength * danger_level * 0.2 / (dist ** 2.0)
                else:
                    # 🚀 远距离：极轻度排斥（不影响吃豆）
                    force_mag = danger_strength * danger_level * 0.05 / (dist ** 1.5)

                # 🛡️ 特别危险的敌人 - 超级排斥
                if size_ratio < 0.5:  # 巨大威胁
                    force_mag *= 8.0
                elif size_ratio < 0.6:
                    force_mag *= 5.0
                elif size_ratio < 0.75:
                    force_mag *= 3.0

                # 🛡️ 如果敌人正在靠近，额外增强排斥
                if 'dx' in enemy and 'dy' in enemy:
                    # 计算敌人是否朝我们移动
                    enemy_to_us_x = px - enemy['x']
                    enemy_to_us_y = py - enemy['y']
                    enemy_vel_x = enemy.get('dx', 0)
                    enemy_vel_y = enemy.get('dy', 0)

                    # 点积判断方向
                    approaching = (enemy_to_us_x * enemy_vel_x + enemy_to_us_y * enemy_vel_y) > 0
                    if approaching and dist < params['danger_distance']:
                        force_mag *= 2.0  # 正在靠近的敌人，双倍排斥

                total_fx -= (dx / dist) * force_mag
                total_fy -= (dy / dist) * force_mag

        # ========== 3. 边界排斥 (增强版 - 防止被逼角落) ==========
        # 🛡️ 早期更大的安全边界
        margin = 600 if phase == 'early' else 400

        # 🛡️ 更强的边界排斥力
        boundary_strength = 500.0 if phase == 'early' else 300.0

        # 左边界
        if px < margin:
            boundary_force = boundary_strength / max(px + 1, 1) ** 2
            total_fx += boundary_force

        # 右边界
        if px > self.world_size - margin:
            dist_to_edge = self.world_size - px
            boundary_force = boundary_strength / max(dist_to_edge + 1, 1) ** 2
            total_fx -= boundary_force

        # 上边界
        if py < margin:
            boundary_force = boundary_strength / max(py + 1, 1) ** 2
            total_fy += boundary_force

        # 下边界
        if py > self.world_size - margin:
            dist_to_edge = self.world_size - py
            boundary_force = boundary_strength / max(dist_to_edge + 1, 1) ** 2
            total_fy -= boundary_force

        # 🛡️ 检测是否在角落，如果是则强制向中心移动
        corner_margin = 800
        in_corner = False

        if (px < corner_margin and py < corner_margin) or \
           (px < corner_margin and py > self.world_size - corner_margin) or \
           (px > self.world_size - corner_margin and py < corner_margin) or \
           (px > self.world_size - corner_margin and py > self.world_size - corner_margin):
            in_corner = True

        if in_corner:
            # 强制向中心移动
            center_x = self.world_size / 2
            center_y = self.world_size / 2
            dx_center = center_x - px
            dy_center = center_y - py
            dist_center = np.hypot(dx_center, dy_center)

            if dist_center > 0:
                # 🛡️ 角落逃逸力量（非常强）
                escape_force = 1000.0
                total_fx += (dx_center / dist_center) * escape_force
                total_fy += (dy_center / dist_center) * escape_force

        # ========== 4. 中心控制吸引（后期） ==========
        if phase == 'late':
            # 巨无霸应该控制中心区域
            center_x = self.world_size / 2
            center_y = self.world_size / 2
            dx = center_x - px
            dy = center_y - py
            dist_to_center = np.hypot(dx, dy)

            if dist_to_center > 500:
                force_mag = 50.0 / (dist_to_center ** 1.1)
                total_fx += (dx / dist_to_center) * force_mag
                total_fy += (dy / dist_to_center) * force_mag

        # 归一化
        magnitude = np.hypot(total_fx, total_fy)
        if magnitude < 1e-5:
            return (0.0, 0.0)

        return (total_fx / magnitude, total_fy / magnitude)

    def evaluate_split_opportunity(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        enemies: List[Dict]
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        🛡️ 评估分裂机会 - 生存优先版本

        分裂条件：
        1. 半径 >= 40 (提高阈值，更保守)
        2. 附近有可捕获的猎物
        3. 环境必须安全（严格检查）
        4. 期望收益 > 风险成本（更高阈值）

        Returns: (should_split, target_pos)
        """
        # 🛡️ 条件1: 提高最小半径，避免过早分裂
        if player_radius < 40:
            return (False, None)

        px, py = player_pos
        split_radius = player_radius / np.sqrt(2)  # 分裂后半径
        split_range = player_radius * 8  # 🎯 增加分裂范围

        # 🎯 全局视野：评估所有可能的分裂目标
        candidates = []

        for enemy in enemies:
            dist = np.hypot(enemy['x'] - px, enemy['y'] - py)
            size_ratio = split_radius / enemy['radius']

            # 条件2: 可捕获（分裂后仍然更大）
            if size_ratio < 1.12:  # 🎯 降低阈值，更激进
                continue

            # 条件2b: 在分裂范围内
            if dist > split_range or dist < 50:
                continue

            # 🎯 全局安全检查：评估周围所有威胁
            threat_score = 0
            for other in enemies:
                if other == enemy:
                    continue
                other_dist = np.hypot(other['x'] - px, other['y'] - py)

                # 只考虑附近的威胁
                if other_dist < 800:
                    other_ratio = split_radius / other['radius']
                    if other_ratio < 0.9:  # 危险的敌人
                        # 威胁程度 = 大小差异 / 距离
                        threat_score += (0.9 - other_ratio) * 100.0 / max(other_dist, 100)

            # 计算期望收益
            # EV = P(成功) × 收益 - 威胁成本

            # 成功概率（距离越近越高）
            prob_success = max(0, 1.0 - dist / split_range) * 0.85

            # 🎯 收益：吃掉敌人的质量 + 大小奖励
            reward = (enemy['radius'] ** 2) * 15  # 增加收益权重
            size_bonus = size_ratio * 5  # 大小优势奖励
            reward += size_bonus

            # 成本：威胁评分
            risk_cost = threat_score * 2

            ev = prob_success * reward - risk_cost

            candidates.append((enemy, ev, dist))

        # 按期望收益排序
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_prey, best_ev, _ = candidates[0]

            # 🛡️ 条件4: 提高阈值，更保守的分裂策略
            # 只在绝对安全且收益高的情况下分裂
            if best_ev > 50 and threat_score < 5:  # 严格条件
                return (True, (best_prey['x'], best_prey['y']))

        return (False, None)

    def calculate_emergency_escape(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        enemies: List[Dict]
    ) -> Optional[Tuple[float, float]]:
        """
        🛡️ 紧急逃生路径计算

        当处于极度危险时，计算最安全的逃跑方向

        Returns: (dx, dy) 逃生方向，如果不需要逃生则返回 None
        """
        px, py = player_pos

        # 找到所有极度危险的敌人（距离近且体积大）
        immediate_threats = []
        for enemy in enemies:
            dist = np.hypot(enemy['x'] - px, enemy['y'] - py)
            size_ratio = player_radius / enemy['radius']

            # 🚀 调整紧急逃生条件：更严格，避免过度敏感
            # 极度危险：距离<200 且 比我们大很多 (或距离<150无论大小)
            if (dist < 200 and size_ratio < 0.6) or (dist < 150 and size_ratio < 0.8):
                immediate_threats.append((enemy, dist, size_ratio))

        if not immediate_threats:
            return None  # 不需要紧急逃生

        # 🛡️ 计算安全逃生方向：远离所有威胁
        escape_fx = 0.0
        escape_fy = 0.0

        for enemy, dist, size_ratio in immediate_threats:
            dx = enemy['x'] - px
            dy = enemy['y'] - py

            # 危险程度
            danger = (0.7 - size_ratio) * 1000.0 / max(dist, 10.0) ** 2

            # 反方向逃跑
            escape_fx -= (dx / dist) * danger
            escape_fy -= (dy / dist) * danger

        # 归一化
        magnitude = np.hypot(escape_fx, escape_fy)
        if magnitude > 0:
            return (escape_fx / magnitude, escape_fy / magnitude)

        return None

    def calculate_risk_score(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        enemies: List[Dict]
    ) -> float:
        """
        🛡️ 贝叶斯风险评估（增强版）

        Risk = Σ P(碰撞|敌人i) × Severity(敌人i)

        Returns: 风险分数 [0, ∞)
        """
        px, py = player_pos
        total_risk = 0.0

        for enemy in enemies:
            dist = np.hypot(enemy['x'] - px, enemy['y'] - py)
            size_ratio = player_radius / enemy['radius']

            # 只考虑危险的敌人
            if size_ratio >= 0.9:
                continue

            # 🛡️ 增强的碰撞概率（距离越近概率越高）
            if dist < 100:
                collision_prob = 0.9
            elif dist < 200:
                collision_prob = 0.7
            elif dist < 400:
                collision_prob = 0.4
            else:
                collision_prob = np.exp(-dist / 300.0)

            # 🛡️ 增强的严重程度
            severity = (0.9 - size_ratio) * 150  # 提高严重度系数

            # 🛡️ 如果敌人正在接近，额外增加风险
            if 'dx' in enemy and 'dy' in enemy:
                enemy_to_us_x = px - enemy['x']
                enemy_to_us_y = py - enemy['y']
                approaching = (enemy_to_us_x * enemy.get('dx', 0) +
                             enemy_to_us_y * enemy.get('dy', 0)) > 0
                if approaching:
                    severity *= 2.0

            total_risk += collision_prob * severity

        return total_risk

    def get_action(self, game_state: Dict) -> Dict:
        """
        主决策函数 - 整合所有数学模型

        决策流程：
        1. 分析食物密度分布
        2. 评估扇区价值
        3. 计算势场方向
        4. 评估分裂机会
        5. 输出最优动作

        Args:
            game_state: {
                'player_x': float,
                'player_y': float,
                'player_radius': float,
                'foods': List[Dict[x, y, radius]],
                'enemies': List[Dict[x, y, radius, dx?, dy?]]
            }

        Returns:
            {
                'angle': float (0-360度),
                'split': bool,
                'debug_info': Dict (可选)
            }
        """
        player_pos = (game_state['player_x'], game_state['player_y'])
        player_radius = game_state['player_radius']
        foods = game_state.get('foods', [])
        enemies = game_state.get('enemies', [])

        # 🛡️ ===== 步骤0: 紧急逃生检查（最高优先级）=====
        escape_direction = self.calculate_emergency_escape(player_pos, player_radius, enemies)
        emergency_mode = escape_direction is not None

        if emergency_mode:
            # 🛡️ 紧急模式：忽略食物和分裂，全力逃跑
            dx, dy = escape_direction
            should_split = False
            risk = 999.0  # 极高风险标记
        else:
            # 正常模式：执行常规决策
            # ===== 步骤1: 食物密度分析 =====
            food_density = self.analyze_food_density(foods)

            # ===== 步骤2: 扇区价值分析 =====
            sectors = self.analyze_sector_values(player_pos, player_radius, foods, enemies)

            # ===== 步骤3: 势场计算 =====
            dx, dy = self.calculate_advanced_potential_field(
                player_pos, player_radius, foods, enemies, food_density
            )

            # ===== 步骤4: 风险评估 =====
            risk = self.calculate_risk_score(player_pos, player_radius, enemies)

            # 🛡️ ===== 步骤5: 风险过高时禁止分裂 =====
            if risk > 15:  # 🛡️ 降低风险阈值，更保守
                should_split = False
            else:
                # 只在低风险时考虑分裂
                should_split, split_target = self.evaluate_split_opportunity(
                    player_pos, player_radius, enemies
                )

                # 如果要分裂，方向指向目标
                if should_split and split_target:
                    dx = split_target[0] - player_pos[0]
                    dy = split_target[1] - player_pos[1]
                    dist = np.hypot(dx, dy)
                    if dist > 0:
                        dx, dy = dx / dist, dy / dist

        # 转换为角度
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360

        # 调试信息
        phase = self.get_phase(player_radius)
        debug_info = {
            'phase': phase,
            'risk_score': risk,
            'emergency_mode': emergency_mode,  # 🛡️ 新增紧急模式标记
            'best_sector_value': sectors[0]['total_value'] if not emergency_mode and sectors else 0,
            'food_hotspots': len(food_density['hotspots']) if not emergency_mode else 0,
            'split_opportunity': should_split
        }

        return {
            'angle': angle,
            'split': should_split,
            'debug_info': debug_info
        }


# ========== 测试和演示 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("巨无霸智能体 - 数学建模系统测试")
    print("=" * 60)

    agent = GiantAgent(world_size=4000)

    # 模拟三个阶段的游戏状态
    test_cases = [
        {
            'name': '早期阶段 (r=20)',
            'state': {
                'player_x': 2000,
                'player_y': 2000,
                'player_radius': 20,
                'foods': [
                    {'x': 2100, 'y': 2050, 'radius': 5},
                    {'x': 2150, 'y': 2100, 'radius': 4},
                    {'x': 1950, 'y': 1980, 'radius': 5},
                    {'x': 1900, 'y': 1900, 'radius': 6},
                ],
                'enemies': [
                    {'x': 2300, 'y': 2200, 'radius': 60, 'dx': -1, 'dy': -1},  # 巨大威胁
                    {'x': 1800, 'y': 1800, 'radius': 15, 'dx': 0.5, 'dy': 0.5},  # 小敌人
                ]
            }
        },
        {
            'name': '中期阶段 (r=40)',
            'state': {
                'player_x': 2000,
                'player_y': 2000,
                'player_radius': 40,
                'foods': [
                    {'x': 2200, 'y': 2100, 'radius': 5},
                    {'x': 1900, 'y': 1900, 'radius': 4},
                ],
                'enemies': [
                    {'x': 2200, 'y': 2200, 'radius': 30, 'dx': 0, 'dy': 0},  # 可捕猎
                    {'x': 1700, 'y': 1700, 'radius': 50, 'dx': 1, 'dy': 1},  # 威胁
                ]
            }
        },
        {
            'name': '后期阶段 (r=80) - 巨无霸',
            'state': {
                'player_x': 1800,
                'player_y': 1800,
                'player_radius': 80,
                'foods': [
                    {'x': 2000, 'y': 2000, 'radius': 5},
                ],
                'enemies': [
                    {'x': 2100, 'y': 2100, 'radius': 40, 'dx': 0, 'dy': 0},  # 猎物
                    {'x': 2200, 'y': 2200, 'radius': 35, 'dx': -0.5, 'dy': -0.5},  # 猎物
                    {'x': 1500, 'y': 1500, 'radius': 25, 'dx': 1, 'dy': 0},  # 小猎物
                ]
            }
        }
    ]

    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试: {test['name']}")
        print('=' * 60)

        action = agent.get_action(test['state'])

        print(f"决策结果:")
        print(f"  - 移动方向: {action['angle']:.1f}°")
        print(f"  - 是否分裂: {'是' if action['split'] else '否'}")
        print(f"\n调试信息:")
        print(f"  - 成长阶段: {action['debug_info']['phase']}")
        print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
        print(f"  - 最佳扇区价值: {action['debug_info']['best_sector_value']:.2f}")
        print(f"  - 食物热点数: {action['debug_info']['food_hotspots']}")
        print(f"  - 分裂机会: {'是' if action['debug_info']['split_opportunity'] else '否'}")

    print(f"\n{'=' * 60}")
    print("测试完成！智能体已准备好成长为巨无霸！")
    print('=' * 60)
