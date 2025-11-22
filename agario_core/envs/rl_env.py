import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AgarioEnv(gym.Env):
    """Agar.io Gym Environment for RL training"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, world_size=4000, food_count=500, enemy_count=30):
        super().__init__()
        
        self.world_size = world_size
        self.food_count = food_count
        self.enemy_count = enemy_count
        
        # State space: normalized observations
        # [player_x, player_y, player_radius, blob_count,
        #  nearest_5_food (cos_angle, sin_angle, radius, distance) * 5 = 20,
        #  nearest_5_enemies (cos_angle, sin_angle, radius, distance, relative_size) * 5 = 25]
        # Total: 4 + 20 + 25 = 49
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(49,), dtype=np.float32
        )
        
        # Action space: [angle (0-360), split (0 or 1)]
        # Using discrete actions for simplicity: 8 directions + split
        # 0-7: move in 8 directions, 8-15: move + split
        self.action_space = spaces.Discrete(16)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.player = {
            'x': self.world_size / 2,
            'y': self.world_size / 2,
            'radius': 20,
        }
        
        self.foods = []
        for _ in range(self.food_count):
            self.foods.append({
                'x': np.random.uniform(0, self.world_size),
                'y': np.random.uniform(0, self.world_size),
                'radius': np.random.uniform(3, 6),
            })
        
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
        
        self.score = 0
        self.steps = 0
        self.max_steps = 5000
        self.prev_radius = 20  # Track previous radius for growth reward
        self.prev_score = 0  # Track score for milestone detection
        self.milestones_reached = set()  # Track achieved milestones
        self.growth_history = []  # Track growth trend
        self.max_radius_achieved = 20  # Track peak size
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get normalized observation vector with directional information"""
        obs = np.zeros(49, dtype=np.float32)
        
        # Player state (normalized)
        obs[0] = self.player['x'] / self.world_size
        obs[1] = self.player['y'] / self.world_size
        obs[2] = self.player['radius'] / 100.0  # Normalize by max expected radius
        
        # NEW: Number of blobs (for split awareness, normalized to 0-1)
        # In training, we simulate this as 1 blob, but frontend will send actual count
        obs[3] = 1.0 / 16.0  # Normalize: 1 blob out of max 16
        
        # Find nearest 5 food items
        food_distances = []
        for food in self.foods:
            dist = np.hypot(food['x'] - self.player['x'], food['y'] - self.player['y'])
            food_distances.append((dist, food))
        
        food_distances.sort(key=lambda x: x[0])
        for i in range(min(5, len(food_distances))):
            dist, food = food_distances[i]
            idx = 4 + i * 4  # Start at 4 now (was 3)
            
            # Calculate relative angle to food (-180 to 180 degrees, normalized to -1 to 1)
            dx = food['x'] - self.player['x']
            dy = food['y'] - self.player['y']
            angle = np.arctan2(dy, dx)  # -pi to pi
            
            obs[idx] = np.cos(angle)  # Direction x component (-1 to 1)
            obs[idx + 1] = np.sin(angle)  # Direction y component (-1 to 1)  
            obs[idx + 2] = food['radius'] / 10.0
            obs[idx + 3] = dist / self.world_size  # Normalized distance
        
        # Find nearest 5 enemies
        enemy_distances = []
        for enemy in self.enemies:
            dist = np.hypot(enemy['x'] - self.player['x'], enemy['y'] - self.player['y'])
            enemy_distances.append((dist, enemy))
        
        enemy_distances.sort(key=lambda x: x[0])
        for i in range(min(5, len(enemy_distances))):
            dist, enemy = enemy_distances[i]
            idx = 24 + i * 5  # Start at 24 now (was 23)
            
            # Calculate relative angle to enemy
            dx = enemy['x'] - self.player['x']
            dy = enemy['y'] - self.player['y']
            angle = np.arctan2(dy, dx)
            
            obs[idx] = np.cos(angle)  # Direction x component
            obs[idx + 1] = np.sin(angle)  # Direction y component
            obs[idx + 2] = enemy['radius'] / 100.0
            obs[idx + 3] = dist / self.world_size
            obs[idx + 4] = self.player['radius'] / enemy['radius']  # Relative size
        
        return obs
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # Decode action
        direction_idx = action % 8
        should_split = action >= 8
        
        # 8 directions: 0, 45, 90, 135, 180, 225, 270, 315 degrees
        angle = direction_idx * 45 * np.pi / 180
        
        # Move player
        speed = 30 / (self.player['radius'] ** 0.4)
        self.player['x'] += np.cos(angle) * speed
        self.player['y'] += np.sin(angle) * speed
        
        # Boundary checks
        self.player['x'] = np.clip(self.player['x'], self.player['radius'], 
                                   self.world_size - self.player['radius'])
        self.player['y'] = np.clip(self.player['y'], self.player['radius'], 
                                   self.world_size - self.player['radius'])
        
        # Update enemies
        for enemy in self.enemies:
            enemy['x'] += enemy['dx']
            enemy['y'] += enemy['dy']
            
            # Bounce off walls
            if enemy['x'] - enemy['radius'] < 0 or enemy['x'] + enemy['radius'] > self.world_size:
                enemy['dx'] *= -1
            if enemy['y'] - enemy['radius'] < 0 or enemy['y'] + enemy['radius'] > self.world_size:
                enemy['dy'] *= -1
        
        # ============ GOAL-ORIENTED REWARD SYSTEM ============
        # Success in Agar.io = Become largest, achieve milestones, dominate game
        
        reward = 0
        terminated = False
        
        # Track initial state for comparison
        initial_radius = self.player['radius']
        initial_score = self.score
        
        # -------- 1. IMMEDIATE ACTIONS --------
        # Eat food
        foods_eaten = 0
        for i in range(len(self.foods) - 1, -1, -1):
            food = self.foods[i]
            dist = np.hypot(food['x'] - self.player['x'], food['y'] - self.player['y'])
            if dist < self.player['radius'] + food['radius']:
                area = np.pi * self.player['radius'] ** 2 + np.pi * food['radius'] ** 2
                self.player['radius'] = np.sqrt(area / np.pi)
                self.score += 1
                foods_eaten += 1
                reward += 1.0  # Base food reward
                # Respawn food
                self.foods[i] = {
                    'x': np.random.uniform(0, self.world_size),
                    'y': np.random.uniform(0, self.world_size),
                    'radius': np.random.uniform(3, 6),
                }
        
        # Efficiency bonus: eating multiple foods in one step
        if foods_eaten > 1:
            reward += foods_eaten * 0.5
        
        # Check enemy collisions
        enemies_eaten = 0
        for enemy in self.enemies:
            dist = np.hypot(enemy['x'] - self.player['x'], enemy['y'] - self.player['y'])
            if dist < self.player['radius'] + enemy['radius']:
                if self.player['radius'] > enemy['radius'] * 1.1:
                    # Eat enemy - reward scales with enemy size
                    area = np.pi * self.player['radius'] ** 2 + np.pi * enemy['radius'] ** 2
                    self.player['radius'] = np.sqrt(area / np.pi)
                    enemy_score = int(enemy['radius'])
                    self.score += enemy_score
                    enemies_eaten += 1
                    # Bigger enemies = bigger reward
                    reward += 10.0 + (enemy_score * 0.2)
                    # Respawn enemy
                    enemy['radius'] = np.random.uniform(15, 50)
                    enemy['x'] = np.random.uniform(0, self.world_size)
                    enemy['y'] = np.random.uniform(0, self.world_size)
                elif enemy['radius'] > self.player['radius'] * 1.1:
                    # Death penalty scales with how much we were dominating
                    dominance_factor = max(1.0, self.score / 100.0)
                    reward -= 10.0 * dominance_factor
                    terminated = True
        
        # -------- 2. MILESTONE ACHIEVEMENTS (Big Success Moments) --------
        # Score milestones: 50, 100, 200, 500, 1000, 2000
        milestones = [50, 100, 200, 500, 1000, 2000, 5000]
        for milestone in milestones:
            if self.score >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.add(milestone)
                # Exponential rewards for higher milestones
                milestone_reward = 50 + (milestone / 10)
                reward += milestone_reward
                print(f"🎯 Milestone achieved: {milestone} points! Bonus: +{milestone_reward:.1f}")
        
        # -------- 3. DOMINANCE & RELATIVE PERFORMANCE --------
        # Calculate dominance: how much bigger are we than average enemy?
        if len(self.enemies) > 0:
            avg_enemy_radius = np.mean([e['radius'] for e in self.enemies])
            size_ratio = self.player['radius'] / avg_enemy_radius
            
            # Reward for being dominant (bigger than average)
            if size_ratio > 1.5:
                reward += (size_ratio - 1.0) * 2.0  # Scaling dominance reward
            
            # Extra reward for being much larger (true dominance)
            if size_ratio > 2.0:
                reward += 5.0
        
        # -------- 4. DANGER AWARENESS (Fear & Confidence) --------
        # AI should fear large enemies and be confident with small ones
        danger_score = 0
        opportunity_score = 0
        
        for enemy in self.enemies:
            dist = np.hypot(enemy['x'] - self.player['x'], enemy['y'] - self.player['y'])
            
            # Only consider nearby enemies (within perception range)
            perception_range = 800
            if dist < perception_range:
                size_ratio = self.player['radius'] / enemy['radius']
                proximity_factor = 1.0 - (dist / perception_range)  # Closer = stronger effect
                
                if size_ratio < 0.9:  # Enemy is larger (DANGER!)
                    # Penalty for being near dangerous enemies
                    danger_penalty = -2.0 * (0.9 - size_ratio) * proximity_factor
                    danger_score += danger_penalty
                    
                    # Extra penalty if very close to much larger enemy
                    if dist < 300 and size_ratio < 0.7:
                        danger_score -= 5.0  # Strong fear signal!
                        
                elif size_ratio > 1.15:  # Enemy is smaller (OPPORTUNITY!)
                    # Reward for hunting vulnerable targets
                    hunt_reward = 1.0 * (size_ratio - 1.0) * proximity_factor
                    opportunity_score += hunt_reward
                    
                    # Extra reward for closing in on prey
                    if dist < 300:
                        opportunity_score += 2.0  # Encourage pursuit!
        
        reward += danger_score
        reward += opportunity_score
        
        # -------- 5. GROWTH TREND (Continuous Improvement) --------
        radius_growth = self.player['radius'] - initial_radius
        if radius_growth > 0:
            # Reward actual growth
            growth_reward = radius_growth * 5.0
            reward += growth_reward
        
        # Track and reward consistent growth trend
        self.growth_history.append(self.player['radius'])
        if len(self.growth_history) > 50:
            self.growth_history.pop(0)
        
        # Reward positive growth trend over last 50 steps
        if len(self.growth_history) >= 10:
            recent_growth = self.growth_history[-1] - self.growth_history[-10]
            if recent_growth > 0:
                reward += recent_growth * 0.5
        
        # Track maximum size achieved
        if self.player['radius'] > self.max_radius_achieved:
            self.max_radius_achieved = self.player['radius']
            reward += 2.0  # Reward for new personal best
        
        # -------- 6. STRATEGIC POSITIONING --------
        # Control center area (most food spawns here)
        center_x = self.world_size / 2
        center_y = self.world_size / 2
        dist_from_center = np.hypot(self.player['x'] - center_x, self.player['y'] - center_y)
        
        # Reward for controlling center when dominant
        if self.player['radius'] > 30:  # Only when reasonably sized
            center_control = 1.0 - (dist_from_center / 1000.0)
            reward += center_control * 0.5
        
        # Penalty for edge-hugging (anti-camping)
        edge_margin = 300
        if (self.player['x'] < edge_margin or self.player['x'] > self.world_size - edge_margin or
            self.player['y'] < edge_margin or self.player['y'] > self.world_size - edge_margin):
            reward -= 0.5
        
        # -------- 7. EFFICIENCY & AGGRESSION --------
        # Reward for high score per step (efficiency)
        efficiency = self.score / max(1, self.steps)
        if efficiency > 0.1:
            reward += efficiency * 2.0
        
        # Reward aggression (eating enemies)
        if enemies_eaten > 0:
            reward += enemies_eaten * 3.0
        
        # -------- 8. SURVIVAL & LONGEVITY --------
        # Base survival reward increases with size (bigger = more valuable to keep alive)
        survival_value = 0.01 * (1 + self.player['radius'] / 50.0)
        reward += survival_value
        
        # Bonus for long survival at large size
        if self.player['radius'] > 50 and self.steps > 1000:
            reward += 1.0
        
        # Update tracking variables
        self.prev_radius = self.player['radius']
        self.prev_score = self.score
        
        # Check if max steps reached
        truncated = self.steps >= self.max_steps
        
        # Final success bonus if we survived to max steps with high score
        if truncated and self.score > 500:
            reward += 100.0
            print(f"🏆 VICTORY! Final score: {self.score}")
        
        return self._get_obs(), reward, terminated, truncated, {'score': self.score}
    
    def render(self):
        """Render the environment (not implemented for headless training)"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
