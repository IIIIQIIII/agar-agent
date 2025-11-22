"""
Rule-Based Agar.io Agent - Mathematical Modeling Approach
No training required, pure algorithmic decision making
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class RuleBasedAgent:
    """
    Mathematical model-based agent using:
    - Potential field theory for navigation
    - Risk assessment for survival
    - Greedy optimization for target selection
    - Game theory for split decisions
    """
    
    def __init__(self, world_size=4000):
        self.world_size = world_size
        
        # Tunable parameters (based on mathematical modeling)
        self.FOOD_ATTRACTION = 50.0      # Attraction strength
        self.DANGER_REPULSION = 200.0    # Repulsion from large enemies
        self.PREY_ATTRACTION = 150.0     # Attraction to small enemies
        self.MIN_SAFE_DISTANCE = 200.0   # Safety margin
        self.SPLIT_THRESHOLD = 40.0      # Min size to consider split
        
    def calculate_potential_field(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        foods: List[Dict],
        enemies: List[Dict]
    ) -> Tuple[float, float]:
        """
        Calculate movement direction using potential field method
        
        Physics-inspired: Each object creates a force field
        - Attractive forces: pull player toward food/prey
        - Repulsive forces: push player away from danger
        
        Returns: (dx, dy) normalized direction vector
        """
        total_force_x = 0.0
        total_force_y = 0.0
        
        px, py = player_pos
        
        # 1. FOOD ATTRACTION (inverse square law)
        for food in foods:
            dx = food['x'] - px
            dy = food['y'] - py
            dist = np.hypot(dx, dy)
            
            if dist < 1e-5:
                continue
            
            # Attraction force ∝ 1/r²
            force_magnitude = self.FOOD_ATTRACTION / (dist ** 1.5)
            total_force_x += (dx / dist) * force_magnitude
            total_force_y += (dy / dist) * force_magnitude
        
        # 2. ENEMY INTERACTIONS (size-dependent)
        for enemy in enemies:
            dx = enemy['x'] - px
            dy = enemy['y'] - py
            dist = np.hypot(dx, dy)
            
            if dist < 1e-5:
                continue
            
            size_ratio = player_radius / enemy['radius']
            
            if size_ratio > 1.15:  # Prey (smaller enemy)
                # ATTRACTION to prey
                # Stronger attraction for bigger size advantage
                advantage_factor = (size_ratio - 1.0) * 2.0
                force_magnitude = self.PREY_ATTRACTION * advantage_factor / (dist ** 1.2)
                total_force_x += (dx / dist) * force_magnitude
                total_force_y += (dy / dist) * force_magnitude
                
            elif size_ratio < 0.9:  # Danger (larger enemy)
                # REPULSION from danger
                # Stronger repulsion when closer and much smaller
                danger_factor = (0.9 - size_ratio) * 3.0
                
                # Inverse cube law for strong short-range repulsion
                force_magnitude = self.DANGER_REPULSION * danger_factor / (dist ** 2.0)
                total_force_x -= (dx / dist) * force_magnitude
                total_force_y -= (dy / dist) * force_magnitude
        
        # 3. BOUNDARY REPULSION (keep away from edges)
        edge_margin = 300
        
        if px < edge_margin:
            total_force_x += 100.0 / (px + 1)
        if px > self.world_size - edge_margin:
            total_force_x -= 100.0 / (self.world_size - px + 1)
        if py < edge_margin:
            total_force_y += 100.0 / (py + 1)
        if py > self.world_size - edge_margin:
            total_force_y -= 100.0 / (self.world_size - py + 1)
        
        # Normalize direction
        magnitude = np.hypot(total_force_x, total_force_y)
        if magnitude < 1e-5:
            return 0.0, 0.0
        
        return total_force_x / magnitude, total_force_y / magnitude
    
    def assess_risk(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        enemies: List[Dict]
    ) -> float:
        """
        Calculate risk score based on proximity to dangerous enemies
        
        Risk model: Σ (danger_level × proximity_factor)
        Returns: Risk score [0, ∞), higher = more dangerous
        """
        total_risk = 0.0
        px, py = player_pos
        
        for enemy in enemies:
            dist = np.hypot(enemy['x'] - px, enemy['y'] - py)
            size_ratio = player_radius / enemy['radius']
            
            # Only consider enemies that are dangerous
            if size_ratio < 0.95:
                # Danger increases with size difference
                danger_level = (0.95 - size_ratio) * 10.0
                
                # Proximity factor: exponential decay
                proximity = np.exp(-dist / 300.0)
                
                total_risk += danger_level * proximity
        
        return total_risk
    
    def select_best_target(
        self,
        player_pos: Tuple[float, float],
        player_radius: float,
        foods: List[Dict],
        enemies: List[Dict]
    ) -> Optional[Tuple[float, float, str]]:
        """
        Greedy target selection with risk-reward analysis
        
        Optimization: maximize (reward / cost)
        where cost = distance + risk
        
        Returns: (target_x, target_y, target_type) or None
        """
        px, py = player_pos
        best_score = -float('inf')
        best_target = None
        
        # Evaluate food targets
        for food in foods[:20]:  # Limit search for efficiency
            dist = np.hypot(food['x'] - px, food['y'] - py)
            if dist < 1:
                continue
            
            # Reward: food value (radius²)
            reward = food['radius'] ** 2
            
            # Cost: distance + local risk
            local_risk = sum(
                1.0 for e in enemies 
                if np.hypot(e['x'] - food['x'], e['y'] - food['y']) < 400
                and e['radius'] > player_radius * 1.1
            )
            cost = dist + local_risk * 100
            
            score = reward / max(cost, 1.0)
            
            if score > best_score:
                best_score = score
                best_target = (food['x'], food['y'], 'food')
        
        # Evaluate enemy targets (prey)
        for enemy in enemies:
            size_ratio = player_radius / enemy['radius']
            
            if size_ratio > 1.2:  # Only chase if clearly larger
                dist = np.hypot(enemy['x'] - px, enemy['y'] - py)
                if dist < 1:
                    continue
                
                # Reward: enemy mass (much higher than food)
                reward = (enemy['radius'] ** 2) * 5.0 * (size_ratio - 1.0)
                
                # Cost: distance
                cost = dist
                
                score = reward / max(cost, 1.0)
                
                if score > best_score:
                    best_score = score
                    best_target = (enemy['x'], enemy['y'], 'enemy')
        
        return best_target
    
    def should_split(
        self,
        player_radius: float,
        target: Optional[Tuple[float, float, str]],
        player_pos: Tuple[float, float],
        enemies: List[Dict]
    ) -> bool:
        """
        Strategic split decision using g theory
        
        Split when:
        1. Large enough to split (radius > threshold)
        2. Can catch valuable prey by splitting
        3. Low risk environment
        4. Target is close enough
        
        Returns: True if should split
        """
        # Rule 1: Minimum size requirement
        if player_radius < self.SPLIT_THRESHOLD:
            return False
        
        # Rule 2: Low risk environment
        risk = self.assess_risk(player_pos, player_radius, enemies)
        if risk > 5.0:
            return False
        
        # Rule 3: Have a valuable target
        if target is None:
            return False
        
        tx, ty, ttype = target
        dist_to_target = np.hypot(tx - player_pos[0], ty - player_pos[1])
        
        # Rule 4: Target is in split range (can catch with split)
        split_range = player_radius * 8  # Approximate split distance
        
        if ttype == 'enemy' and dist_to_target < split_range:
            # Game theory: Expected value of split
            # EV = P(catch) × reward - P(fail) × cost
            
            catch_prob = max(0.0, 1.0 - dist_to_target / split_range)
            
            if catch_prob > 0.6:  # High confidence catch
                return True
        
        return False
    
    def get_action(self, game_state: Dict) -> Dict:
        """
        Main decision function: combines all mathematical models
        
        Args:
            game_state: {
                'player_x': float,
                'player_y': float,
                'player_radius': float,
                'foods': List[Dict],
                'enemies': List[Dict]
            }
        
        Returns:
            {
                'angle': float (degrees),
                'split': bool
            }
        """
        player_pos = (game_state['player_x'], game_state['player_y'])
        player_radius = game_state['player_radius']
        foods = game_state['foods']
        enemies = game_state['enemies']
        
        # Step 1: Calculate movement direction (potential field)
        dx, dy = self.calculate_potential_field(
            player_pos, player_radius, foods, enemies
        )
        
        # Step 2: Select best target (greedy optimization)
        target = self.select_best_target(
            player_pos, player_radius, foods, enemies
        )
        
        # Step 3: Decide on split (game theory)
        should_split = self.should_split(
            player_radius, target, player_pos, enemies
        )
        
        # Convert direction to angle
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
        
        return {
            'angle': angle,
            'split': should_split
        }


# Example usage and testing
if __name__ == '__main__':
    agent = RuleBasedAgent()
    
    # Test with sample game state
    game_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 30,
        'foods': [
            {'x': 2100, 'y': 2050, 'radius': 5},
            {'x': 1950, 'y': 1980, 'radius': 4},
        ],
        'enemies': [
            {'x': 2200, 'y': 2100, 'radius': 50},  # Dangerous
            {'x': 1900, 'y': 1900, 'radius': 20},  # Prey
        ]
    }
    
    action = agent.get_action(game_state)
    print(f"Decision: Move {action['angle']:.1f}°, Split: {action['split']}")
    
    # Expected: Should avoid large enemy, hunt small one or go for food
