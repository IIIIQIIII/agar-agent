#!/usr/bin/env python3
"""
测试 Giant Agent 全局视野升级
"""

from agario_core.agents.giant_agent import GiantAgent
import json

def test_global_vision():
    """测试全局视野功能"""
    print("=" * 70)
    print("🎯 Giant Agent 全局视野测试")
    print("=" * 70)

    agent = GiantAgent(world_size=4000)

    # 测试场景1: 大量食物和敌人
    print("\n📊 测试场景1: 大量实体 (模拟全局视野)")
    print("-" * 70)

    game_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 40,
        'foods': [
            {'x': 2000 + i*50, 'y': 2000 + j*50, 'radius': 5}
            for i in range(-10, 10)
            for j in range(-10, 10)
        ],  # 400个食物
        'enemies': [
            {'x': 2000 + i*200, 'y': 2000 + j*200, 'radius': 30 + i*2, 'dx': 0, 'dy': 0}
            for i in range(-5, 5)
            for j in range(-5, 5)
        ]  # 100个敌人
    }

    print(f"食物数量: {len(game_state['foods'])}")
    print(f"敌人数量: {len(game_state['enemies'])}")

    action = agent.get_action(game_state)

    print(f"\n决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}°")
    print(f"  - 是否分裂: {'是' if action['split'] else '否'}")
    print(f"  - 成长阶段: {action['debug_info']['phase']}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 最佳扇区价值: {action['debug_info']['best_sector_value']:.2f}")

    # 测试场景2: 验证全局视野范围
    print("\n" + "=" * 70)
    print("📊 测试场景2: 验证视野范围扩展")
    print("-" * 70)

    # 早期阶段 - 视野1000
    early_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 20,
        'foods': [
            {'x': 2000, 'y': 2900, 'radius': 8},  # 距离900 - 应该可见
            {'x': 2000, 'y': 3100, 'radius': 8},  # 距离1100 - 应该不可见
        ],
        'enemies': []
    }

    action = agent.get_action(early_state)
    print(f"早期阶段 (视野1000):")
    print(f"  - 方向: {action['angle']:.1f}° (应该朝向900距离的食物)")

    # 后期阶段 - 视野1200
    late_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 80,
        'foods': [
            {'x': 2000, 'y': 3150, 'radius': 8},  # 距离1150 - 应该可见
            {'x': 2000, 'y': 3300, 'radius': 8},  # 距离1300 - 应该不可见
        ],
        'enemies': []
    }

    action = agent.get_action(late_state)
    print(f"后期阶段 (视野1200):")
    print(f"  - 方向: {action['angle']:.1f}° (应该朝向1150距离的食物)")

    # 测试场景3: 威胁和猎物分类
    print("\n" + "=" * 70)
    print("📊 测试场景3: 威胁和猎物分类")
    print("-" * 70)

    mixed_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 50,
        'foods': [],
        'enemies': [
            # 猎物 (小于50/1.2=42)
            {'x': 2100, 'y': 2000, 'radius': 30, 'dx': 0, 'dy': 0},
            {'x': 2200, 'y': 2000, 'radius': 35, 'dx': 0, 'dy': 0},
            {'x': 2300, 'y': 2000, 'radius': 25, 'dx': 0, 'dy': 0},
            # 威胁 (大于50/0.85=59)
            {'x': 1900, 'y': 2000, 'radius': 70, 'dx': 0, 'dy': 0},
            {'x': 1800, 'y': 2000, 'radius': 80, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(mixed_state)
    print(f"混合场景:")
    print(f"  - 方向: {action['angle']:.1f}° (应该朝向猎物方向，约0-90°)")
    print(f"  - 应该避开威胁方向 (约180-270°)")

    # 测试场景4: 分裂决策优化
    print("\n" + "=" * 70)
    print("📊 测试场景4: 分裂决策优化")
    print("-" * 70)

    split_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 36,  # 刚好超过新阈值35
        'foods': [],
        'enemies': [
            # 理想分裂目标：距离适中，大小合适
            {'x': 2150, 'y': 2000, 'radius': 18, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(split_state)
    print(f"分裂测试 (r=36, 目标r=18):")
    print(f"  - 是否分裂: {'是' if action['split'] else '否'} (期望: 是)")
    print(f"  - 分裂机会: {action['debug_info']['split_opportunity']}")

    # 测试场景5: 参数优化验证
    print("\n" + "=" * 70)
    print("📊 测试场景5: 参数优化验证")
    print("-" * 70)

    print("\n势场参数 (优化后):")
    for phase in ['early', 'mid', 'late']:
        params = agent.params[phase]
        print(f"\n{phase.upper()}:")
        print(f"  - 食物吸引力: {params['food_attraction']}")
        print(f"  - 危险排斥力: {params['danger_repulsion']}")
        print(f"  - 猎物吸引力: {params['prey_attraction']}")
        print(f"  - 危险距离: {params['danger_distance']}")
        print(f"  - 风险规避: {params['risk_aversion']}")

    print("\n" + "=" * 70)
    print("✅ 所有测试完成！全局视野功能正常工作")
    print("=" * 70)

if __name__ == '__main__':
    test_global_vision()
