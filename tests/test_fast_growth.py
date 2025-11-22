#!/usr/bin/env python3
"""
测试 Giant Agent 快速成长优化
"""

from agario_core.agents.giant_agent import GiantAgent

def test_fast_growth():
    """测试快速成长策略"""
    print("=" * 70)
    print("🚀 Giant Agent 快速成长测试")
    print("=" * 70)

    agent = GiantAgent(world_size=4000)

    # 测试场景1: 早期有食物和远处威胁
    print("\n📊 测试场景1: 早期 - 远处有威胁但应该继续吃豆")
    print("-" * 70)

    early_safe_food = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 20,
        'foods': [
            # 附近有很多食物
            {'x': 2050, 'y': 2000, 'radius': 5},
            {'x': 2100, 'y': 2050, 'radius': 6},
            {'x': 1950, 'y': 2050, 'radius': 5},
            {'x': 2000, 'y': 2100, 'radius': 7},
            {'x': 2050, 'y': 2100, 'radius': 4},
        ],
        'enemies': [
            # 远处有大威胁，但距离500+，不应该影响吃豆
            {'x': 2600, 'y': 2000, 'radius': 80, 'dx': 0, 'dy': 0},
            {'x': 1400, 'y': 2000, 'radius': 90, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(early_safe_food)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}°")
    print(f"  - 紧急模式: {'⚠️ 是（不应该）' if action['debug_info']['emergency_mode'] else '✅ 否（正确）'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 食物热点数: {action['debug_info']['food_hotspots']}")
    print(f"  - 期望: 应该朝向最近的食物，不触发紧急模式")

    # 测试场景2: 近处有安全食物
    print("\n" + "=" * 70)
    print("📊 测试场景2: 早期 - 近处有安全食物")
    print("-" * 70)

    close_safe_food = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 22,
        'foods': [
            # 非常近的食物
            {'x': 2030, 'y': 2000, 'radius': 6},  # 距离30
            {'x': 2000, 'y': 2040, 'radius': 5},  # 距离40
            {'x': 2060, 'y': 2060, 'radius': 7},  # 距离85
        ],
        'enemies': [
            # 威胁在300+距离外，安全
            {'x': 2400, 'y': 2400, 'radius': 70, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(close_safe_food)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该朝向最近的食物)")
    print(f"  - 紧急模式: {'⚠️ 是' if action['debug_info']['emergency_mode'] else '✅ 否'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 应该积极追最近的食物")

    # 测试场景3: 危险食物 vs 安全食物
    print("\n" + "=" * 70)
    print("📊 测试场景3: 早期 - 选择安全食物而非危险食物")
    print("-" * 70)

    dangerous_vs_safe = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 25,
        'foods': [
            # 近但危险的食物（大敌人守着）
            {'x': 2100, 'y': 2000, 'radius': 8},
            # 稍远但安全的食物
            {'x': 1900, 'y': 2000, 'radius': 6},
        ],
        'enemies': [
            # 大敌人守在右边的食物附近
            {'x': 2150, 'y': 2000, 'radius': 80, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(dangerous_vs_safe)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该朝左180°，选安全食物)")
    print(f"  - 紧急模式: {'⚠️ 是' if action['debug_info']['emergency_mode'] else '✅ 否'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 应该选择远离威胁的安全食物")

    # 测试场景4: 真正的紧急情况
    print("\n" + "=" * 70)
    print("📊 测试场景4: 真正紧急 - 巨大敌人近距离接近")
    print("-" * 70)

    real_emergency = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 20,
        'foods': [
            {'x': 2050, 'y': 2050, 'radius': 5},
        ],
        'enemies': [
            # 超大敌人近距离（200）正在接近
            {'x': 2200, 'y': 2000, 'radius': 100, 'dx': -3, 'dy': 0},
        ]
    }

    action = agent.get_action(real_emergency)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该朝左逃跑，约180°)")
    print(f"  - 紧急模式: {'✅ 是（正确）' if action['debug_info']['emergency_mode'] else '⚠️ 否（不应该）'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 应该触发紧急逃生")

    # 测试场景5: 食物大小优先级
    print("\n" + "=" * 70)
    print("📊 测试场景5: 早期 - 优先大食物")
    print("-" * 70)

    size_priority = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 23,
        'foods': [
            # 近但小的食物
            {'x': 2050, 'y': 2000, 'radius': 3},  # 距离50，小
            # 稍远但大的食物
            {'x': 2000, 'y': 2100, 'radius': 8},  # 距离100，大
        ],
        'enemies': []
    }

    action = agent.get_action(size_priority)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}°")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 应该考虑食物大小，可能朝大食物")

    # 参数验证
    print("\n" + "=" * 70)
    print("📊 快速成长参数验证")
    print("-" * 70)

    print("\n早期阶段参数:")
    params = agent.params['early']
    print(f"  🚀 食物吸引力: {params['food_attraction']} (高 = 快速吃豆)")
    print(f"  🛡️ 危险排斥力: {params['danger_repulsion']} (高 = 安全)")
    print(f"  🛡️ 危险距离: {params['danger_distance']} (感知范围)")
    print(f"  ⚖️ 风险规避: {params['risk_aversion']} (平衡值)")

    print("\n紧急逃生条件:")
    print(f"  - 触发距离: < 250 (之前是400)")
    print(f"  - 大小比例: < 0.65 (之前是0.7)")
    print(f"  - 说明: 更严格的条件，只在真正危险时触发")

    print("\n" + "=" * 70)
    print("✅ 快速成长测试完成！")
    print("=" * 70)
    print("\n🚀 主要优化:")
    print("  1. AI更新频率: 333ms → 100ms (每秒10次)")
    print("  2. 早期食物吸引力: 100 → 200 (+100%)")
    print("  3. 风险规避: 5.0 → 3.5 (-30%, 允许适度冒险)")
    print("  4. 食物安全评估: 避开危险食物")
    print("  5. 食物大小奖励: 优先大食物")
    print("  6. 紧急逃生: 更严格条件 (250距离, 0.65比例)")
    print("  7. 远距离排斥: 大幅减弱 (0.05倍)")
    print("=" * 70)

if __name__ == '__main__':
    test_fast_growth()
