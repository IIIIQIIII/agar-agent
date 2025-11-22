#!/usr/bin/env python3
"""
测试 Giant Agent 生存能力优化
"""

from agario_core.agents.giant_agent import GiantAgent

def test_survival_improvements():
    """测试生存能力改进"""
    print("=" * 70)
    print("🛡️ Giant Agent 生存能力测试")
    print("=" * 70)

    agent = GiantAgent(world_size=4000)

    # 测试场景1: 被大型敌人包围
    print("\n📊 测试场景1: 被大型敌人包围（紧急逃生）")
    print("-" * 70)

    dangerous_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 25,
        'foods': [
            {'x': 2050, 'y': 2050, 'radius': 5},  # 附近有食物但很危险
        ],
        'enemies': [
            # 四个方向都有大型威胁，距离很近
            {'x': 2150, 'y': 2000, 'radius': 80, 'dx': -1, 'dy': 0},   # 右边正在接近
            {'x': 1850, 'y': 2000, 'radius': 70, 'dx': 1, 'dy': 0},    # 左边正在接近
            {'x': 2000, 'y': 2150, 'radius': 75, 'dx': 0, 'dy': -1},   # 下边正在接近
            {'x': 2000, 'y': 1850, 'radius': 85, 'dx': 0, 'dy': 1},    # 上边正在接近
        ]
    }

    action = agent.get_action(dangerous_state)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}°")
    print(f"  - 是否分裂: {'是' if action['split'] else '否'}")
    print(f"  - 紧急模式: {'⚠️ 是' if action['debug_info']['emergency_mode'] else '否'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 应该触发紧急逃生，远离最近的威胁")

    # 测试场景2: 单个巨大威胁逼近
    print("\n" + "=" * 70)
    print("📊 测试场景2: 单个巨大威胁快速逼近")
    print("-" * 70)

    chase_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 20,
        'foods': [
            {'x': 2100, 'y': 2100, 'radius': 6},  # 食物在危险方向
        ],
        'enemies': [
            # 一个超大威胁正在快速接近
            {'x': 2250, 'y': 2000, 'radius': 100, 'dx': -5, 'dy': 0},
        ]
    }

    action = agent.get_action(chase_state)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该朝左逃跑，约180°)")
    print(f"  - 紧急模式: {'⚠️ 是' if action['debug_info']['emergency_mode'] else '否'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")

    # 测试场景3: 角落困境
    print("\n" + "=" * 70)
    print("📊 测试场景3: 被逼到角落")
    print("-" * 70)

    corner_state = {
        'player_x': 300,   # 接近左上角
        'player_y': 300,
        'player_radius': 30,
        'foods': [
            {'x': 200, 'y': 200, 'radius': 5},  # 角落里有食物
        ],
        'enemies': [
            {'x': 800, 'y': 800, 'radius': 60, 'dx': -1, 'dy': -1},  # 威胁从中心逼近
        ]
    }

    action = agent.get_action(corner_state)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该朝中心逃跑，约135°)")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 触发角落逃逸，向中心移动")

    # 测试场景4: 安全环境（不应触发紧急模式）
    print("\n" + "=" * 70)
    print("📊 测试场景4: 安全环境（正常模式）")
    print("-" * 70)

    safe_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 35,
        'foods': [
            {'x': 2100, 'y': 2000, 'radius': 5},
            {'x': 2000, 'y': 2100, 'radius': 5},
        ],
        'enemies': [
            # 远处有威胁，但距离安全
            {'x': 2600, 'y': 2600, 'radius': 80, 'dx': 0, 'dy': 0},
            # 附近有小敌人可以捕猎
            {'x': 2150, 'y': 2150, 'radius': 20, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(safe_state)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}°")
    print(f"  - 是否分裂: {'是' if action['split'] else '否'}")
    print(f"  - 紧急模式: {'⚠️ 是' if action['debug_info']['emergency_mode'] else '✅ 否'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 正常模式，追食物或猎物")

    # 测试场景5: 边界排斥测试
    print("\n" + "=" * 70)
    print("📊 测试场景5: 边界排斥力测试")
    print("-" * 70)

    boundary_state = {
        'player_x': 100,   # 非常靠近左边界
        'player_y': 2000,
        'player_radius': 25,
        'foods': [
            {'x': 50, 'y': 2000, 'radius': 8},  # 边界外有大食物
        ],
        'enemies': []
    }

    action = agent.get_action(boundary_state)
    print(f"决策结果:")
    print(f"  - 移动方向: {action['angle']:.1f}° (应该远离左边界，约0-90°)")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 被边界排斥，不会撞墙")

    # 测试场景6: 高风险禁止分裂
    print("\n" + "=" * 70)
    print("📊 测试场景6: 高风险环境禁止分裂")
    print("-" * 70)

    risky_split_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 50,  # 足够大可以分裂
        'foods': [],
        'enemies': [
            # 附近有小敌人可以吃
            {'x': 2150, 'y': 2000, 'radius': 25, 'dx': 0, 'dy': 0},
            # 但周围有多个大威胁
            {'x': 2300, 'y': 2300, 'radius': 70, 'dx': 0, 'dy': 0},
            {'x': 1700, 'y': 1700, 'radius': 65, 'dx': 0, 'dy': 0},
            {'x': 2400, 'y': 1600, 'radius': 75, 'dx': 0, 'dy': 0},
        ]
    }

    action = agent.get_action(risky_split_state)
    print(f"决策结果:")
    print(f"  - 是否分裂: {'⚠️ 是（不应该）' if action['split'] else '✅ 否（正确）'}")
    print(f"  - 风险评分: {action['debug_info']['risk_score']:.2f}")
    print(f"  - 期望: 风险过高，禁止分裂")

    # 参数验证
    print("\n" + "=" * 70)
    print("📊 防御参数验证")
    print("-" * 70)

    print("\n势场参数（生存优化版）:")
    for phase in ['early', 'mid', 'late']:
        params = agent.params[phase]
        print(f"\n{phase.upper()}:")
        print(f"  🛡️ 危险排斥力: {params['danger_repulsion']} (越高越安全)")
        print(f"  🛡️ 危险距离: {params['danger_distance']} (感知范围)")
        print(f"  🛡️ 风险规避: {params['risk_aversion']} (越高越保守)")
        print(f"  📍 食物吸引力: {params['food_attraction']}")
        print(f"  🎯 猎物吸引力: {params['prey_attraction']}")

    print("\n" + "=" * 70)
    print("✅ 生存能力测试完成！")
    print("=" * 70)
    print("\n🛡️ 主要改进:")
    print("  1. 紧急逃生系统：遇到极度危险时全力逃跑")
    print("  2. 增强危险排斥：距离越近排斥力呈指数增长")
    print("  3. 接近检测：正在靠近的敌人双倍排斥")
    print("  4. 角落逃逸：被困在角落时强制向中心移动")
    print("  5. 风险禁止分裂：高风险环境不分裂")
    print("  6. 分裂保守策略：更高的安全阈值")
    print("=" * 70)

if __name__ == '__main__':
    test_survival_improvements()
