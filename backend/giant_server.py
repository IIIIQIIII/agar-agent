"""
巨无霸智能体服务器
Giant Agent Server - Flask API for mathematical model-based agent

不需要训练，直接使用数学建模算法决策
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agario_core.agents.giant_agent import GiantAgent
import numpy as np

app = Flask(__name__)
CORS(app)

# 全局智能体实例
giant_agent = GiantAgent(world_size=4000)

# 统计数据
stats = {
    'total_decisions': 0,
    'split_decisions': 0,
    'max_radius_seen': 0,
    'phase_counts': {'early': 0, 'mid': 0, 'late': 0}
}


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'agent_type': 'GiantAgent',
        'version': '1.0',
        'requires_training': False
    })


@app.route('/api/action', methods=['POST'])
def get_action():
    """
    获取智能体决策

    Request JSON:
    {
        "player_x": float,
        "player_y": float,
        "player_radius": float,
        "foods": [{"x": float, "y": float, "radius": float}, ...],
        "enemies": [{"x": float, "y": float, "radius": float, "dx": float?, "dy": float?}, ...]
    }

    Response JSON:
    {
        "angle": float (0-360),
        "split": bool,
        "debug_info": {...}
    }
    """
    try:
        game_state = request.json

        # 验证必需字段
        required_fields = ['player_x', 'player_y', 'player_radius']
        for field in required_fields:
            if field not in game_state:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # 确保foods和enemies存在
        game_state.setdefault('foods', [])
        game_state.setdefault('enemies', [])

        # 调用智能体决策
        action = giant_agent.get_action(game_state)

        # 更新统计
        stats['total_decisions'] += 1
        if action['split']:
            stats['split_decisions'] += 1

        radius = game_state['player_radius']
        if radius > stats['max_radius_seen']:
            stats['max_radius_seen'] = radius

        phase = action['debug_info']['phase']
        stats['phase_counts'][phase] += 1

        return jsonify(action)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取智能体统计数据"""
    return jsonify({
        'total_decisions': stats['total_decisions'],
        'split_decisions': stats['split_decisions'],
        'split_rate': stats['split_decisions'] / max(stats['total_decisions'], 1),
        'max_radius_seen': stats['max_radius_seen'],
        'phase_distribution': stats['phase_counts']
    })


@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """重置统计数据"""
    stats['total_decisions'] = 0
    stats['split_decisions'] = 0
    stats['max_radius_seen'] = 0
    stats['phase_counts'] = {'early': 0, 'mid': 0, 'late': 0}
    return jsonify({'status': 'reset'})


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取智能体配置参数"""
    return jsonify({
        'world_size': giant_agent.world_size,
        'early_phase_threshold': giant_agent.EARLY_PHASE_THRESHOLD,
        'mid_phase_threshold': giant_agent.MID_PHASE_THRESHOLD,
        'parameters': giant_agent.params
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """
    更新智能体配置

    Request JSON:
    {
        "phase": "early" | "mid" | "late",
        "params": {
            "food_attraction": float,
            "danger_repulsion": float,
            ...
        }
    }
    """
    try:
        data = request.json
        phase = data.get('phase')
        params = data.get('params')

        if phase not in ['early', 'mid', 'late']:
            return jsonify({'error': 'Invalid phase. Must be early, mid, or late'}), 400

        if params:
            giant_agent.params[phase].update(params)

        return jsonify({
            'status': 'updated',
            'phase': phase,
            'new_params': giant_agent.params[phase]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_state():
    """
    分析游戏状态（调试用）

    返回详细的分析数据，包括：
    - 食物密度热点
    - 扇区价值分析
    - 风险评估
    - 推荐策略
    """
    try:
        game_state = request.json

        player_pos = (game_state['player_x'], game_state['player_y'])
        player_radius = game_state['player_radius']
        foods = game_state.get('foods', [])
        enemies = game_state.get('enemies', [])

        # 分析食物密度
        food_density = giant_agent.analyze_food_density(foods)

        # 分析扇区
        sectors = giant_agent.analyze_sector_values(
            player_pos, player_radius, foods, enemies
        )

        # 评估风险
        risk = giant_agent.calculate_risk_score(player_pos, player_radius, enemies)

        # 分裂机会
        should_split, split_target = giant_agent.evaluate_split_opportunity(
            player_pos, player_radius, enemies
        )

        # 成长阶段
        phase = giant_agent.get_phase(player_radius)

        return jsonify({
            'phase': phase,
            'risk_score': risk,
            'food_hotspots': [
                {'x': h[0], 'y': h[1], 'density': h[2]}
                for h in food_density['hotspots']
            ],
            'best_food_region': {
                'x': food_density['best_region'][0],
                'y': food_density['best_region'][1]
            },
            'sectors': [
                {
                    'angle': s['angle'],
                    'food_value': s['food_value'],
                    'threat_value': s['threat_value'],
                    'total_value': s['total_value']
                }
                for s in sectors
            ],
            'split_opportunity': {
                'should_split': should_split,
                'target': {'x': split_target[0], 'y': split_target[1]} if split_target else None
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# 测试端点
@app.route('/api/test', methods=['GET'])
def test():
    """测试端点 - 使用示例数据"""
    test_state = {
        'player_x': 2000,
        'player_y': 2000,
        'player_radius': 30,
        'foods': [
            {'x': 2100, 'y': 2050, 'radius': 5},
            {'x': 1950, 'y': 1980, 'radius': 4},
        ],
        'enemies': [
            {'x': 2200, 'y': 2100, 'radius': 50, 'dx': -1, 'dy': 0},
            {'x': 1900, 'y': 1900, 'radius': 20, 'dx': 0.5, 'dy': 0.5},
        ]
    }

    action = giant_agent.get_action(test_state)
    return jsonify({
        'test_state': test_state,
        'action': action
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🎮 巨无霸智能体服务器启动中...")
    print("=" * 70)
    print("\n📡 API端点:")
    print("  POST /api/action      - 获取智能体决策")
    print("  GET  /api/stats       - 查看统计数据")
    print("  POST /api/stats/reset - 重置统计")
    print("  GET  /api/config      - 查看配置")
    print("  POST /api/config      - 更新配置")
    print("  POST /api/analyze     - 分析游戏状态")
    print("  GET  /api/test        - 测试端点")
    print("  GET  /api/health      - 健康检查")
    print("\n🚀 智能体特性:")
    print("  ✅ 无需训练，纯数学建模")
    print("  ✅ 多阶段自适应策略（早期/中期/后期）")
    print("  ✅ 势场理论 + 博弈论 + 空间分析")
    print("  ✅ 实时风险评估与决策优化")
    print("\n" + "=" * 70)
    print(f"🌐 服务器运行在: http://localhost:5002")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)
