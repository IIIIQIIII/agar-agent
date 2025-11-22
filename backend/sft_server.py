"""
SFT Agent Server - Flask API for Behavior Cloning Model
使用从Giant Agent学习的监督微调模型进行决策
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agario_core.models.behavior_cloner import BehaviorCloner
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 全局模型实例
sft_model = None
model_loaded = False

# 统计数据
stats = {
    'total_decisions': 0,
    'split_decisions': 0,
    'max_radius_seen': 0,
    'action_distribution': [0] * 16,  # 16个动作
    'model_path': None
}


def load_default_model():
    """尝试加载默认模型"""
    global sft_model, model_loaded, stats

    default_paths = [
        'models/sft_giant/best_model.pt',
        'models/sft_giant/final_model.pt'
    ]

    # 训练时使用的架构是 [256, 256, 128]
    hidden_dims = [256, 256, 128]

    for path in default_paths:
        if os.path.exists(path):
            try:
                print(f"Loading SFT model from: {path}")
                sft_model = BehaviorCloner(
                    obs_dim=49,
                    action_dim=16,
                    hidden_dims=hidden_dims  # 使用正确的架构
                )
                sft_model.load(path)
                model_loaded = True
                stats['model_path'] = path
                print(f"✅ SFT model loaded successfully from {path}")
                return True
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")

    print("⚠️  No SFT model found. Please train a model first.")
    return False


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'agent_type': 'SFT_Agent',
        'version': '1.0',
        'model_loaded': model_loaded,
        'model_path': stats['model_path'],
        'requires_training': True,
        'description': 'Behavior Cloning model trained from Giant Agent demonstrations'
    })


@app.route('/api/action', methods=['POST'])
def get_action():
    """
    获取SFT模型决策

    Request JSON:
    {
        "observation": [float] * 49  # 标准化观察向量
    }

    Response JSON:
    {
        "action": int (0-15),
        "angle": float (0-360),
        "split": bool
    }
    """
    try:
        # 检查模型是否加载
        if not model_loaded or sft_model is None:
            error_msg = 'No model loaded. Please load a model first.'
            print(f"❌ Error: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # 获取请求数据
        data = request.json

        if data is None:
            error_msg = 'No JSON data received'
            print(f"❌ Error: {error_msg}")
            print(f"   Request data: {request.data}")
            print(f"   Content-Type: {request.content_type}")
            return jsonify({'error': error_msg}), 400

        # 检查observation字段
        if 'observation' not in data:
            error_msg = f'Missing observation field. Received keys: {list(data.keys())}'
            print(f"❌ Error: {error_msg}")
            print(f"   Full data: {data}")
            return jsonify({'error': 'Missing observation field'}), 400

        # 转换为numpy数组
        observation = np.array(data['observation'], dtype=np.float32)

        # 检查维度
        if observation.shape[0] != 49:
            error_msg = f'Invalid observation shape. Expected 49, got {observation.shape[0]}'
            print(f"❌ Error: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # 使用模型预测动作
        action = sft_model.predict(observation)

        # 解码动作: 0-7为8个方向, 8-15为8个方向+分裂
        direction_idx = action % 8
        should_split = action >= 8
        angle = direction_idx * 45  # 转换为度数

        # 更新统计
        stats['total_decisions'] += 1
        stats['action_distribution'][action] += 1
        if should_split:
            stats['split_decisions'] += 1

        return jsonify({
            'action': int(action),
            'angle': float(angle),
            'split': bool(should_split)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """
    加载SFT模型

    Request JSON:
    {
        "path": "models/sft_giant/best_model.pt"
    }
    """
    global sft_model, model_loaded, stats

    try:
        data = request.json or {}
        path = data.get('path', 'models/sft_giant/best_model.pt')

        if not os.path.exists(path):
            return jsonify({'error': f'Model file not found: {path}'}), 404

        print(f"Loading SFT model from: {path}")
        # 使用训练时的架构
        sft_model = BehaviorCloner(
            obs_dim=49,
            action_dim=16,
            hidden_dims=[256, 256, 128]  # 匹配训练时的架构
        )
        sft_model.load(path)
        model_loaded = True
        stats['model_path'] = path

        return jsonify({
            'status': 'loaded',
            'path': path,
            'model_info': {
                'obs_dim': 49,
                'action_dim': 16,
                'hidden_dims': [256, 256, 128],
                'description': 'Behavior Cloning from Giant Agent'
            }
        })

    except Exception as e:
        model_loaded = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if not model_loaded or sft_model is None:
        return jsonify({'error': 'No model loaded'}), 400

    return jsonify({
        'loaded': model_loaded,
        'path': stats['model_path'],
        'architecture': {
            'type': 'BehaviorCloner',
            'obs_dim': 49,
            'action_dim': 16,
            'hidden_layers': [256, 256, 128]
        },
        'training_source': 'Giant Agent demonstrations',
        'action_space': {
            '0-7': 'Move in 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)',
            '8-15': 'Move in 8 directions + Split'
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    total = max(stats['total_decisions'], 1)

    # 计算动作分布百分比
    action_dist_pct = [count / total * 100 for count in stats['action_distribution']]

    # 计算分裂动作比例（动作8-15）
    split_action_count = sum(stats['action_distribution'][8:])

    return jsonify({
        'total_decisions': stats['total_decisions'],
        'split_decisions': stats['split_decisions'],
        'split_rate': stats['split_decisions'] / total,
        'max_radius_seen': stats['max_radius_seen'],
        'action_distribution': {
            'counts': stats['action_distribution'],
            'percentages': action_dist_pct
        },
        'model_path': stats['model_path'],
        'split_action_ratio': split_action_count / total
    })


@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """重置统计数据"""
    stats['total_decisions'] = 0
    stats['split_decisions'] = 0
    stats['max_radius_seen'] = 0
    stats['action_distribution'] = [0] * 16
    return jsonify({'status': 'reset'})


@app.route('/api/model/list', methods=['GET'])
def list_models():
    """列出可用的SFT模型"""
    models = []

    # 检查常见路径
    search_paths = [
        'models/sft_giant',
        'models'
    ]

    for base_path in search_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.pt'):
                        full_path = os.path.join(root, file)
                        # 尝试获取文件大小
                        try:
                            size_mb = os.path.getsize(full_path) / (1024 * 1024)
                            models.append({
                                'path': full_path,
                                'name': file,
                                'size_mb': round(size_mb, 2)
                            })
                        except:
                            pass

    return jsonify({
        'models': models,
        'count': len(models)
    })


@app.route('/api/test', methods=['GET'])
def test():
    """测试端点 - 使用随机观察"""
    if not model_loaded or sft_model is None:
        return jsonify({'error': 'No model loaded'}), 400

    # 生成测试观察向量
    test_obs = np.random.randn(49).astype(np.float32)
    test_obs = np.clip(test_obs, -1, 1)  # 限制在合理范围内

    action = sft_model.predict(test_obs)
    direction_idx = action % 8
    should_split = action >= 8
    angle = direction_idx * 45

    return jsonify({
        'test_observation_shape': test_obs.shape,
        'action': int(action),
        'angle': float(angle),
        'split': bool(should_split),
        'note': 'This is a test with random observation'
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🤖 SFT Agent Server 启动中...")
    print("=" * 70)
    print("\n📡 API端点:")
    print("  POST /api/action        - 获取SFT模型决策")
    print("  GET  /api/stats         - 查看统计数据")
    print("  POST /api/stats/reset   - 重置统计")
    print("  POST /api/model/load    - 加载模型")
    print("  GET  /api/model/info    - 模型信息")
    print("  GET  /api/model/list    - 列出可用模型")
    print("  GET  /api/test          - 测试端点")
    print("  GET  /api/health        - 健康检查")
    print("\n🎯 SFT特性:")
    print("  ✅ 从Giant Agent专家演示学习")
    print("  ✅ 神经网络行为克隆")
    print("  ✅ 49维观察空间 → 16个动作")
    print("  ✅ 快速推理，无需实时计算")
    print("\n" + "=" * 70)

    # 尝试加载默认模型
    load_default_model()

    print(f"🌐 服务器运行在: http://localhost:5003")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
