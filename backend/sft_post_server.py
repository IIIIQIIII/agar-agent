"""
Flask server for SFT Post-Training Agent API
Serves the trained PPO model that was initialized from SFT and further trained with RL
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure AgarioEnv is available for unpickling if needed
from agario_core.envs.rl_env import AgarioEnv
import numpy as np

app = Flask(__name__)
CORS(app)

# Global model instance
model = None
model_path = None

# Statistics
stats = {
    'total_decisions': 0,
    'split_decisions': 0,
    'action_distribution': {
        'counts': [0] * 16,
        'percentages': [0.0] * 16
    }
}


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': model_path
    })


@app.route('/api/action', methods=['POST'])
def get_action():
    """Get action from trained SFT Post-Training model"""
    global stats

    try:
        data = request.json
        observation = np.array(data['observation'], dtype=np.float32)

        if model is None:
            return jsonify({'error': 'No model loaded. Please load a model first.'}), 400

        # Get action from model
        action, _states = model.predict(observation, deterministic=True)
        action = int(action)

        # Update statistics
        stats['total_decisions'] += 1
        stats['action_distribution']['counts'][action] += 1

        # Update percentages
        total = stats['total_decisions']
        stats['action_distribution']['percentages'] = [
            (count / total * 100) for count in stats['action_distribution']['counts']
        ]

        # Decode action to angle and split
        direction_idx = action % 8
        should_split = action >= 8
        angle = direction_idx * 45  # degrees

        if should_split:
            stats['split_decisions'] += 1

        return jsonify({
            'action': action,
            'angle': angle,
            'split': bool(should_split)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load SFT Post-Training model"""
    global model, model_path

    try:
        data = request.json or {}
        path = data.get('path', 'models/sft_post_training/best_model')

        # Check if model exists
        if not os.path.exists(f"{path}.zip"):
            return jsonify({
                'error': f'Model not found at {path}.zip',
                'hint': 'Make sure you have trained the SFT Post-Training model first'
            }), 404

        # Load model
        model = PPO.load(path)
        model_path = path

        return jsonify({
            'status': 'loaded',
            'path': path,
            'model_info': {
                'type': 'PPO',
                'description': 'SFT Post-Training model (initialized from SFT, trained with RL)',
                'action_space': 16,
                'observation_space': 49
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    split_rate = stats['split_decisions'] / max(1, stats['total_decisions'])

    return jsonify({
        'total_decisions': stats['total_decisions'],
        'split_rate': split_rate,
        'action_distribution': stats['action_distribution'],
        'model_path': model_path,
        'model_type': 'SFT Post-Training (PPO)'
    })


@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """Reset statistics"""
    global stats

    stats = {
        'total_decisions': 0,
        'split_decisions': 0,
        'action_distribution': {
            'counts': [0] * 16,
            'percentages': [0.0] * 16
        }
    }

    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SFT Post-Training Server")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /api/health          - Health check")
    print("  POST /api/action          - Get action from model")
    print("  POST /api/model/load      - Load model")
    print("  GET  /api/stats           - Get statistics")
    print("  POST /api/stats/reset     - Reset statistics")
    print("\nDefault model path: models/sft_post_training/best_model")
    print("\nStarting server on http://localhost:5004...")
    print("=" * 70)

    # Try to load default model
    default_path = 'models/sft_post_training/best_model'
    if os.path.exists(f"{default_path}.zip"):
        try:
            model = PPO.load(default_path)
            model_path = default_path
            print(f"\n✅ Default model loaded: {default_path}\n")
        except Exception as e:
            print(f"\n⚠️  Could not load default model: {e}\n")
    else:
        print(f"\n⚠️  Default model not found at {default_path}.zip")
        print("   You can load a model using POST /api/model/load\n")

    app.run(host='0.0.0.0', port=5004, debug=True, threaded=True)
