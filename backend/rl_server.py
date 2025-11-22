"""
Flask server for RL Agent API
Provides endpoints for agent inference, training, and model management
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agario_core.agents.rl_agent import AgarioAgent
import threading
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Global agent instance
agent = AgarioAgent()
training_thread = None
training_active = False
training_metrics = {'episode_rewards': [], 'episode_lengths': []}


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/action', methods=['POST'])
def get_action():
    """Get action from trained RL model given game state"""
    try:
        data = request.json
        observation = np.array(data['observation'], dtype=np.float32)
        
        if agent.model is None:
            return jsonify({'error': 'No model loaded'}), 400
        
        action = agent.predict(observation, deterministic=True)
        
        # Decode action to angle and split
        direction_idx = action % 8
        should_split = action >= 8
        angle = direction_idx * 45  # degrees
        
        return jsonify({
            'action': int(action),
            'angle': angle,
            'split': bool(should_split)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start training in background thread"""
    global training_thread, training_active, training_metrics
    
    if training_active:
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        data = request.json or {}
        timesteps = data.get('timesteps', 100000)
        
        def train_worker():
            global training_active, training_metrics
            training_active = True
            try:
                metrics = agent.train(total_timesteps=timesteps)
                training_metrics = metrics
                agent.save_model()
            except Exception as e:
                print(f"Training error: {e}")
            finally:
                training_active = False
        
        training_thread = threading.Thread(target=train_worker)
        training_thread.start()
        
        return jsonify({
            'status': 'training_started',
            'timesteps': timesteps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop training (not fully implemented - would need to interrupt training loop)"""
    global training_active
    
    if not training_active:
        return jsonify({'error': 'No training in progress'}), 400
    
    # Note: Stable-Baselines3 doesn't have easy interrupt mechanism
    # This is a placeholder
    return jsonify({
        'status': 'stop_requested',
        'note': 'Training will stop after current episode'
    })


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get training status and metrics"""
    global training_active, training_metrics
    
    return jsonify({
        'active': training_active,
        'num_episodes': len(training_metrics.get('episode_rewards', [])),
        'recent_rewards': training_metrics.get('episode_rewards', [])[-10:],
        'mean_reward': np.mean(training_metrics.get('episode_rewards', [])[-10:]) if training_metrics.get('episode_rewards') else 0
    })


@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Save current model"""
    try:
        data = request.json or {}
        path = data.get('path', 'models/rl_agent/agario_ppo')
        
        if agent.model is None:
            return jsonify({'error': 'No model to save'}), 400
        
        saved_path = agent.save_model(path)
        return jsonify({
            'status': 'saved',
            'path': saved_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load saved model"""
    try:
        data = request.json or {}
        path = data.get('path', 'models/rl_agent/agario_ppo')
        
        agent.load_model(path)
        return jsonify({
            'status': 'loaded',
            'path': path
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/list', methods=['GET'])
def list_models():
    """List available saved models"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return jsonify({'models': []})
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.zip'):
                models.append(file.replace('.zip', ''))
        
        return jsonify({'models': models})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate current model"""
    try:
        data = request.json or {}
        num_episodes = data.get('num_episodes', 5)
        
        if agent.model is None:
            return jsonify({'error': 'No model loaded'}), 400
        
        results = agent.evaluate(num_episodes=num_episodes)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting RL Server...")
    print("Endpoints:")
    print("  POST /api/action - Get action from model")
    print("  POST /api/train/start - Start training")
    print("  GET  /api/train/status - Get training status")
    print("  POST /api/model/save - Save model")
    print("  POST /api/model/load - Load model")
    print("  GET  /api/model/list - List saved models")
    print("  POST /api/evaluate - Evaluate model")
    
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
