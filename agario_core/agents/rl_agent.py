import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from agario_core.envs.rl_env import AgarioEnv


class TrainingCallback(BaseCallback):
    """Callback for logging training metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        return True


class AgarioAgent:
    """RL Agent for Agar.io game"""
    
    def __init__(self, model_path='models/rl_agent/agario_ppo'):
        self.model_path = model_path
        self.env = None
        self.model = None
        self.callback = TrainingCallback()
        
    def create_env(self):
        """Create training environment"""
        self.env = AgarioEnv()
        return self.env
    
    def create_model(self):
        """Create new PPO model"""
        if self.env is None:
            self.create_env()
        
        self.model = PPO(
            'MlpPolicy',
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        return self.model
    
    def train(self, total_timesteps=100000):
        """Train the agent"""
        if self.model is None:
            self.create_model()
        
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=False  # Disable progress bar to avoid tqdm/rich dependency
        )
        print("Training completed!")
        
        return {
            'episode_rewards': self.callback.episode_rewards,
            'episode_lengths': self.callback.episode_lengths,
        }
    
    def save_model(self, path=None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, path=None):
        """Load trained model"""
        load_path = path or self.model_path
        
        if not os.path.exists(f"{load_path}.zip"):
            raise FileNotFoundError(f"Model not found at {load_path}")
        
        if self.env is None:
            self.create_env()
        
        self.model = PPO.load(load_path, env=self.env)
        print(f"Model loaded from {load_path}")
        return self.model
    
    def predict(self, observation, deterministic=True):
        """Get action from trained model"""
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent"""
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        episode_rewards = []
        episode_scores = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_scores.append(info['score'])
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Score={info['score']}")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_score': np.mean(episode_scores),
            'std_score': np.std(episode_scores),
        }


if __name__ == '__main__':
    # Example usage
    agent = AgarioAgent()
    
    # Train
    print("Training agent...")
    metrics = agent.train(total_timesteps=50000)
    
    # Save
    agent.save_model()
    
    # Evaluate
    print("\nEvaluating agent...")
    eval_results = agent.evaluate(num_episodes=5)
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Mean Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}")
