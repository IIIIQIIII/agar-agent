#!/usr/bin/env python3
"""
Long training script for Agar.io RL agent
Trains for approximately 30 minutes and saves the model
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 60)
    print("Starting 30-minute training session")
    print("=" * 60)
    
    # Create agent
    agent = AgarioAgent()
    
    # Calculate timesteps for ~30 minutes
    # Based on observed speed of ~888 FPS
    timesteps = 1_600_000  # ~30 minutes at 888 FPS
    
    print(f"\nTraining for {timesteps:,} timesteps")
    print(f"Estimated time: ~30 minutes\n")
    
    start_time = time.time()
    
    # Train
    try:
        metrics = agent.train(total_timesteps=timesteps)
        
        # Save model
        print("\n" + "=" * 60)
        print("Training completed! Saving model...")
        print("=" * 60)
        agent.save_model()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training Summary:")
        print(f"   Total timesteps: {timesteps:,}")
        print(f"   Training time: {elapsed_time/60:.1f} minutes")
        print(f"   Episodes completed: {len(metrics['episode_rewards'])}")
        
        if metrics['episode_rewards']:
            avg_reward = sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards']))
            print(f"   Average reward (last 10): {avg_reward:.2f}")
        
        print(f"\n✅ Model saved to: models/agario_ppo")
        print("\nYou can now load this model in the game UI!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current model...")
        agent.save_model()
        print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Attempting to save model...")
        try:
            agent.save_model()
            print("✅ Model saved!")
        except:
            print("❌ Could not save model")

if __name__ == '__main__':
    main()
