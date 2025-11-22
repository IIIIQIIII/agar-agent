#!/usr/bin/env python3
"""
Fine-tune existing model with new exploration-focused reward function
Trains for approximately 3 minutes
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 60)
    print("Fine-tuning with exploration-focused rewards")
    print("=" * 60)
    
    # Create agent and load existing model
    agent = AgarioAgent()
    
    print("\nLoading existing model...")
    try:
        agent.load_model()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("⚠️  No existing model found, creating new one...")
        agent.create_model()
    
    # Calculate timesteps for ~3 minutes
    # Based on observed speed of ~974 FPS
    timesteps = 180_000  # ~3 minutes at 974 FPS
    
    print(f"\nFine-tuning for {timesteps:,} timesteps")
    print(f"Estimated time: ~3 minutes\n")
    print("New reward structure:")
    print("  ✅ Reduced death penalty: -100 → -20")
    print("  ✅ Increased food reward: +1 → +2")
    print("  ✅ Increased enemy reward: +10 → +15")
    print("  ✅ Added exploration bonus (center areas)")
    print("  ✅ Added edge-hugging penalty")
    print("  ✅ Added growth bonus\n")
    
    start_time = time.time()
    
    # Train
    try:
        metrics = agent.train(total_timesteps=timesteps)
        
        # Save model
        print("\n" + "=" * 60)
        print("Fine-tuning completed! Saving model...")
        print("=" * 60)
        agent.save_model()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Fine-tuning Summary:")
        print(f"   Total timesteps: {timesteps:,}")
        print(f"   Training time: {elapsed_time/60:.1f} minutes")
        print(f"   Episodes completed: {len(metrics['episode_rewards'])}")
        
        if metrics['episode_rewards']:
            avg_reward = sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards']))
            print(f"   Average reward (last 10): {avg_reward:.2f}")
        
        print(f"\n✅ Model saved to: models/agario_ppo")
        print("\n🎮 The agent should now explore more and avoid corner-hiding!")
        print("Load the model in the game UI to test the improvements.")
        
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
