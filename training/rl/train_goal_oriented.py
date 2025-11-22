#!/usr/bin/env python3
"""
Train with goal-oriented reward function
Focus on Agar.io success: becoming largest, achieving milestones, dominating
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 70)
    print("GOAL-ORIENTED TRAINING - Agar.io Success Metrics")
    print("=" * 70)
    
    # Create agent and load existing model as starting point
    agent = AgarioAgent()
    
    print("\nLoading existing model as starting point...")
    try:
        agent.load_model()
        print("✅ Model loaded! Continuing training with new reward structure.")
    except FileNotFoundError:
        print("⚠️  No existing model found, creating new one...")
        agent.create_model()
    
    # Train for 5 minutes (~300k timesteps at 1000 FPS)
    timesteps = 300_000
    
    print(f"\nTraining for {timesteps:,} timesteps (~5 minutes)")
    print("\n🎯 NEW GOAL-ORIENTED REWARD STRUCTURE:")
    print("=" * 70)
    print("1. MILESTONE ACHIEVEMENTS")
    print("   • 50, 100, 200, 500, 1000, 2000, 5000 points")
    print("   • Exponential rewards for higher milestones")
    print("")
    print("2. DOMINANCE & RELATIVE PERFORMANCE")
    print("   • Reward for being bigger than average enemy")
    print("   • Extra bonus for true dominance (2x+ size)")
    print("")
    print("3. GROWTH TREND TRACKING")
    print("   • Continuous growth rewards")
    print("   • Personal best bonuses")
    print("   • Trend analysis over 50 steps")
    print("")
    print("4. STRATEGIC POSITIONING")
    print("   • Center control when dominant")
    print("   • Anti-camping penalties")
    print("")
    print("5. EFFICIENCY & AGGRESSION")
    print("   • High score-per-step rewards")
    print("   • Enemy hunting bonuses")
    print("")
    print("6. SCALABLE REWARDS")
    print("   • Bigger enemies = bigger rewards")
    print("   • Death penalty scales with dominance")
    print("")
    print("7. VICTORY CONDITIONS")
    print("   • Massive bonus for 500+ score survival")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train
    try:
        metrics = agent.train(total_timesteps=timesteps)
        
        # Save model
        print("\n" + "=" * 70)
        print("Training completed! Saving goal-oriented model...")
        print("=" * 70)
        agent.save_model()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training Summary:")
        print(f"   Total timesteps: {timesteps:,}")
        print(f"   Training time: {elapsed_time/60:.1f} minutes")
        print(f"   Episodes completed: {len(metrics['episode_rewards'])}")
        
        if metrics['episode_rewards']:
            avg_reward = sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards']))
            max_reward = max(metrics['episode_rewards'])
            print(f"   Average reward (last 10): {avg_reward:.2f}")
            print(f"   Best episode reward: {max_reward:.2f}")
        
        print(f"\n✅ Model saved to: models/agario_ppo")
        print("\n🎮 Expected improvements:")
        print("   • More aggressive score hunting")
        print("   • Better milestone achievement")
        print("   • Sustained growth patterns")
        print("   • Strategic positioning")
        print("\nLoad the model in the game UI to test!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current model...")
        agent.save_model()
        print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save model...")
        try:
            agent.save_model()
            print("✅ Model saved!")
        except:
            print("❌ Could not save model")

if __name__ == '__main__':
    main()
