#!/usr/bin/env python3
"""
Quick training with improved directional observation space
10-minute training to adapt to new directional inputs
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 70)
    print("DIRECTIONAL PERCEPTION TRAINING")
    print("=" * 70)
    
    print("\n🔧 Observation Space Update:")
    print("   OLD: Absolute positions (x, y coordinates)")
    print("   NEW: Relative directions (cos θ, sin θ vectors)")
    print("\n   Benefits:")
    print("   ✅ AI knows 'which direction' to move")
    print("   ✅ More intuitive for navigation")
    print("   ✅ Better obstacle avoidance")
    print()
    
    # Create new agent (must start fresh due to observation space change)
    agent = AgarioAgent()
    agent.create_model()
    
    # Train for 10 minutes (~600k timesteps at 1000 FPS)
    timesteps = 600_000
    
    print(f"Training for {timesteps:,} timesteps (~10 minutes)")
    print("\nWith directional perception, the AI should:")
    print("   • Navigate directly toward food")
    print("   • Avoid dangerous enemies")
    print("   • Chase vulnerable targets")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train
    try:
        metrics = agent.train(total_timesteps=timesteps)
        
        # Save model
        print("\n" + "=" * 70)
        print("Training completed! Saving directional perception model...")
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
        print("\n🎮 The AI should now:")
        print("   • See food and enemies clearly")
        print("   • Navigate efficiently")
        print("   • Make smarter decisions")
        print("\nLoad the model and test it!")
        
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
