#!/usr/bin/env python3
"""
Fine-tune for multi-blob coordination
5-minute fine-tuning to adapt to team-based observations
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 70)
    print("MULTI-BLOB COORDINATION FINE-TUNING")
    print("=" * 70)
    
    print("\n🎯 Key Changes:")
    print("   • Observation now based on team centroid (mass-weighted center)")
    print("   • AI sees total team size and blob count")
    print("   • All blobs move together as coordinated team")
    print()
    print("Benefits:")
    print("   ✅ Split blobs stay coordinated")
    print("   ✅ Team acts as single unit")
    print("   ✅ Better split strategy")
    print()
    
    # Create new agent (observation space changed from 48 to 49)
    agent = AgarioAgent()
    
    print("Creating new model with team-based observations...")
    agent.create_model()
    print("✅ Model created! Starting training...")
    
    # Train for 10 minutes (~600k timesteps)
    timesteps = 600_000
    
    print(f"\nTraining for {timesteps:,} timesteps (~10 minutes)")
    print("Learning team-based coordination from scratch...")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train
    try:
        metrics = agent.train(total_timesteps=timesteps)
        
        # Save model
        print("\n" + "=" * 70)
        print("Fine-tuning completed! Saving model...")
        print("=" * 70)
        agent.save_model()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n✅ Fine-tuning Summary:")
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
        print("   • Split blobs move in coordination")
        print("   • Team positioning optimized")
        print("   • Better merge/split decisions")
        print("\nLoad and watch the coordinated multi-blob gameplay!")
        
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
