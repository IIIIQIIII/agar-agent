#!/usr/bin/env python3
"""
Fine-tune with danger awareness (fear & confidence)
5-minute fine-tuning to teach fear of large enemies and confidence toward small ones
"""

from agario_core.agents.rl_agent import AgarioAgent
import time

def main():
    print("=" * 70)
    print("DANGER AWARENESS FINE-TUNING")
    print("=" * 70)
    
    print("\n🎯 Teaching AI:")
    print("   😨 FEAR: Avoid large enemies (penalty when close)")
    print("   😎 CONFIDENCE: Hunt small enemies (reward for pursuit)")
    print()
    print("New Reward Components:")
    print("   • Danger penalty: -2 to -5 when near larger enemies")
    print("   • Hunt reward: +1 to +2 when closing in on prey")
    print("   • Distance-scaled: Closer = stronger effect")
    print()
    
    # Load existing model
    agent = AgarioAgent()
    
    print("Loading current model...")
    try:
        agent.load_model()
        print("✅ Model loaded! Starting fine-tuning...")
    except FileNotFoundError:
        print("⚠️  No model found, creating new one...")
        agent.create_model()
    
    # Fine-tune for 5 minutes (~300k timesteps)
    timesteps = 300_000
    
    print(f"\nFine-tuning for {timesteps:,} timesteps (~5 minutes)")
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
        print("\n🎮 Expected behavior improvements:")
        print("   • Flee from dangerous larger enemies")
        print("   • Actively chase smaller targets")
        print("   • Better risk assessment")
        print("   • More survival-oriented gameplay")
        print("\nLoad the model and watch the AI's new fear & confidence!")
        
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
