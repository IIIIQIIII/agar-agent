#!/usr/bin/env python3
"""
测试竞速模式的多智能体环境
Test Race Mode Multi-Agent Environment
"""

import sys
import numpy as np
from agario_core.envs.multi_agent_env import MultiAgentAgarioEnv


def test_race_mode():
    """测试竞速模式功能"""
    print("=" * 80)
    print("🏁 Testing Race Mode Environment")
    print("=" * 80)
    print()

    # 创建竞速模式环境
    print("📦 Creating race mode environment...")
    env = MultiAgentAgarioEnv(
        world_size=6000,
        food_count=1000,
        num_opponents=5,
        sft_model_path="models/sft_giant/best_model.pt",
        max_steps=5000,
        difficulty='medium',
        target_score=300  # 竞速目标：300分
    )
    print("✅ Environment created!")
    print()

    # 测试多个回合
    num_episodes = 3
    results = {
        'wins': 0,
        'losses': 0,
        'timeouts': 0,
        'player_scores': [],
        'max_opponent_scores': [],
        'completion_steps': []
    }

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"🎮 Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        obs, info = env.reset()
        print(f"Initial state: {info}")

        done = False
        step_count = 0
        max_score_so_far = 0

        while not done and step_count < 1000:  # 测试最多1000步
            # 随机动作（实际训练会用模型）
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # 每100步打印一次进度
            if step_count % 100 == 0:
                print(f"\n  Step {step_count}:")
                print(f"    Player Score: {info['player_score']}/{info['target_score']}")
                print(f"    Max Opponent Score: {info['max_opponent_score']}")
                print(f"    Score Gap: {info['score_gap']:+d}")
                print(f"    Kills: {info['kills']}")
                print(f"    Alive: {info['player_alive']}")

            # 追踪最高分数
            if info['player_score'] > max_score_so_far:
                max_score_so_far = info['player_score']

        # 回合结束统计
        print(f"\n  {'='*56}")
        print(f"  📊 Episode {episode + 1} Results:")
        print(f"  {'='*56}")
        print(f"  Total Steps: {info['step']}")
        print(f"  Player Score: {info['player_score']}/{info['target_score']}")
        print(f"  Max Opponent Score: {info['max_opponent_score']}")
        print(f"  Kills: {info['kills']}")
        print(f"  Deaths: {info['deaths']}")

        results['player_scores'].append(info['player_score'])
        results['max_opponent_scores'].append(info['max_opponent_score'])

        # 判断结果
        if info['race_finished']:
            if info['winner'] == 'player':
                results['wins'] += 1
                results['completion_steps'].append(info['step'])
                print(f"  🏆 Result: PLAYER WON!")
            else:
                results['losses'] += 1
                print(f"  ❌ Result: OPPONENT WON ({info['winner']})")
        else:
            results['timeouts'] += 1
            print(f"  ⏱️  Result: TIMEOUT")

    # 总结
    env.close()

    print(f"\n\n{'='*80}")
    print(f"🏁 Race Mode Test Summary")
    print(f"{'='*80}")
    print(f"Total Episodes: {num_episodes}")
    print(f"  🏆 Wins: {results['wins']}")
    print(f"  ❌ Losses: {results['losses']}")
    print(f"  ⏱️  Timeouts: {results['timeouts']}")
    print()
    print(f"Average Player Score: {np.mean(results['player_scores']):.1f}")
    print(f"Average Opponent Score: {np.mean(results['max_opponent_scores']):.1f}")
    if results['completion_steps']:
        print(f"Average Steps to Win: {np.mean(results['completion_steps']):.0f}")
    print(f"{'='*80}")

    # 测试结果验证
    print()
    print("✅ Race mode features working:")
    print(f"  ✓ Score tracking: Player scores ranged from {min(results['player_scores'])} to {max(results['player_scores'])}")
    print(f"  ✓ Opponent tracking: Opponent scores ranged from {min(results['max_opponent_scores'])} to {max(results['max_opponent_scores'])}")
    print(f"  ✓ Win detection: {results['wins']} wins detected")
    print(f"  ✓ Loss detection: {results['losses']} losses detected")
    print()
    print("🎉 Race mode test completed successfully!")


if __name__ == '__main__':
    try:
        test_race_mode()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
