#!/usr/bin/env python3
"""
测试SFT Post-Training集成
验证服务器和API是否正常工作
"""

import requests
import numpy as np
import time
import json

SFT_POST_URL = 'http://localhost:5004'

def test_health():
    """测试健康检查"""
    print("=" * 70)
    print("测试 1: 健康检查")
    print("=" * 70)

    try:
        response = requests.get(f"{SFT_POST_URL}/api/health", timeout=5)
        data = response.json()

        print(f"✅ 服务器状态: {data['status']}")
        print(f"✅ 模型已加载: {data['model_loaded']}")
        print(f"✅ 模型路径: {data['model_path']}")
        return True
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_action():
    """测试动作预测"""
    print("\n" + "=" * 70)
    print("测试 2: 动作预测")
    print("=" * 70)

    # 创建一个模拟的观察向量 (49维)
    observation = np.random.rand(49).tolist()

    try:
        response = requests.post(
            f"{SFT_POST_URL}/api/action",
            json={'observation': observation},
            timeout=5
        )

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   错误: {response.json()}")
            return False

        data = response.json()

        print(f"✅ 动作预测成功!")
        print(f"   动作编号: {data['action']}")
        print(f"   移动角度: {data['angle']}°")
        print(f"   是否分裂: {data['split']}")

        # 验证数据格式
        assert 0 <= data['action'] <= 15, "动作编号应在0-15之间"
        assert data['angle'] in [0, 45, 90, 135, 180, 225, 270, 315], "角度应为45的倍数"
        assert isinstance(data['split'], bool), "split应为布尔值"

        print("✅ 数据验证通过!")
        return True

    except Exception as e:
        print(f"❌ 动作预测失败: {e}")
        return False


def test_stats():
    """测试统计信息"""
    print("\n" + "=" * 70)
    print("测试 3: 统计信息")
    print("=" * 70)

    try:
        response = requests.get(f"{SFT_POST_URL}/api/stats", timeout=5)
        data = response.json()

        print(f"✅ 统计信息获取成功!")
        print(f"   总决策数: {data['total_decisions']}")
        print(f"   分裂率: {data['split_rate']*100:.2f}%")
        print(f"   模型类型: {data['model_type']}")

        # 显示前5个最常用的动作
        action_labels = [
            'Move 0° (Right)', 'Move 45° (Down-Right)', 'Move 90° (Down)', 'Move 135° (Down-Left)',
            'Move 180° (Left)', 'Move 225° (Up-Left)', 'Move 270° (Up)', 'Move 315° (Up-Right)',
            'Split 0° (Right)', 'Split 45° (Down-Right)', 'Split 90° (Down)', 'Split 135° (Down-Left)',
            'Split 180° (Left)', 'Split 225° (Up-Left)', 'Split 270° (Up)', 'Split 315° (Up-Right)'
        ]

        percentages = data['action_distribution']['percentages']
        counts = data['action_distribution']['counts']

        # 按百分比排序
        action_data = [(i, action_labels[i], percentages[i], counts[i]) for i in range(16)]
        action_data.sort(key=lambda x: x[2], reverse=True)

        print("\n   前5个最常用动作:")
        for i, (idx, label, pct, count) in enumerate(action_data[:5], 1):
            bar = "█" * int(pct / 2)  # 简单条形图
            print(f"   {i}. {label:25s} {pct:5.1f}% {bar} ({count})")

        return True

    except Exception as e:
        print(f"❌ 统计信息获取失败: {e}")
        return False


def test_multiple_actions():
    """测试多次动作预测"""
    print("\n" + "=" * 70)
    print("测试 4: 连续动作预测 (10次)")
    print("=" * 70)

    actions = []
    angles = []
    splits = 0

    try:
        for i in range(10):
            observation = np.random.rand(49).tolist()
            response = requests.post(
                f"{SFT_POST_URL}/api/action",
                json={'observation': observation},
                timeout=5
            )

            data = response.json()
            actions.append(data['action'])
            angles.append(data['angle'])
            if data['split']:
                splits += 1

            print(f"   预测 {i+1}/10: 动作={data['action']:2d}, 角度={data['angle']:3d}°, 分裂={data['split']}")

        print(f"\n✅ 连续预测成功!")
        print(f"   动作分布: {set(actions)}")
        print(f"   角度分布: {set(angles)}")
        print(f"   分裂次数: {splits}/10")

        return True

    except Exception as e:
        print(f"❌ 连续预测失败: {e}")
        return False


def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 70)
    print("测试 5: 性能基准测试 (100次预测)")
    print("=" * 70)

    observation = np.random.rand(49).tolist()

    try:
        start_time = time.time()

        for _ in range(100):
            response = requests.post(
                f"{SFT_POST_URL}/api/action",
                json={'observation': observation},
                timeout=5
            )
            response.json()

        end_time = time.time()
        duration = end_time - start_time
        fps = 100 / duration

        print(f"✅ 性能测试完成!")
        print(f"   总耗时: {duration:.2f}秒")
        print(f"   平均延迟: {duration/100*1000:.2f}ms")
        print(f"   吞吐量: {fps:.1f} 预测/秒")

        if fps >= 10:
            print(f"   ⚡ 性能优秀! (游戏需要10 FPS)")
        else:
            print(f"   ⚠️  性能可能不足 (游戏需要10 FPS)")

        return True

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🚀 SFT Post-Training 集成测试")
    print("=" * 70)
    print(f"服务器: {SFT_POST_URL}")
    print("=" * 70)

    tests = [
        ("健康检查", test_health),
        ("动作预测", test_action),
        ("统计信息", test_stats),
        ("连续预测", test_multiple_actions),
        ("性能测试", benchmark_performance),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        time.sleep(0.5)

    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")

    print("=" * 70)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 70)

    if passed == total:
        print("\n🎉 所有测试通过! SFT Post-Training 集成正常工作!")
        print("\n下一步:")
        print("  1. 打开 index.html 在浏览器中")
        print("  2. 启用 AI Mode")
        print("  3. 选择 'SFT Post-Training (SFT + RL)'")
        print("  4. 点击 'Play Game' 观看AI玩游戏!")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("请检查服务器日志和配置")

    print("=" * 70)


if __name__ == '__main__':
    main()
