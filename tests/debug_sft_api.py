#!/usr/bin/env python3
"""
调试SFT API调用
"""

import requests
import numpy as np
import json

SFT_SERVER_URL = 'http://localhost:5003'

print("=" * 70)
print("🔍 Debugging SFT API")
print("=" * 70)
print()

# 1. 检查健康状态
print("1. Checking health...")
try:
    response = requests.get(f'{SFT_SERVER_URL}/api/health')
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    print()
except Exception as e:
    print(f"   ❌ Error: {e}")
    print()

# 2. 测试动作API（使用正确的观察向量）
print("2. Testing action API with 49D observation...")
try:
    # 生成一个49维的观察向量
    observation = np.random.randn(49).astype(float).tolist()

    print(f"   Observation shape: {len(observation)}")
    print(f"   First 5 values: {observation[:5]}")

    payload = {
        'observation': observation
    }

    print(f"   Sending POST request to {SFT_SERVER_URL}/api/action")
    response = requests.post(
        f'{SFT_SERVER_URL}/api/action',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )

    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    print()

except Exception as e:
    print(f"   ❌ Error: {e}")
    print()

# 3. 测试模型信息
print("3. Getting model info...")
try:
    response = requests.get(f'{SFT_SERVER_URL}/api/model/info')
    if response.status_code == 200:
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   Status: {response.status_code}")
        print(f"   Error: {response.json()}")
    print()
except Exception as e:
    print(f"   ❌ Error: {e}")
    print()

# 4. 测试统计API
print("4. Getting statistics...")
try:
    response = requests.get(f'{SFT_SERVER_URL}/api/stats')
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    print()
except Exception as e:
    print(f"   ❌ Error: {e}")
    print()

print("=" * 70)
print("✅ Debug complete!")
print("=" * 70)
