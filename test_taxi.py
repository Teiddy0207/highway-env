#!/usr/bin/env python3
"""
Script test nhanh cho taxi environment
"""

from taxi_environment import TaxiEnvironment

def test_taxi_environment():
    """Test cơ bản taxi environment"""
    print("🧪 Testing Taxi Environment...")
    
    env = TaxiEnvironment()
    state, _ = env.reset()
    
    print(f"📊 State shape: {state.shape}")
    print(f"📊 Action space: {env.action_space.n}")
    print(f"📊 Initial state: {state}")
    
    # Test một vài hành động
    for step in range(5):
        action = 1  # Tăng tốc
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action} (Tăng tốc)")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        
        env.render()
        
        if terminated or truncated:
            print("Episode kết thúc!")
            break
            
        state = next_state
    
    print("\n✅ Test hoàn thành!")

if __name__ == "__main__":
    test_taxi_environment()
