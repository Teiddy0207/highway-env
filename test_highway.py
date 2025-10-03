#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script cho Highway Environment với agent màu xanh lá
"""

import numpy as np
from highway_environment import HighwayEnvironment

def test_highway_environment():
    """Test cơ bản môi trường highway"""
    print("Testing Highway Environment...")
    
    # Tạo môi trường
    env = HighwayEnvironment(render_mode='human')
    
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of lanes: {env.num_lanes}")
    print(f"Road length: {env.road_length}m")
    print(f"Agent color: {env.agent_color}")
    
    # Test reset
    state, info = env.reset()
    print(f"Reset successful. State shape: {state.shape}")
    
    # Test một vài bước
    print("\nTesting actions...")
    for step in range(10):
        action = np.random.randint(0, 5)  # Random action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, Done={terminated or truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print("\nTest hoan thanh!")
    env.close()

def test_visualization():
    """Test visualization của môi trường"""
    print("\nTesting Visualization...")
    
    env = HighwayEnvironment(render_mode='human')
    state, _ = env.reset()
    
    print("Visualization window se mo...")
    print("Quan sat agent mau xanh la va cac xe khac!")
    
    # Chạy một episode ngắn với visualization
    for step in range(50):
        action = np.random.randint(0, 5)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: Agent o vi tri {env.agent_position:.1f}m, lan {env.agent_lane}, toc do {env.agent_speed:.1f} km/h")
        
        if terminated or truncated:
            print("Episode ket thuc!")
            break
    
    print("Visualization test hoan thanh!")
    env.close()

if __name__ == "__main__":
    print("HIGHWAY ENVIRONMENT TEST")
    print("=" * 50)
    
    # Test cơ bản
    test_highway_environment()
    
    # Test visualization
    test_visualization()
    
    print("\nTat ca test da hoan thanh!")
    print("De chay huan luyen, su dung: python main.py")
