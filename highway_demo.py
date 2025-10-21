#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để huấn luyện và sử dụng model cho Highway Environment
"""

import numpy as np
import torch
from ppo_agent import DQNAgent
from highway_environment import HighwayEnvironment

def quick_train():
    """Huấn luyện nhanh model cho highway environment"""
    print("Bat dau huan luyen nhanh cho Highway Environment...")
    
    env = HighwayEnvironment(render_mode=None)  # Không render khi train để nhanh hơn
    state_dim = env.observation_space.shape[0]  # 63
    action_dim = env.action_space.n             # 5

    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=0.9, epsilon_end=0.01)

    episodes = 50  # Huấn luyện nhanh với ít episodes
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("Muc tieu: Duy tri toc do 30-40 km/h, tranh va cham")
    print("=" * 60)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(100):  # Tối đa 100 bước mỗi episode
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Cập nhật target network mỗi 10 episodes
        if ep % 10 == 0:
            agent.update_target()

        # In thông tin mỗi 10 episodes
        if ep % 10 == 0 or ep < 5:
            print(f"Episode {ep:3d} | Reward: {total_reward:6.2f} | Steps: {steps:3d} | Epsilon: {agent.epsilon:.3f}")

    # Lưu model
    torch.save(agent.q_net.state_dict(), 'highway_dqn_model.pth')
    print(f"\nModel da duoc luu: highway_dqn_model.pth")
    
    return agent

def demo_with_trained_model():
    """Demo với model đã được huấn luyện"""
    print("\nDemo Highway Agent voi model da huan luyen...")
    
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    try:
        agent.q_net.load_state_dict(torch.load('highway_dqn_model.pth'))
        print("Da load model da huan luyen!")
    except FileNotFoundError:
        print("Khong tim thay model, se huan luyen nhanh truoc...")
        agent = quick_train()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Hanh dong: 0=Giu nguyen, 1=Tang toc, 2=Giam toc, 3=Chuyen trai, 4=Chuyen phai")
    print("=" * 60)

    while not done and steps < 200:
        action = agent.act(state, use_random=False)  # Greedy policy - sử dụng model đã học
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Hiển thị thông tin
        if steps % 20 == 0 or done:
            print(f"Step {steps:3d} | Action: {action} | Reward: {reward:5.2f} | Total: {total_reward:6.2f}")
            print(f"Agent: Vi tri {env.agent_position:.1f}m, Lan {env.agent_lane}, Toc do {env.agent_speed:.1f} km/h")

        state = next_state

    print(f"\nDemo hoan thanh!")
    print(f"Tong reward: {total_reward:.2f}")
    print(f"So buoc: {steps}")
    env.close()

def compare_random_vs_trained():
    """So sánh hành vi ngẫu nhiên vs model đã huấn luyện"""
    print("\nSo sanh Random vs Trained Model...")
    
    # Test với hành động ngẫu nhiên
    print("\n1. Test voi hanh dong ngau nhien:")
    env = HighwayEnvironment(render_mode='human')
    state, _ = env.reset()
    total_reward_random = 0
    steps = 0
    
    for step in range(50):
        action = np.random.randint(0, 5)  # Random action
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward_random += reward
        steps += 1
        
        if step % 10 == 0:
            print(f"Random Step {step}: Action={action}, Reward={reward:.2f}, Total={total_reward_random:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Test với model đã huấn luyện
    print("\n2. Test voi model da huan luyen:")
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    try:
        agent.q_net.load_state_dict(torch.load('highway_dqn_model.pth'))
        print("Su dung model da huan luyen!")
    except FileNotFoundError:
        print("Khong tim thay model, huan luyen nhanh...")
        agent = quick_train()
    
    state, _ = env.reset()
    total_reward_trained = 0
    steps = 0
    
    for step in range(50):
        action = agent.act(state, use_random=False)  # Sử dụng model đã học
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward_trained += reward
        steps += 1
        
        if step % 10 == 0:
            print(f"Trained Step {step}: Action={action}, Reward={reward:.2f}, Total={total_reward_trained:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"\nKet qua so sanh:")
    print(f"Random actions: {total_reward_random:.2f}")
    print(f"Trained model: {total_reward_trained:.2f}")
    print(f"Model cai thien: {total_reward_trained - total_reward_random:.2f}")

if __name__ == "__main__":
    print("HIGHWAY ENVIRONMENT DEMO")
    print("=" * 50)
    print("1. Demo voi model da huan luyen")
    print("2. So sanh Random vs Trained")
    print("3. Huan luyen nhanh moi")
    
    choice = input("Chon (1/2/3): ").strip()
    
    if choice == "1":
        demo_with_trained_model()
    elif choice == "2":
        compare_random_vs_trained()
    elif choice == "3":
        agent = quick_train()
        demo_with_trained_model()
    else:
        print("Lua chon khong hop le!")
