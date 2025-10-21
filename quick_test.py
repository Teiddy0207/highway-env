#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script để kiểm tra hiệu suất model sau training
"""

import numpy as np
from agent import DQNAgent
from agent_ddqn import DoubleDQNAgent
from highway_environment import HighwayEnvironment


def quick_test(algorithm='ddqn', num_episodes=5):
    """Test nhanh hiệu suất model"""
    print(f"🧪 Quick test {algorithm.upper()} với {num_episodes} episodes...")
    
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Tạo agent
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
        model_name = 'highway_dqn_model.pth'
    elif algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_dim, action_dim)
        model_name = 'highway_ddqn_model.pth'
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
    
    # Load model
    try:
        agent.load_model(model_name)
        print(f"✅ Loaded {algorithm.upper()} model")
    except:
        print(f"⚠️ Không tìm thấy model, sử dụng random weights")
    
    # Test episodes
    total_rewards = []
    total_steps = []
    collisions = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        has_collision = False
        
        print(f"\nEpisode {ep + 1}:")
        
        while not done and steps < 200:
            action = agent.act(state, use_random=False)  # Greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated and not has_collision:
                collisions += 1
                has_collision = True
                print(f"  💥 Va chạm ở step {steps}")
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # In thông tin mỗi 20 steps
            if steps % 20 == 0:
                print(f"  Step {steps}: Pos={env.agent_position:.1f}m, Speed={env.agent_speed:.1f}km/h, Reward={reward:.2f}")
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        print(f"  📊 Episode {ep + 1} kết thúc:")
        print(f"     Total Reward: {episode_reward:.2f}")
        print(f"     Steps: {steps}")
        print(f"     Final Position: {env.agent_position:.1f}m")
        print(f"     Final Speed: {env.agent_speed:.1f}km/h")
        print(f"     Collision: {'Có' if has_collision else 'Không'}")
    
    # Thống kê tổng hợp
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = (num_episodes - collisions) / num_episodes * 100
    
    print(f"\n📈 KẾT QUẢ TỔNG HỢP:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Success Rate: {success_rate:.1f}% (không va chạm)")
    print(f"   Collisions: {collisions}/{num_episodes}")
    
    # Đánh giá
    if avg_reward > 50:
        print("   🎉 Model học tốt!")
    elif avg_reward > 20:
        print("   👍 Model học khá")
    elif avg_reward > 0:
        print("   🤔 Model học chậm")
    else:
        print("   😞 Model chưa học được gì")
    
    env.close()
    return avg_reward, success_rate


if __name__ == "__main__":
    print("🧪 QUICK TEST HIGHWAY MODELS")
    print("=" * 40)
    
    # Test Double DQN
    print("\n1. Testing Double DQN...")
    ddqn_reward, ddqn_success = quick_test('ddqn', 3)
    
    # Test DQN
    print("\n2. Testing DQN...")
    dqn_reward, dqn_success = quick_test('dqn', 3)
    
    # So sánh
    print(f"\n🏆 SO SÁNH:")
    print(f"   Double DQN: Reward={ddqn_reward:.2f}, Success={ddqn_success:.1f}%")
    print(f"   DQN:        Reward={dqn_reward:.2f}, Success={dqn_success:.1f}%")
    
    if ddqn_reward > dqn_reward:
        print("   🥇 Double DQN tốt hơn!")
    elif dqn_reward > ddqn_reward:
        print("   🥇 DQN tốt hơn!")
    else:
        print("   🤝 Cả hai tương đương!")

