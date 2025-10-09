#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để so sánh hiệu suất của DQN, Double DQN và Dueling DQN
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from agent import DQNAgent
from agent_ddqn import DoubleDQNAgent
from highway_environment import HighwayEnvironment


def train_algorithm(algorithm, episodes=100, render_mode=None):
    """Huấn luyện một thuật toán cụ thể"""
    print(f"\n🚀 Bắt đầu huấn luyện {algorithm.upper()}...")
    
    env = HighwayEnvironment(render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Tạo agent
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01)
    elif algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01)
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
    
    # Training
    episode_rewards = []
    episode_steps = []
    epsilon_history = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(200):  # Tối đa 200 bước mỗi episode
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
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilon_history.append(agent.get_epsilon())
        
        # In thông tin mỗi 20 episodes
        if ep % 20 == 0 or ep < 5:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Episode {ep:3d} | Avg Reward: {avg_reward:6.2f} | Steps: {steps:3d} | Epsilon: {agent.get_epsilon():.3f}")
    
    training_time = time.time() - start_time
    
    # Lưu model
    model_name = f'highway_{algorithm}_model.pth'
    agent.save_model(model_name)
    
    print(f"✅ Hoàn thành huấn luyện {algorithm.upper()} trong {training_time:.1f}s")
    print(f"📊 Average Reward (10 episodes cuối): {np.mean(episode_rewards[-10:]):.2f}")
    
    env.close()
    
    return {
        'algorithm': algorithm,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'epsilon_history': epsilon_history,
        'training_time': training_time,
        'final_avg_reward': np.mean(episode_rewards[-10:])
    }


def test_algorithm(algorithm, num_episodes=10):
    """Test hiệu suất của một thuật toán đã được huấn luyện"""
    print(f"\n🧪 Testing {algorithm.upper()}...")
    
    env = HighwayEnvironment(render_mode=None)  # Không render để test nhanh
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Tạo agent
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_dim, action_dim)
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
    
    # Load model
    model_name = f'highway_{algorithm}_model.pth'
    try:
        agent.load_model(model_name)
        print(f"✅ Loaded {algorithm.upper()} model")
    except:
        print(f"⚠️ Không tìm thấy {algorithm.upper()} model, sử dụng random weights")
    
    # Test episodes
    test_rewards = []
    test_steps = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action = agent.act(state, use_random=False)  # Greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            state = next_state
        
        test_rewards.append(episode_reward)
        test_steps.append(steps)
    
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    avg_steps = np.mean(test_steps)
    
    print(f"📈 {algorithm.upper()} Test Results:")
    print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    
    env.close()
    
    return {
        'algorithm': algorithm,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_steps': avg_steps,
        'rewards': test_rewards
    }


def plot_training_results(results):
    """Vẽ biểu đồ kết quả training"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    plt.subplot(2, 2, 1)
    for result in results:
        plt.plot(result['episode_rewards'], label=f"{result['algorithm'].upper()}", alpha=0.7)
    plt.title('Episode Rewards During Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving Average Rewards
    plt.subplot(2, 2, 2)
    for result in results:
        rewards = result['episode_rewards']
        moving_avg = np.convolve(rewards, np.ones(20)/20, mode='valid')
        plt.plot(moving_avg, label=f"{result['algorithm'].upper()}", linewidth=2)
    plt.title('Moving Average Rewards (Window=20)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    plt.subplot(2, 2, 3)
    for result in results:
        plt.plot(result['epsilon_history'], label=f"{result['algorithm'].upper()}", alpha=0.7)
    plt.title('Epsilon Decay During Training')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Training Time Comparison
    plt.subplot(2, 2, 4)
    algorithms = [r['algorithm'].upper() for r in results]
    times = [r['training_time'] for r in results]
    bars = plt.bar(algorithms, times, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Thêm giá trị lên các cột
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_test_results(test_results):
    """Vẽ biểu đồ kết quả test"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Average Rewards Comparison
    plt.subplot(1, 2, 1)
    algorithms = [r['algorithm'].upper() for r in test_results]
    avg_rewards = [r['avg_reward'] for r in test_results]
    std_rewards = [r['std_reward'] for r in test_results]
    
    bars = plt.bar(algorithms, avg_rewards, yerr=std_rewards, 
                   color=['skyblue', 'lightcoral', 'lightgreen'], 
                   capsize=5, alpha=0.7)
    plt.title('Average Test Rewards Comparison')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    
    # Thêm giá trị lên các cột
    for bar, avg, std in zip(bars, avg_rewards, std_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5, 
                f'{avg:.2f}±{std:.2f}', ha='center', va='bottom')
    
    # Plot 2: Individual Test Episodes
    plt.subplot(1, 2, 2)
    for result in test_results:
        plt.scatter([result['algorithm']] * len(result['rewards']), 
                   result['rewards'], alpha=0.6, s=50)
    plt.title('Individual Test Episode Rewards')
    plt.ylabel('Episode Reward')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Hàm chính để chạy so sánh"""
    print("🛣️ HIGHWAY DQN vs DOUBLE DQN COMPARISON")
    print("=" * 50)
    
    algorithms = ['dqn', 'ddqn']
    
    # Hỏi người dùng muốn làm gì
    print("\nChọn chế độ:")
    print("1. Huấn luyện DQN và Double DQN")
    print("2. Test các thuật toán đã có")
    print("3. Huấn luyện + Test + So sánh")
    
    choice = input("Chọn (1-3): ").strip()
    
    if choice == "1":
        # Chỉ huấn luyện
        results = []
        for algo in algorithms:
            result = train_algorithm(algo, episodes=100, render_mode=None)
            results.append(result)
        
        plot_training_results(results)
        
    elif choice == "2":
        # Chỉ test
        test_results = []
        for algo in algorithms:
            result = test_algorithm(algo, num_episodes=20)
            test_results.append(result)
        
        plot_test_results(test_results)
        
    elif choice == "3":
        # Huấn luyện + Test + So sánh
        print("\n🚀 Bắt đầu huấn luyện DQN và Double DQN...")
        training_results = []
        for algo in algorithms:
            result = train_algorithm(algo, episodes=100, render_mode=None)
            training_results.append(result)
        
        print("\n🧪 Bắt đầu test DQN và Double DQN...")
        test_results = []
        for algo in algorithms:
            result = test_algorithm(algo, num_episodes=20)
            test_results.append(result)
        
        # Vẽ biểu đồ
        plot_training_results(training_results)
        plot_test_results(test_results)
        
        # In bảng so sánh
        print("\n" + "="*80)
        print("📊 BẢNG SO SÁNH DQN vs DOUBLE DQN")
        print("="*80)
        print(f"{'Algorithm':<12} {'Training Time':<15} {'Final Avg Reward':<18} {'Test Avg Reward':<16} {'Test Std':<10}")
        print("-"*80)
        
        for i, algo in enumerate(algorithms):
            train_result = training_results[i]
            test_result = test_results[i]
            print(f"{algo.upper():<12} {train_result['training_time']:<15.1f} "
                  f"{train_result['final_avg_reward']:<18.2f} {test_result['avg_reward']:<16.2f} "
                  f"{test_result['std_reward']:<10.2f}")
        
        # Tìm thuật toán tốt nhất
        best_test = max(test_results, key=lambda x: x['avg_reward'])
        print(f"\n🏆 Thuật toán tốt nhất: {best_test['algorithm'].upper()}")
        print(f"   Average Test Reward: {best_test['avg_reward']:.2f} ± {best_test['std_reward']:.2f}")
        
    else:
        print("Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()
