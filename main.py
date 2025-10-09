import time
import torch
import numpy as np
from agent import DQNAgent
from agent_ddqn import DoubleDQNAgent
from highway_environment import HighwayEnvironment


def train(algorithm='dqn'):
    """Huấn luyện highway agent với thuật toán được chọn"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]  # 63
    action_dim = env.action_space.n             # 5

    # Chọn thuật toán
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01)
        model_name = 'highway_dqn_model.pth'
        algo_name = "DQN"
    elif algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01)
        model_name = 'highway_ddqn_model.pth'
        algo_name = "Double DQN"
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")

    episodes = 200
    print(f"🛣️ Bắt đầu huấn luyện highway agent với {algo_name}...")
    print(f" State dim: {state_dim}, Action dim: {action_dim}")
    print(" Mục tiêu: Duy trì tốc độ 30-40 km/h, tránh va chạm")
    print("=" * 60)

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

        # In thông tin mỗi 50 episodes
        if ep % 50 == 0 or ep < 10:
            print(f"Episode {ep:3d} | Reward: {total_reward:6.2f} | Steps: {steps:3d} | Epsilon: {agent.epsilon:.3f}")

    # Lưu model
    agent.save_model(model_name)
    
    return agent


def demo(agent):
    """Demo highway agent đã được huấn luyện (xe màu xanh lá)"""
    env = HighwayEnvironment(render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("\n🛣️ Demo highway agent (xe màu xanh lá)...")
    print("Hành động: 0=Giữ nguyên, 1=Tăng tốc, 2=Giảm tốc, 3=Chuyển trái, 4=Chuyển phải")
    print("=" * 60)

    while not done and steps < 200:
        action = agent.act(state, use_random=False)  # Greedy policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Hiển thị thông tin
        if steps % 20 == 0 or done:
            env.render()
            print(f"Step {steps:3d} | Action: {action} | Reward: {reward:5.2f} | Total: {total_reward:6.2f}")

        state = next_state

    print(f"\n✅ Demo hoàn thành!")
    print(f"📊 Tổng reward: {total_reward:.2f}")
    print(f"📊 Số bước: {steps}")


def load_and_demo(algorithm='dqn'):
    """Load model đã lưu và chạy demo"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Chọn thuật toán
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
        model_name = 'highway_dqn_model.pth'
        algo_name = "DQN"
    elif algorithm == 'ddqn':
        agent = DoubleDQNAgent(state_dim, action_dim)
        model_name = 'highway_ddqn_model.pth'
        algo_name = "Double DQN"
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
    
    try:
        agent.load_model(model_name)
        print(f"✅ Đã load {algo_name} model đã huấn luyện")
    except:
        print(f"⚠️ Không tìm thấy {algo_name} model, sử dụng weights ngẫu nhiên")
    
    demo(agent)


def compare_algorithms():
    """So sánh hiệu suất của các thuật toán"""
    print("🔄 Bắt đầu so sánh các thuật toán...")
    
    algorithms = ['dqn', 'ddqn']
    results = {}
    
    for algo in algorithms:
        print(f"\n📊 Đang test {algo.upper()}...")
        
        # Load model nếu có
        env = HighwayEnvironment(render_mode=None)  # Không render để test nhanh
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if algo == 'dqn':
            agent = DQNAgent(state_dim, action_dim)
            model_name = 'highway_dqn_model.pth'
        elif algo == 'ddqn':
            agent = DoubleDQNAgent(state_dim, action_dim)
            model_name = 'highway_ddqn_model.pth'
        
        try:
            agent.load_model(model_name)
            print(f"✅ Loaded {algo.upper()} model")
        except:
            print(f"⚠️ Không tìm thấy {algo.upper()} model, sử dụng random weights")
        
        # Test 10 episodes
        total_rewards = []
        for ep in range(10):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = agent.act(state, use_random=False)  # Greedy policy
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        results[algo] = {'avg': avg_reward, 'std': std_reward, 'rewards': total_rewards}
        
        print(f"📈 {algo.upper()}: Avg Reward = {avg_reward:.2f} ± {std_reward:.2f}")
    
    # In kết quả so sánh
    print("\n" + "="*60)
    print("📊 KẾT QUẢ SO SÁNH CÁC THUẬT TOÁN")
    print("="*60)
    
    for algo, result in results.items():
        print(f"{algo.upper():>10}: {result['avg']:6.2f} ± {result['std']:5.2f}")
    
    # Tìm thuật toán tốt nhất
    best_algo = max(results.keys(), key=lambda x: results[x]['avg'])
    print(f"\n🏆 Thuật toán tốt nhất: {best_algo.upper()}")
    print(f"   Average Reward: {results[best_algo]['avg']:.2f}")
    
    env.close()


if __name__ == "__main__":
    print("🛣️ HIGHWAY DQN TRAINING SYSTEM")
    print("1. Huấn luyện DQN")
    print("2. Huấn luyện Double DQN")
    print("3. Demo DQN")
    print("4. Demo Double DQN")
    print("5. So sánh DQN vs Double DQN")
    
    choice = input("Chọn (1-5): ").strip()
    
    if choice == "1":
        agent = train('dqn')
        demo(agent)
    elif choice == "2":
        agent = train('ddqn')
        demo(agent)
    elif choice == "3":
        load_and_demo('dqn')
    elif choice == "4":
        load_and_demo('ddqn')
    elif choice == "5":
        print("🔄 So sánh DQN vs Double DQN...")
        compare_algorithms()
    else:
        print("Lựa chọn không hợp lệ!")
