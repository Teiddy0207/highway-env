import time
import torch
from agent import DQNAgent
from taxi_environment import TaxiEnvironment


def train():
    """Huấn luyện taxi agent"""
    env = TaxiEnvironment()
    state_dim = env.observation_space.shape[0]  # 18
    action_dim = env.action_space.n             # 5

    agent = DQNAgent(state_dim, action_dim, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01)

    episodes = 300  # Giảm số episodes để test nhanh hơn
    print("🚕 Bắt đầu huấn luyện taxi agent...")
    print(f"📊 State dim: {state_dim}, Action dim: {action_dim}")
    print("🎯 Mục tiêu: Duy trì tốc độ 30-40 km/h, tránh va chạm")
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
    torch.save(agent.q_net.state_dict(), 'taxi_dqn_model.pth')
    print(f"\n💾 Model đã được lưu: taxi_dqn_model.pth")
    
    return agent


def demo(agent):
    """Demo taxi agent đã được huấn luyện"""
    env = TaxiEnvironment()
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("\n🚕 Demo taxi agent...")
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


def load_and_demo():
    """Load model đã lưu và chạy demo"""
    env = TaxiEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    try:
        agent.q_net.load_state_dict(torch.load('taxi_dqn_model.pth'))
        print("✅ Đã load model đã huấn luyện")
    except:
        print("⚠️ Không tìm thấy model, sử dụng weights ngẫu nhiên")
    
    demo(agent)


if __name__ == "__main__":
    print("🚕 TAXI DQN TRAINING SYSTEM")
    print("1. Huấn luyện mới")
    print("2. Demo với model đã lưu")
    
    choice = input("Chọn (1/2): ").strip()
    
    if choice == "1":
        agent = train()
        demo(agent)
    elif choice == "2":
        load_and_demo()
    else:
        print("Lựa chọn không hợp lệ!")
