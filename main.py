import time
import torch
from agent import SACAgent
from highway_environment import HighwayEnvironment

def train():
    """Huấn luyện highway agent (xe màu xanh lá) với SAC"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SACAgent(state_dim, action_dim, lr=1e-3)

    episodes = 200
    print("🛣️ Bắt đầu huấn luyện highway agent (SAC)...")
    print(f" State dim: {state_dim}, Action dim: {action_dim}")
    print(" Mục tiêu: Duy trì tốc độ 30-40 km/h, tránh va chạm")
    print("=" * 60)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(200):
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

        # In thông tin mỗi 50 episodes
        if ep % 50 == 0 or ep < 10:
            print(f"Episode {ep:3d} | Reward: {total_reward:6.2f} | Steps: {steps:3d}")

    # Lưu model
    torch.save(agent.actor.state_dict(), 'highway_sac_actor.pth')
    torch.save(agent.critic1.state_dict(), 'highway_sac_critic1.pth')
    torch.save(agent.critic2.state_dict(), 'highway_sac_critic2.pth')
    print(f"\n💾 Model đã được lưu: highway_sac_actor.pth, highway_sac_critic1.pth, highway_sac_critic2.pth")
    
    return agent

def demo(agent):
    """Demo highway agent đã được huấn luyện (xe màu xanh lá) với SAC"""
    env = HighwayEnvironment(render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("\n🛣️ Demo highway agent (SAC)...")
    print("Hành động: 0=Giữ nguyên, 1=Tăng tốc, 2=Giảm tốc, 3=Chuyển trái, 4=Chuyển phải")
    print("=" * 60)

    while not done and steps < 200:
        action = agent.act(state, eval_mode=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if steps % 20 == 0 or done:
            env.render()
            print(f"Step {steps:3d} | Action: {action} | Reward: {reward:5.2f} | Total: {total_reward:6.2f}")

        state = next_state

    print(f"\n✅ Demo hoàn thành!")
    print(f"📊 Tổng reward: {total_reward:.2f}")
    print(f"📊 Số bước: {steps}")

def load_and_demo():
    """Load model đã lưu và chạy demo"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SACAgent(state_dim, action_dim)

    try:
        agent.actor.load_state_dict(torch.load('highway_sac_actor.pth'))
        agent.critic1.load_state_dict(torch.load('highway_sac_critic1.pth'))
        agent.critic2.load_state_dict(torch.load('highway_sac_critic2.pth'))
        print("✅ Đã load model đã huấn luyện (SAC)")
    except:
        print("⚠️ Không tìm thấy model, sử dụng weights ngẫu nhiên")

    demo(agent)

if __name__ == "__main__":
    print("🛣️ HIGHWAY SAC TRAINING SYSTEM")
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