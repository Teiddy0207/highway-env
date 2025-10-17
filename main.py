import time
import torch
import numpy as np
from agent import DDPGAgent
from highway_environment import HighwayEnvironment


def train():
    """Huấn luyện highway agent bằng DDPG"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action=max_action)

    episodes = 300
    print("🚗 Bắt đầu huấn luyện Highway DDPG agent ...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("=" * 60)

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        env.current_episode = ep  # ✅ Hiển thị số episode trên giao diện
        total_reward = 0

        for step in range(200):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        # Lưu model mỗi 20 episode
        if ep % 20 == 0 and ep > 0:
            agent.save(f"ddpg_highway_ep{ep}")

        # In ra log
        print(f"Episode {ep:3d} | Reward: {total_reward:7.2f}")

    agent.save("ddpg_highway_final")
    print("\n✅ Huấn luyện hoàn tất và model đã được lưu.")
    env.close()
    return agent


def demo(agent=None):
    """Chạy demo với model DDPG đã huấn luyện"""
    env = HighwayEnvironment(render_mode='human')
    state, _ = env.reset()
    total_reward = 0

    print("\n🛣️ Demo highway DDPG agent (continuous control)...")
    print("=" * 60)

    for step in range(400):
        action = agent.act(state, noise_scale=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        state = next_state
        if done:
            break

    env.close()
    print(f"✅ Demo hoàn tất! Tổng reward: {total_reward:.2f}")


def load_and_demo():
    """Load model và chạy demo"""
    env = HighwayEnvironment(render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action=max_action)
    try:
        agent.load("ddpg_highway_final")
        print("✅ Đã load model đã huấn luyện thành công.")
    except FileNotFoundError:
        print("⚠️ Không tìm thấy model đã huấn luyện, hãy huấn luyện trước!")

    demo(agent)


if __name__ == "__main__":
    print("🚦 HIGHWAY DDPG TRAINING SYSTEM")
    print("1. Huấn luyện mới (Train)")
    print("2. Demo với model đã lưu")

    choice = input("Chọn (1/2): ").strip()

    if choice == "1":
        agent = train()
        demo(agent)
    elif choice == "2":
        load_and_demo()
    else:
        print("Lựa chọn không hợp lệ!")
