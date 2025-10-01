import gymnasium as gym
import highway_env
import time
import torch
from agent import DQNAgent

def load_and_demo():
    # Tạo agent với cùng config như training
    env = gym.make("highway-v0", render_mode=None)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Load model đã train (nếu có)
    try:
        agent.q_net.load_state_dict(torch.load('highway_dqn_model.pth'))
        print("✅ Loaded trained model")
    except:
        print("⚠️ No saved model found, using random weights")
    
    # Chạy demo
    env = gym.make("highway-v0", render_mode="human")
    
    print("\n🚗 Running demo episodes...")
    print("Nhấn Ctrl+C để dừng demo")
    
    try:
        for demo_ep in range(10):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            print(f"\n--- Demo Episode {demo_ep + 1} ---")
            
            while not done and step < 300:
                action = agent.act(state, use_random=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                env.render()
                time.sleep(0.1)

                state = next_state
                
                if step % 50 == 0:
                    print(f"Step {step}, Reward: {total_reward:.2f}")

            print(f"✅ Demo Episode {demo_ep + 1} finished - Total Reward: {total_reward:.2f}")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    
    env.close()
    print("🎮 Demo window closed")

if __name__ == "__main__":
    load_and_demo()
