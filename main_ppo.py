# main_ppo.py

import torch
from highway_environment import HighwayEnvironment
from ppo_agent import PPOAgent, Memory

def train_ppo():
    """Huấn luyện highway agent bằng PPO"""
    
    # --- Hyperparameters ---
    render = True
    solved_reward = 100         # Reward trung bình để coi là "đã giải quyết"
    log_interval = 20           # In log mỗi 20 episodes
    max_episodes = 1000         # Số episodes tối đa
    max_timesteps = 200         # Số bước tối đa mỗi episode
    
    update_timestep = 400       # Cập nhật policy mỗi N bước
    K_epochs = 40               # Cập nhật policy K lần với cùng 1 batch dữ liệu
    eps_clip = 0.2              # Epsilon cho PPO clip
    gamma = 0.99                # Discount factor
    lr = 0.001                  # Learning rate
    # -----------------------

    env = HighwayEnvironment(render_mode='human' if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    memory = Memory()
    ppo = PPOAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    
    print("🛣️  Bắt đầu huấn luyện highway agent bằng PPO...")
    print(f" State dim: {state_dim}, Action dim: {action_dim}")
    print("=" * 60)
    
    running_reward = 0
    time_step = 0

    # Vòng lặp huấn luyện
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Chọn hành động từ policy cũ
            action = ppo.select_action(state, memory)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Lưu reward và is_terminals
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            ep_reward += reward

            # Cập nhật nếu đủ số bước
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            
            if done:
                break
        
        running_reward += ep_reward
        
        # In log
        if i_episode % log_interval == 0:
            avg_reward = running_reward / log_interval
            print(f'Episode {i_episode} \t Avg reward: {avg_reward:.2f}')
            running_reward = 0

            # Lưu model nếu đạt kết quả tốt
            if avg_reward > solved_reward:
                print("\n🎉 Đã giải quyết! Lưu model...")
                torch.save(ppo.policy.state_dict(), 'highway_ppo_model.pth')
                break

    env.close()

if __name__ == '__main__':
    train_ppo()