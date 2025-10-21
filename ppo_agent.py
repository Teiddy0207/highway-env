# ppo_agent.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Lưu trữ các bước đi trong một trajectory
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# Mạng Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network (Value) trả về đúng 1 giá trị duy nhất là tốt hay xấu
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
       
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):# dùng để huấn luyện agent , tính log xác xuất , giá trị trạng thái, entropy
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# Agent PPO
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss() # sai số bình phương trung bình 

    def select_action(self, state, memory):# hàm chọn hành động trong quá trình thu thập dữ liệu
        state = torch.FloatTensor(state).unsqueeze(0)
        action, action_logprob = self.policy_old.act(state)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.item()

    def update(self, memory):
        # Tính toán Reward-to-go
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward) # công thức chiết khấu 
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32) 
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) #chuẩn hoá dữ liệu để học ổn định hơn

        # Chuyển list thành tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        # Tối ưu policy trong K epochs
        for _ in range(self.K_epochs):
            # Đánh giá state, action cũ
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Tìm tỉ lệ (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Tìm advantage
            advantages = rewards - state_values.detach()
            
            # Tính toán PPO loss
            surr1 = ratios * advantages # Tính toán tỷ lệ gốc (surr1)

            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages # "Kẹp" tỷ lệ và tính toán mục tiêu đã kẹp (surr2)

            
            # Loss cuối cùng
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy # Lấy giá trị nhỏ nhất (bi quan nhất) để làm loss cho actor


            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Sao chép weights mới vào policy cũ
        self.policy_old.load_state_dict(self.policy.state_dict())