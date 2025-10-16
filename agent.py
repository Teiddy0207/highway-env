import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-3, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(max_size=100000)

    def act(self, state, noise_scale=0.1):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        self.actor.train()
        if noise_scale and noise_scale > 0:
            action = action + np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action, -self.max_action, self.max_action)

    def remember(self, s, a, r, s2, done):
        # ensure numpy arrays and correct shapes
        self.replay.add(np.array(s, dtype=np.float32),
                        np.array(a, dtype=np.float32),
                        float(r),
                        np.array(s2, dtype=np.float32),
                        bool(done))

    def train_step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.astype(float)).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, prefix="ddpg_highway"):
        torch.save(self.actor.state_dict(), f"{prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pth")
        print(f"Saved models: {prefix}_actor.pth, {prefix}_critic.pth")

    def load(self, prefix="ddpg_highway"):
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{prefix}_critic.pth", map_location=self.device))
        print(f"Loaded models: {prefix}_actor.pth, {prefix}_critic.pth")
