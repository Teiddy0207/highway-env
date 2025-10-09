import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleDQNAgent:
    """
    Double DQN Agent - Cải thiện DQN bằng cách giảm overestimation bias
    """
    def __init__(self, state_dim, action_dim, capacity=10000, batch_size=64, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity)

        # Q-networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def act(self, state, use_random=True):
        """Chọn hành động theo epsilon-greedy policy"""
        if use_random and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 63)
        q_values = self.q_net(state).detach().cpu().numpy()[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Thực hiện một bước training với Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch từ replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)              # (batch, 63)
        actions = torch.LongTensor(actions).to(self.device).squeeze(1)  # (batch,)
        rewards = torch.FloatTensor(rewards).to(self.device)            # (batch,)
        next_states = torch.FloatTensor(next_states).to(self.device)    # (batch, 63)
        dones = torch.FloatTensor(dones).to(self.device)                # (batch,)

        # States đã là 1D, không cần flatten
        states_flat = states      # (batch, 63)
        next_states_flat = next_states  # (batch, 63)

        # Q(s,a) - giá trị Q hiện tại
        q_values = self.q_net(states_flat).gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Double DQN: Sử dụng main network để chọn action, target network để đánh giá
        with torch.no_grad():
            # Bước 1: Sử dụng main network để chọn action tốt nhất
            next_actions = self.q_net(next_states_flat).argmax(1)  # (batch,)
            
            # Bước 2: Sử dụng target network để đánh giá action đó
            target_q_values = self.target_q_net(next_states_flat).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (batch,)
            
            # Bước 3: Tính target Q-value
            target_q = rewards + (1 - dones) * self.gamma * target_q_values

        # Tính loss và cập nhật
        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        """Cập nhật target network"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, filepath):
        """Lưu model"""
        torch.save(self.q_net.state_dict(), filepath)
        print(f"💾 Double DQN model đã được lưu: {filepath}")

    def load_model(self, filepath):
        """Load model"""
        try:
            self.q_net.load_state_dict(torch.load(filepath))
            self.target_q_net.load_state_dict(torch.load(filepath))
            print(f"✅ Đã load Double DQN model từ: {filepath}")
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file: {filepath}")
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")

    def get_epsilon(self):
        """Lấy giá trị epsilon hiện tại"""
        return self.epsilon

    def set_epsilon(self, epsilon):
        """Set giá trị epsilon"""
        self.epsilon = max(epsilon, self.epsilon_min)


