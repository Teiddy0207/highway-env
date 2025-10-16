import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class HighwayEnvironment(gym.Env):
    """
    Môi trường highway tương thích DDPG
    - Action: 2 chiều liên tục [throttle, lane_change]
        + throttle: [-1, 1] -> giảm tốc / tăng tốc
        + lane_change: [-1, 1] -> lệch làn trái/phải (giữ nguyên nếu gần 0)
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.road_length = 200
        self.num_lanes = 4
        self.lane_width = 3.5
        self.max_speed = 80
        self.target_speed_min = 55
        self.target_speed_max = 60

        self.agent_position = 0
        self.agent_lane = 1
        self.agent_speed = 0
        self.agent_color = 'green'

        self.num_other_cars = 20
        self.other_cars = []

        # ✅ Hành động liên tục: [throttle, lane_change]
        # throttle ∈ [-1, 1], lane_change ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Trạng thái: [agent_pos, agent_lane, agent_speed, other_cars_info]
        obs_dim = 3 + (self.num_other_cars * 3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        self.collision_penalty = -5
        self.slow_penalty = -0.1
        self.speed_reward = 1
        self.progress_reward = 0.1
        
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.agent_patch = None
        self.car_patches = []

    # --- Khởi tạo ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = 0
        self.agent_lane = 1
        self.agent_speed = 35
        self._generate_other_cars()
        if self.render_mode == 'human':
            self._init_visualization()
        return self._get_observation(), {}

    def _generate_other_cars(self):
        self.other_cars = []
        for _ in range(self.num_other_cars):
            while True:
                position = random.uniform(30, self.road_length - 20)
                lane = random.randint(0, self.num_lanes - 1)
                if lane != self.agent_lane or abs(position - self.agent_position) > 10:
                    break
            self.other_cars.append({
                'position': position,
                'lane': lane,
                'speed': random.uniform(25, 50),
                'color': 'black'
            })

    def _get_observation(self):
        obs = [
            self.agent_position / self.road_length,
            self.agent_lane / (self.num_lanes - 1),
            self.agent_speed / self.max_speed
        ]
        for car in self.other_cars:
            obs.extend([
                car['position'] / self.road_length,
                car['lane'] / (self.num_lanes - 1),
                car['speed'] / self.max_speed
            ])
        return np.array(obs, dtype=np.float32)

    # --- Step ---
    def step(self, action):
        old_position = self.agent_position
        old_lane = self.agent_lane
        old_speed = self.agent_speed

        self._execute_action(action)
        self._update_other_cars()
        reward = self._calculate_reward(old_position, old_lane, old_speed)
        terminated = self._check_termination()
        truncated = self.agent_position >= self.road_length

        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), reward, terminated, truncated, {}

    def _execute_action(self, action):
        """Thực thi hành động liên tục"""
        throttle, lane_change = float(action[0]), float(action[1])

        # ✅ throttle: [-1,1] => giảm/tăng tốc
        self.agent_speed = np.clip(self.agent_speed + throttle * 3, 0, self.max_speed)

        # ✅ lane_change: [-1,1] => chuyển làn mượt
        if lane_change < -0.5 and self.agent_lane > 0:
            self.agent_lane -= 1
        elif lane_change > 0.5 and self.agent_lane < self.num_lanes - 1:
            self.agent_lane += 1

        # ✅ Cập nhật vị trí theo vận tốc
        if self.agent_speed < 0.01:
            self.agent_speed = 0.0  # chống trôi
        self.agent_position += self.agent_speed * 0.1 / 3.6

    def _update_other_cars(self):
        for car in self.other_cars:
            car['position'] += car['speed'] * 0.1 / 3.6
            if car['position'] > self.road_length + 30:
                car['position'] = -30
                car['lane'] = random.randint(0, self.num_lanes - 1)
                car['speed'] = random.uniform(25, 50)

    def _calculate_reward(self, old_position, old_lane, old_speed):
        reward = 0
        if self.target_speed_min <= self.agent_speed <= self.target_speed_max:
            reward += self.speed_reward
        if self.agent_speed < self.target_speed_min:
            reward += self.slow_penalty
        if self._check_collision():
            reward += self.collision_penalty
        if self.agent_position > old_position:
            reward += self.progress_reward
        if 1 <= self.agent_lane <= 2:
            reward += 0.05
        return reward

    def _check_collision(self):
        for car in self.other_cars:
            if car['lane'] == self.agent_lane and abs(car['position'] - self.agent_position) < 5:
                return True
        return False

    def _check_termination(self):
        return self._check_collision()

    # --- Visualization giữ nguyên ---
    def _init_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, self.road_length)
        self.ax.set_ylim(-1, self.num_lanes * self.lane_width + 1)
        self.ax.set_aspect('equal')
        self.ax.set_title('Highway Environment - Agent màu xanh lá', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Vị trí (mét)')
        self.ax.set_ylabel('Làn đường')

        for i in range(self.num_lanes + 1):
            y = i * self.lane_width
            self.ax.axhline(y=y, color='white', linewidth=2)
        self.ax.axhline(y=0, color='black', linewidth=3)
        self.ax.axhline(y=self.num_lanes * self.lane_width, color='black', linewidth=3)

        self.agent_patch = patches.Rectangle((0, 0), 4, 2, facecolor=self.agent_color, edgecolor='black', linewidth=2)
        self.ax.add_patch(self.agent_patch)

        self.car_patches = []
        for car in self.other_cars:
            patch = patches.Rectangle((0, 0), 4, 2, facecolor=car['color'], edgecolor='black', linewidth=1)
            self.ax.add_patch(patch)
            self.car_patches.append(patch)

        plt.tight_layout()

    def render(self, mode='human'):
        if mode == 'human' and self.render_mode == 'human':
            agent_y = self.agent_lane * self.lane_width + self.lane_width/2 - 1
            self.agent_patch.set_xy((self.agent_position, agent_y))
            for i, car in enumerate(self.other_cars):
                car_y = car['lane'] * self.lane_width + self.lane_width/2 - 1
                self.car_patches[i].set_xy((car['position'], car_y))
            self.ax.set_title(f'Highway Environment - Agent màu xanh lá | Vị trí (mét)={self.agent_position:.1f}m | Làn đường={self.agent_lane} | Tốc độ={self.agent_speed:.2f} km/h',
                            fontsize=12, fontweight='bold')
            self.ax.set_xlim(max(0, self.agent_position - 50), min(self.road_length, self.agent_position + 50))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.05)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
