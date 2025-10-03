import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time

class HighwayEnvironment(gym.Env):
    """
    Môi trường highway với visualization đẹp và agent màu xanh lá
    - Trạng thái: vị trí, tốc độ, làn đường của agent và các xe khác
    - Hành động: 5 hành động (giữ nguyên, tăng tốc, giảm tốc, chuyển làn trái, chuyển làn phải)
    - Reward: +1 tốc độ đúng (30-40 km/h), -5 va chạm, -0.1 quá chậm
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Thông số môi trường
        self.road_length = 200  # Chiều dài đường
        self.num_lanes = 4      # Số làn đường
        self.lane_width = 3.5   # Chiều rộng làn đường (mét)
        self.max_speed = 60     # Tốc độ tối đa (km/h)
        self.target_speed_min = 30  # Tốc độ mục tiêu tối thiểu
        self.target_speed_max = 40  # Tốc độ mục tiêu tối đa

        # Tốc độ cao tốc 80
        # Trạng thái agent (xe màu xanh lá)
        self.agent_position = 0      # Vị trí trên đường
        self.agent_lane = 1           # Làn đường (0, 1, 2, 3)
        self.agent_speed = 0          # Tốc độ hiện tại
        self.agent_color = 'aquamarine'    # Màu xanh lá
        
        # Các xe khác
        self.other_cars = []          # Danh sách xe khác
        self.num_other_cars = 20      # Số xe khác - tăng lên để thực tế hơn
        
        # Action space: 5 hành động
        self.action_space = spaces.Discrete(5)
        # 0: Giữ nguyên
        # 1: Tăng tốc
        # 2: Giảm tốc  
        # 3: Chuyển làn trái
        # 4: Chuyển làn phải
        #Note: chú ý sửa phần này
        # Observation space: [agent_pos, agent_lane, agent_speed, other_cars_info]
        # other_cars_info: [pos, lane, speed] cho mỗi xe
        obs_dim = 3 + (self.num_other_cars * 3)  # 3 + 20*3 = 63
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_dim,), dtype=np.float32
        )
        
        # Reward parameters
        #note: trừ nặng hơn khi va chạm
        self.collision_penalty = -5
        self.slow_penalty = -0.1
        self.speed_reward = 1
        self.progress_reward = 0.1
        
        # Visualization
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.agent_patch = None
        self.car_patches = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent (xe màu xanh lá)
        self.agent_position = 0
        self.agent_lane = 1
        self.agent_speed = 35  # Tốc độ ban đầu trong khoảng mục tiêu
        
        # Tạo các xe khác
        self._generate_other_cars()
        
        # Khởi tạo visualization nếu cần
        if self.render_mode == 'human':
            self._init_visualization()
        
        return self._get_observation(), {}
    
    def _generate_other_cars(self):
        """Tạo các xe khác trên đường"""
        self.other_cars = []
        car_colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'yellow', 
                     'cyan', 'magenta', 'lime', 'navy', 'maroon', 'olive', 'teal', 'silver',
                     'gold', 'coral', 'indigo', 'violet']
        #xa khác đổi sang trắng hoặc đen (emo)
        
        for i in range(self.num_other_cars):
            # Đảm bảo không có xe nào quá gần agent ban đầu
            while True:
                position = random.uniform(30, self.road_length - 20)
                lane = random.randint(0, self.num_lanes - 1)
                # Nếu xe ở cùng làn với agent, phải cách xa ít nhất 20 đơn vị
                if lane != self.agent_lane or abs(position - self.agent_position) > 10:
                    break
            
            car = {
                'position': position,
                'lane': lane,
                'speed': random.uniform(25, 50),
                'color': car_colors[i % len(car_colors)]
            }
            self.other_cars.append(car)
    
    def _get_observation(self):
        """Lấy trạng thái hiện tại"""
        obs = [
            self.agent_position / self.road_length,  # Chuẩn hóa vị trí
            self.agent_lane / (self.num_lanes - 1),  # Chuẩn hóa làn đường
            self.agent_speed / self.max_speed         # Chuẩn hóa tốc độ
        ]
        
        # Thêm thông tin các xe khác
        for car in self.other_cars:
            obs.extend([
                car['position'] / self.road_length,
                car['lane'] / (self.num_lanes - 1),
                car['speed'] / self.max_speed
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Thực hiện hành động và trả về kết quả"""
        # Lưu trạng thái cũ
        old_position = self.agent_position
        old_lane = self.agent_lane
        old_speed = self.agent_speed
        
        # Thực hiện hành động
        self._execute_action(action)
        
        # Cập nhật vị trí các xe khác
        self._update_other_cars()
        
        # Tính reward
        reward = self._calculate_reward(old_position, old_lane, old_speed)
        
        # Kiểm tra điều kiện kết thúc
        terminated = self._check_termination()
        truncated = self.agent_position >= self.road_length
        
        # Render nếu cần
        if self.render_mode == 'human':
            self.render()
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _execute_action(self, action):
        """Thực hiện hành động cụ thể"""
        if action == 0:  # Giữ nguyên
            pass
        elif action == 1:  # Tăng tốc
            self.agent_speed = min(self.agent_speed + 3, self.max_speed)
        elif action == 2:  # Giảm tốc
            self.agent_speed = max(self.agent_speed - 3, 0)
        elif action == 3:  # Chuyển làn trái
            if self.agent_lane > 0:
                self.agent_lane -= 1
        elif action == 4:  # Chuyển làn phải
            if self.agent_lane < self.num_lanes - 1:
                self.agent_lane += 1
        
        # Cập nhật vị trí dựa trên tốc độ (mỗi step = 0.1 giây)
        self.agent_position += self.agent_speed * 0.1 / 3.6  # Chuyển km/h sang m/s
    
    def _update_other_cars(self):
        """Cập nhật vị trí các xe khác"""
        for car in self.other_cars:
            car['position'] += car['speed'] * 0.1 / 3.6  # Chuyển km/h sang m/s
            # Nếu xe đi quá xa, tạo xe mới
            if car['position'] > self.road_length + 30:
                car['position'] = -30
                car['lane'] = random.randint(0, self.num_lanes - 1)
                car['speed'] = random.uniform(25, 50)
    
    def _calculate_reward(self, old_position, old_lane, old_speed):
        """Tính reward dựa trên hành vi"""
        reward = 0
        
        # 1. Reward cho tốc độ đúng mục tiêu (30-40 km/h)
        if self.target_speed_min <= self.agent_speed <= self.target_speed_max:
            reward += self.speed_reward
        
        # 2. Penalty cho tốc độ quá chậm
        if self.agent_speed < self.target_speed_min:
            reward += self.slow_penalty
        
        # 3. Penalty cho va chạm
        if self._check_collision():
            reward += self.collision_penalty
        
        # 4. Bonus cho việc tiến về phía trước
        if self.agent_position > old_position:
            reward += self.progress_reward
        
        # 5. Bonus nhỏ cho việc ở giữa đường (tránh biên)
        if 1 <= self.agent_lane <= 2:
            reward += 0.05
        
        return reward
    
    def _check_collision(self):
        """Kiểm tra va chạm với các xe khác"""
        for car in self.other_cars:
            # Kiểm tra cùng làn và khoảng cách gần
            if (car['lane'] == self.agent_lane and 
                abs(car['position'] - self.agent_position) < 5):
                return True
        return False
    
    def _check_termination(self):
        """Kiểm tra điều kiện kết thúc episode"""
        return self._check_collision()
    
    def _init_visualization(self):
        """Khởi tạo visualization"""
        plt.ion()  # Bật interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, self.road_length)
        self.ax.set_ylim(-1, self.num_lanes * self.lane_width + 1)
        self.ax.set_aspect('equal')
        self.ax.set_title('Highway Environment - Agent màu xanh lá', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Vị trí (mét)')
        self.ax.set_ylabel('Làn đường')
        
        # Vẽ các làn đường
        for i in range(self.num_lanes + 1):
            y = i * self.lane_width
            self.ax.axhline(y=y, color='white', linewidth=2)
        
        # Vẽ đường viền
        self.ax.axhline(y=0, color='black', linewidth=3)
        self.ax.axhline(y=self.num_lanes * self.lane_width, color='black', linewidth=3)
        
        # Khởi tạo patches cho xe
        self.agent_patch = patches.Rectangle((0, 0), 4, 2, 
                                           facecolor=self.agent_color, 
                                           edgecolor='black', linewidth=2)
        self.ax.add_patch(self.agent_patch)
        
        # Khởi tạo patches cho các xe khác
        self.car_patches = []
        for car in self.other_cars:
            patch = patches.Rectangle((0, 0), 4, 2, 
                                    facecolor=car['color'], 
                                    edgecolor='black', linewidth=1)
            self.ax.add_patch(patch)
            self.car_patches.append(patch)
        
        plt.tight_layout()
    
    def render(self, mode='human'):
        """Hiển thị môi trường"""
        if mode == 'human' and self.render_mode == 'human':
            # Cập nhật vị trí agent (xe màu xanh lá)
            agent_y = self.agent_lane * self.lane_width + self.lane_width/2 - 1
            self.agent_patch.set_xy((self.agent_position, agent_y))
            
            # Cập nhật vị trí các xe khác
            for i, car in enumerate(self.other_cars):
                car_y = car['lane'] * self.lane_width + self.lane_width/2 - 1
                self.car_patches[i].set_xy((car['position'], car_y))
            
            # Cập nhật title với thông tin hiện tại
            self.ax.set_title(f'Highway Environment - Agent màu xanh lá | '
                            f'Vị trí: {self.agent_position:.1f}m | '
                            f'Làn: {self.agent_lane} | '
                            f'Tốc độ: {self.agent_speed:.1f} km/h', 
                            fontsize=12, fontweight='bold')
            
            # Cập nhật view để theo dõi agent
            self.ax.set_xlim(max(0, self.agent_position - 50), 
                           min(self.road_length, self.agent_position + 50))
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.05)  # Delay để có thể nhìn thấy chuyển động
        
        elif mode == 'console':
            print(f"Agent (xanh lá): Pos={self.agent_position:.1f}m, Lane={self.agent_lane}, Speed={self.agent_speed:.1f} km/h")
            for i, car in enumerate(self.other_cars):
                print(f"Xe {i+1} ({car['color']}): Pos={car['position']:.1f}m, Lane={car['lane']}, Speed={car['speed']:.1f} km/h")
            print("-" * 80)
    
    def close(self):
        """Đóng visualization"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
