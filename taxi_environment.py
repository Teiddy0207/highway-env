import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame

class TaxiEnvironment(gym.Env):
    """
    Môi trường taxi với:
    - Trạng thái: vị trí, tốc độ, làn đường của taxi và các xe xung quanh
    - Hành động: 5 hành động (giữ nguyên, tăng tốc, giảm tốc, chuyển làn trái, chuyển làn phải)
    - Reward: +1 tốc độ đúng (30-40 km/h), -5 va chạm, -0.1 quá chậm
    """
    
    def __init__(self):
        super().__init__()
        
        # Thông số môi trường
        self.road_length = 100  # Chiều dài đường
        self.num_lanes = 3      # Số làn đường
        self.max_speed = 50     # Tốc độ tối đa (km/h)
        self.target_speed_min = 30  # Tốc độ mục tiêu tối thiểu
        self.target_speed_max = 40  # Tốc độ mục tiêu tối đa
        
        # Trạng thái taxi
        self.taxi_position = 0      # Vị trí trên đường
        self.taxi_lane = 1          # Làn đường (0, 1, 2)
        self.taxi_speed = 0         # Tốc độ hiện tại
        
        # Các xe khác
        self.other_cars = []        # Danh sách xe khác
        self.num_other_cars = 5     # Số xe khác
        
        # Action space: 5 hành động
        self.action_space = spaces.Discrete(5)
        # 0: Giữ nguyên
        # 1: Tăng tốc
        # 2: Giảm tốc  
        # 3: Chuyển làn trái
        # 4: Chuyển làn phải
        
        # Observation space: [taxi_pos, taxi_lane, taxi_speed, other_cars_info]
        # other_cars_info: [pos, lane, speed] cho mỗi xe
        obs_dim = 3 + (self.num_other_cars * 3)  # 3 + 5*3 = 18
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_dim,), dtype=np.float32
        )
        
        # Reward parameters
        self.collision_penalty = -5
        self.slow_penalty = -0.1
        self.speed_reward = 1

        # --- Thêm pygame attributes để render ---
        self.screen = None
        self.width = 600
        self.height = 400
        self.lane_height = self.height // self.num_lanes
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset taxi
        self.taxi_position = 0
        self.taxi_lane = 1
        self.taxi_speed = 30  # Tốc độ ban đầu trong khoảng mục tiêu
        
        # Tạo các xe khác
        self._generate_other_cars()
        
        return self._get_observation(), {}
    
    def _generate_other_cars(self):
        """Tạo các xe khác trên đường"""
        self.other_cars = []
        for _ in range(self.num_other_cars):
            # Đảm bảo không có xe nào quá gần taxi ban đầu
            while True:
                position = random.uniform(20, self.road_length - 10)
                lane = random.randint(0, self.num_lanes - 1)
                # Nếu xe ở cùng làn với taxi, phải cách xa ít nhất 15 đơn vị
                if lane != self.taxi_lane or abs(position - self.taxi_position) > 15:
                    break
            
            car = {
                'position': position,
                'lane': lane,
                'speed': random.uniform(20, 45)
            }
            self.other_cars.append(car)
    
    def _get_observation(self):
        """Lấy trạng thái hiện tại"""
        obs = [
            self.taxi_position / self.road_length,  # Chuẩn hóa vị trí
            self.taxi_lane / (self.num_lanes - 1),  # Chuẩn hóa làn đường
            self.taxi_speed / self.max_speed         # Chuẩn hóa tốc độ
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
        old_position = self.taxi_position
        old_lane = self.taxi_lane
        old_speed = self.taxi_speed
        
        # Thực hiện hành động
        self._execute_action(action)
        
        # Cập nhật vị trí các xe khác
        self._update_other_cars()
        
        # Tính reward
        reward = self._calculate_reward(old_position, old_lane, old_speed)
        
        # Kiểm tra điều kiện kết thúc
        terminated = self._check_termination()
        truncated = self.taxi_position >= self.road_length
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _execute_action(self, action):
        """Thực hiện hành động cụ thể"""
        if action == 0:  # Giữ nguyên
            pass
        elif action == 1:  # Tăng tốc
            self.taxi_speed = min(self.taxi_speed + 5, self.max_speed)
        elif action == 2:  # Giảm tốc
            self.taxi_speed = max(self.taxi_speed - 5, 0)
        elif action == 3:  # Chuyển làn trái
            if self.taxi_lane > 0:
                self.taxi_lane -= 1
        elif action == 4:  # Chuyển làn phải
            if self.taxi_lane < self.num_lanes - 1:
                self.taxi_lane += 1
        
        # Cập nhật vị trí dựa trên tốc độ (mỗi step = 0.05 giờ)
        self.taxi_position += self.taxi_speed * 0.05
    
    def _update_other_cars(self):
        """Cập nhật vị trí các xe khác"""
        for car in self.other_cars:
            car['position'] += car['speed'] * 0.05
            # Nếu xe đi quá xa, tạo xe mới
            if car['position'] > self.road_length + 20:
                car['position'] = -20
                car['lane'] = random.randint(0, self.num_lanes - 1)
                car['speed'] = random.uniform(20, 45)
    
    def _calculate_reward(self, old_position, old_lane, old_speed):
        """Tính reward dựa trên hành vi"""
        reward = 0
        
        # 1. Reward cho tốc độ đúng mục tiêu (30-40 km/h)
        if self.target_speed_min <= self.taxi_speed <= self.target_speed_max:
            reward += self.speed_reward
        
        # 2. Penalty cho tốc độ quá chậm
        if self.taxi_speed < self.target_speed_min:
            reward += self.slow_penalty
        
        # 3. Penalty cho va chạm
        if self._check_collision():
            reward += self.collision_penalty
        
        # 4. Bonus nhỏ cho việc tiến về phía trước
        if self.taxi_position > old_position:
            reward += 0.1
        
        return reward
    
    def _check_collision(self):
        """Kiểm tra va chạm với các xe khác"""
        for car in self.other_cars:
            # Kiểm tra cùng làn và khoảng cách gần (tăng ngưỡng va chạm)
            if (car['lane'] == self.taxi_lane and 
                abs(car['position'] - self.taxi_position) < 3):
                return True
        return False
    
    def _check_termination(self):
        """Kiểm tra điều kiện kết thúc episode"""
        return self._check_collision()
    
    # def render(self, mode='human'):
    #     """Hiển thị môi trường (đơn giản)"""
    #     if mode == 'human':
    #         print(f"Taxi: Pos={self.taxi_position:.1f}, Lane={self.taxi_lane}, Speed={self.taxi_speed:.1f} km/h")
    #         for i, car in enumerate(self.other_cars):
    #             print(f"Car {i}: Pos={car['position']:.1f}, Lane={car['lane']}, Speed={car['speed']:.1f} km/h")
    #         print("-" * 50)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Taxi Environment")
            self.clock = pygame.time.Clock()

        # Vẽ nền đường
        self.screen.fill((50, 50, 50))

        # Vẽ các làn
        for lane in range(1, self.num_lanes):
            y = lane * self.lane_height
            pygame.draw.line(self.screen, (255, 255, 255), (0, y), (self.width, y), 2)

        # Tỉ lệ chuyển từ position → pixel
        scale = self.width / self.road_length

        # Vẽ taxi (màu vàng)
        taxi_x = int(self.taxi_position * scale)
        taxi_y = self.taxi_lane * self.lane_height + self.lane_height // 4
        pygame.draw.rect(self.screen, (255, 255, 0), (taxi_x, taxi_y, 30, self.lane_height // 2))

        # Vẽ các xe khác (màu đỏ)
        for car in self.other_cars:
            car_x = int(car['position'] * scale)
            car_y = car['lane'] * self.lane_height + self.lane_height // 4
            pygame.draw.rect(self.screen, (200, 0, 0), (car_x, car_y, 30, self.lane_height // 2))

        # Update màn hình
        pygame.display.flip()
        self.clock.tick(30)

