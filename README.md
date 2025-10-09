# 🛣️ Highway Reinforcement Learning Project

Dự án huấn luyện AI lái xe trên đường cao tốc với agent màu xanh lá, hỗ trợ nhiều thuật toán DQN.

## 📋 Mô tả

Hệ thống huấn luyện AI lái xe trên highway với mục tiêu:
- Duy trì tốc độ tối ưu (30-40 km/h)
- Tránh va chạm với các xe khác
- Chuyển làn an toàn
- Di chuyển hiệu quả

## 🧠 Thuật toán hỗ trợ

### 1. **DQN (Deep Q-Network)**
- Thuật toán cơ bản với Experience Replay
- Target Network để ổn định training
- Epsilon-greedy exploration

### 2. **Double DQN (DDQN)**
- Cải thiện DQN bằng cách giảm overestimation bias
- Sử dụng main network để chọn action, target network để đánh giá
- Hiệu suất tốt hơn DQN thông thường

### 3. **Dueling DQN**
- Tách biệt Value function và Advantage function
- Kiến trúc mạng neural đặc biệt
- Học hiệu quả hơn trong môi trường phức tạp

## 🎮 Môi trường Highway

### Trạng thái (State):
- **63 chiều**: Vị trí, tốc độ, làn đường của agent + thông tin 20 xe khác
- **Format**: `[agent_pos, agent_lane, agent_speed, car1_pos, car1_lane, car1_speed, ...]`

### Hành động (Actions):
- `0`: Giữ nguyên
- `1`: Tăng tốc
- `2`: Giảm tốc  
- `3`: Chuyển làn trái
- `4`: Chuyển làn phải

### Reward System:
- `+1`: Tốc độ đúng mục tiêu (30-40 km/h)
- `-5`: Va chạm
- `-0.1`: Đi quá chậm
- `+0.1`: Tiến về phía trước
- `+0.05`: Ở giữa đường (tránh biên)

## 🏗️ Cấu trúc dự án

```
highway/
├── agent.py              # Thuật toán DQN cơ bản
├── agent_ddqn.py         # Thuật toán Double DQN và Dueling DQN
├── highway_environment.py # Môi trường highway với agent màu xanh lá
├── highway_demo.py        # Demo với model đã huấn luyện
├── main.py               # Script huấn luyện chính (hỗ trợ tất cả thuật toán)
├── replay_buffer.py      # Bộ nhớ cho DQN
├── test_highway.py       # Test môi trường highway
├── test_ddqn.py          # So sánh hiệu suất các thuật toán
└── README.md
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies:
```bash
pip install torch numpy gymnasium matplotlib
```

### 2. Test môi trường:
```bash
python test_highway.py
```

### 3. Demo với model đã huấn luyện:
```bash
python highway_demo.py
```

### 4. Huấn luyện và demo:
```bash
python main.py
```
Chọn thuật toán:
- `1`: Huấn luyện DQN
- `2`: Huấn luyện Double DQN  
- `3`: Huấn luyện Dueling DQN
- `4`: Demo DQN
- `5`: Demo Double DQN
- `6`: Demo Dueling DQN
- `7`: So sánh tất cả thuật toán

### 5. So sánh hiệu suất:
```bash
python test_ddqn.py
```
Chọn chế độ:
- `1`: Huấn luyện tất cả thuật toán
- `2`: Test các thuật toán đã có
- `3`: Huấn luyện + Test + So sánh

## 🎨 Visualization

- **Agent màu xanh lá**: Xe được điều khiển bởi AI
- **Các xe khác**: Màu sắc khác nhau, di chuyển tự động
- **4 làn đường**: Môi trường highway thực tế
- **Real-time**: Hiển thị vị trí, tốc độ, làn đường

## 📊 Đặc điểm

### Môi trường Highway:
- **Road length**: 200 mét
- **Number of lanes**: 4
- **Max speed**: 60 km/h
- **Target speed**: 30-40 km/h
- **Number of other cars**: 20

### Hyperparameters:
- **State dimension**: 63
- **Action dimension**: 5
- **Learning rate**: 1e-3
- **Epsilon decay**: 0.995
- **Batch size**: 64
- **Buffer capacity**: 10,000
- **Target update**: Mỗi 10 episodes

## 🎯 Kết quả mong đợi

Agent màu xanh lá sẽ học được:
- Duy trì tốc độ tối ưu (30-40 km/h)
- Tránh va chạm thông minh
- Chuyển làn an toàn
- Di chuyển hiệu quả trên highway

## 📊 So sánh thuật toán

### Double DQN vs DQN:
- **Ưu điểm**: Giảm overestimation bias, hiệu suất ổn định hơn
- **Nhược điểm**: Phức tạp hơn một chút
- **Khi nào dùng**: Khi DQN có vấn đề overestimation

### Dueling DQN vs DQN:
- **Ưu điểm**: Học hiệu quả hơn, đặc biệt khi có nhiều action không quan trọng
- **Nhược điểm**: Kiến trúc phức tạp hơn
- **Khi nào dùng**: Khi môi trường có nhiều state-action pairs tương tự

## 🔧 Tùy chỉnh

### Thay đổi hyperparameters:
Chỉnh sửa trong `main.py` hoặc `agent_ddqn.py`:
```python
agent = DoubleDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-3,              # Learning rate
    epsilon_start=1.0,    # Epsilon ban đầu
    epsilon_end=0.01,     # Epsilon cuối
    epsilon_decay=0.995,  # Tốc độ giảm epsilon
    batch_size=64,        # Kích thước batch
    capacity=10000        # Kích thước replay buffer
)
```

## 📞 Liên hệ

Nếu có thắc mắc, tạo issue hoặc liên hệ team leader.

---

**Happy Coding! 🛣️✨**

Code trên nhánh tạo ra từ nhán dev
Mỗi nhánh là 1 thuật toán
B1 clone code
B2 git check out develop
B3 git branch (Kiểm tra nhánh hiện tại)
B4 git pull origin develop
5 git checkout -b feature/add-login
Tạo nhánh đúng format feature/(tên thuật toán)

phát triển thuật toán trong file agent
và thay đổi main sao cho phù hợp