# 🛣️ Highway Reinforcement Learning Project

Dự án huấn luyện AI lái xe trên đường cao tốc với agent màu xanh lá.

## 📋 Mô tả

Hệ thống huấn luyện AI lái xe trên highway với mục tiêu:
- Duy trì tốc độ tối ưu (30-40 km/h)
- Tránh va chạm với các xe khác
- Chuyển làn an toàn
- Di chuyển hiệu quả

## 🎮 Môi trường Highway

### Trạng thái (State):
- **27 chiều**: Vị trí, tốc độ, làn đường của agent + thông tin 8 xe khác
- **Format**: `[agent_pos, agent_lane, agent_speed, car1_pos, car1_lane, car1_speed, ...]`

### Hành động (Actions):
- `0`: Giữ nguyên
- `1`: Tăng tốc
- `2`: Giảm tốc  
- `3`: Chuyển làn trái
- `4`: Chuyển làn phải

### Reward System:
- `+1`: Tốc độ đúng mục tiêu (30-40 km/h)
- `-10`: Va chạm
- `-0.1`: Đi quá chậm
- `+0.1`: Tiến về phía trước
- `+0.05`: Ở giữa đường (tránh biên)

## 🏗️ Cấu trúc dự án

```
highway/
├── agent.py              # Thuật toán DQN
├── highway_environment.py # Môi trường highway với agent màu xanh lá
├── highway_demo.py        # Demo với model đã huấn luyện
├── main.py               # Script huấn luyện chính
├── replay_buffer.py      # Bộ nhớ cho DQN
├── test_highway.py       # Test môi trường highway
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

### 4. Huấn luyện mới:
```bash
python main.py
```

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
- **Number of other cars**: 8

### Thuật toán DQN:
- **State dimension**: 27
- **Action dimension**: 5
- **Learning rate**: 1e-3
- **Epsilon decay**: 0.995
- **Batch size**: 64

## 🎯 Kết quả mong đợi

Agent màu xanh lá sẽ học được:
- Duy trì tốc độ tối ưu
- Tránh va chạm thông minh
- Chuyển làn an toàn
- Di chuyển hiệu quả trên highway

## 📞 Liên hệ

Nếu có thắc mắc, tạo issue hoặc liên hệ team leader.

---

**Happy Coding! 🛣️✨**