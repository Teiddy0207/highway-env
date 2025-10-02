# 🚕 Taxi Reinforcement Learning Project

Dự án so sánh các thuật toán Reinforcement Learning cho bài toán lái xe taxi tự động.

## 📋 Mô tả

Hệ thống huấn luyện AI lái xe taxi với mục tiêu:
- Duy trì tốc độ tối ưu (30-40 km/h)
- Tránh va chạm với các xe khác
- Chuyển làn an toàn
- Di chuyển hiệu quả

## 🎮 Môi trường

### Trạng thái (State):
- **18 chiều**: Vị trí, tốc độ, làn đường của taxi + thông tin 5 xe khác
- **Format**: `[taxi_pos, taxi_lane, taxi_speed, car1_pos, car1_lane, car1_speed, ...]`

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

## 🏗️ Cấu trúc dự án

```
highway/
├── agent.py              # Thuật toán RL (mỗi branch khác nhau)
├── taxi_environment.py   # Môi trường taxi (dùng chung)
├── main.py              # Script huấn luyện và demo
├── replay_buffer.py     # Bộ nhớ cho DQN
└── README.md
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies:
```bash
pip install torch numpy gymnasium
```

### 2. Chạy huấn luyện:
```bash
python main.py
```

**Chọn:**
- `1`: Huấn luyện mới
- `2`: Demo với model đã lưu

### 3. Test môi trường:
```bash
python test_taxi.py
```

## 🌳 Git Workflow

### Branch Strategy:
- `main`: Code gốc với DQN
- `dqn-algorithm`: Cải thiện DQN
- `ppo-algorithm`: Implement PPO
- `a3c-algorithm`: Implement A3C

### Cho mỗi thành viên:

1. **Clone repo:**
```bash
git clone <repo-url>
cd highway
```

2. **Tạo branch riêng:**
```bash
git checkout -b your-algorithm-name
```

3. **Implement thuật toán:**
- Sửa `agent.py` → implement thuật toán của bạn
- Có thể sửa `main.py` nếu cần
- **KHÔNG sửa** `taxi_environment.py` (môi trường chung)

4. **Test:**
```bash
python main.py
```

5. **Commit & Push:**
```bash
git add .
git commit -m "Implement [Algorithm] for taxi"
git push origin your-algorithm-name
```

6. **Tạo Pull Request**

## 📊 Đánh giá thuật toán

### Metrics so sánh:
- **Average Reward**: Tổng reward trung bình
- **Success Rate**: Tỷ lệ hoàn thành episode không va chạm
- **Average Steps**: Số bước trung bình mỗi episode
- **Training Time**: Thời gian huấn luyện
- **Convergence**: Tốc độ hội tụ

### Cách đánh giá:
```bash
# Chạy evaluation cho tất cả thuật toán
python evaluation/compare_algorithms.py
```

## 🔧 Cấu hình

### Tham số môi trường:
- **Road length**: 100 đơn vị
- **Number of lanes**: 3
- **Max speed**: 50 km/h
- **Target speed**: 30-40 km/h
- **Number of other cars**: 5

### Tham số huấn luyện:
- **Episodes**: 300
- **Max steps per episode**: 200
- **Learning rate**: 1e-3
- **Epsilon decay**: 0.995
- **Batch size**: 64

## 📈 Kết quả mong đợi

### DQN:
- Học cách duy trì tốc độ tối ưu
- Sử dụng experience replay
- Epsilon-greedy exploration

### PPO:
- Policy gradient method
- Stable learning
- Good sample efficiency

### A3C:
- Asynchronous learning
- Actor-Critic architecture
- Parallel training

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Implement thuật toán
4. Test kỹ lưỡng
5. Tạo Pull Request

## 📝 Lưu ý

- **Môi trường chung**: Tất cả thuật toán dùng cùng `taxi_environment.py`
- **Interface chung**: Cùng format input/output
- **Đánh giá công bằng**: Cùng điều kiện test
- **Documentation**: Ghi rõ thuật toán và kết quả



