import torch
import numpy as np

# Tạo dữ liệu giả để debug
def debug_tensor_shapes():
    print("=== DEBUG TENSOR SHAPES ===")
    
    # Giả lập dữ liệu từ replay buffer
    batch_size = 4
    state_dim = 5
    action_dim = 3
    
    # Tạo dữ liệu như trong replay_buffer.py
    states = np.random.randn(batch_size, state_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, (batch_size, 1)).astype(np.int64)  # Shape (batch, 1)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, state_dim).astype(np.float32)
    dones = np.random.randint(0, 2, batch_size).astype(np.float32)
    
    print("1. Dữ liệu từ replay buffer:")
    print(f"   states: {states.shape}")
    print(f"   actions: {actions.shape}")
    print(f"   rewards: {rewards.shape}")
    print(f"   next_states: {next_states.shape}")
    print(f"   dones: {dones.shape}")
    
    # Chuyển đổi sang tensor như trong agent.py
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions).squeeze(1)  # (batch, 1) -> (batch,)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)
    
    print("\n2. Sau khi chuyển đổi sang tensor:")
    print(f"   states_tensor: {states_tensor.shape}")
    print(f"   actions_tensor: {actions_tensor.shape}")
    print(f"   rewards_tensor: {rewards_tensor.shape}")
    print(f"   next_states_tensor: {next_states_tensor.shape}")
    print(f"   dones_tensor: {dones_tensor.shape}")
    
    # Tạo Q-network giả
    q_net = torch.nn.Linear(state_dim, action_dim)
    
    # Test Q-network output
    q_output = q_net(states_tensor)
    print(f"\n3. Q-network output: {q_output.shape}")
    
    # Test các cách sử dụng gather
    print("\n4. Test gather operations:")
    
    # Cách 1: actions.unsqueeze(1) - (batch,) -> (batch, 1)
    actions_unsqueezed = actions_tensor.unsqueeze(1)
    print(f"   actions_unsqueezed: {actions_unsqueezed.shape}")
    
    try:
        gathered1 = q_output.gather(1, actions_unsqueezed)
        print(f"   gather(1, actions_unsqueezed): {gathered1.shape} ✅")
        print(f"   squeeze(1): {gathered1.squeeze(1).shape} ✅")
    except Exception as e:
        print(f"   gather(1, actions_unsqueezed): ERROR - {e}")
    
    # Cách 2: Sử dụng actions trực tiếp từ replay buffer
    actions_original = torch.LongTensor(actions)  # (batch, 1)
    print(f"   actions_original: {actions_original.shape}")
    
    try:
        gathered2 = q_output.gather(1, actions_original)
        print(f"   gather(1, actions_original): {gathered2.shape} ✅")
        print(f"   squeeze(1): {gathered2.squeeze(1).shape} ✅")
    except Exception as e:
        print(f"   gather(1, actions_original): ERROR - {e}")

if __name__ == "__main__":
    debug_tensor_shapes()

