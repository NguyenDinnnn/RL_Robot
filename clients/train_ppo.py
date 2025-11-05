# clients/train_ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import time
from tqdm import tqdm

# Import environment từ thư mục app (lùi 1 cấp rồi vào app)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# --- Kiểm tra thiết bị (CPU/GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# ---------------------------------
# 1. Định nghĩa mạng Actor-Critic
# (Kiến trúc 5 -> 32 -> 64 -> 64 để khớp với server)
# ---------------------------------
class ActorCritic(nn.Module):
    def __init__(self, in_channels, height, width, n_actions):
        super(ActorCritic, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_actions = n_actions

        # Kiến trúc CNN: 5 -> 32 -> 64 -> 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # conv.0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),          # conv.2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),          # conv.4
            nn.ReLU()
        )
        
        conv_out_size = 64 * height * width
        
        # Tên layer (fc_shared, actor, critic)
        self.fc_shared = nn.Linear(conv_out_size, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc_shared(x))
        policy_logits = self.actor(x)
        state_value = self.critic(x)
        return policy_logits, state_value

# ---------------------------------
# 2. Định nghĩa Rollout Buffer
# ---------------------------------
class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, n_actions, device):
        self.buffer_size = buffer_size
        self.device = device
        
        # Khởi tạo các mảng để lưu trữ
        self.states = torch.zeros((buffer_size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, 1), dtype=torch.int64)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.log_probs_old = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)
        
        # Dùng cho GAE (Generalized Advantage Estimation)
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32)
        
        self.ptr = 0 # Con trỏ vị trí hiện tại trong buffer

    def store(self, state, action, reward, value, log_prob, done):
        """Lưu trữ một transition vào buffer"""
        if self.ptr < self.buffer_size:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs_old[self.ptr] = log_prob
            self.dones[self.ptr] = done
            self.ptr += 1

    def compute_advantages_and_returns(self, last_value, gamma, lambda_gae):
        """
        Tính toán GAE và Returns (Rewards-to-go) sau khi thu thập đủ T timesteps.
        Đây là bước "Compute advantage estimates" trong pseudocode PPO.
        """
        last_gae_lam = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t] # 0.0 nếu done, 1.0 nếu chưa
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_value = self.values[t+1]
            
            # Tính delta (TD error)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # Tính GAE
            last_gae_lam = delta + gamma * lambda_gae * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        
        # Tính returns (target cho value function)
        # Return = Advantage + V(s_t)
        self.returns = self.advantages + self.values

    def get_batch(self, batch_size):
        """Tạo ra các minibatches để update trong K epochs"""
        # Trộn ngẫu nhiên (Shuffle)
        indices = np.random.permutation(self.buffer_size)
        
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                self.states[batch_indices].to(self.device),
                self.actions[batch_indices].to(self.device),
                self.log_probs_old[batch_indices].to(self.device),
                self.advantages[batch_indices].to(self.device),
                self.returns[batch_indices].to(self.device)
            )

    def clear(self):
        """Reset buffer sau mỗi iteration"""
        self.ptr = 0

# ---------------------------------
# 3. Định nghĩa Agent PPO
# ---------------------------------
class PPOAgent:
    def __init__(self, env, lr, gamma, lambda_gae, clip_epsilon, k_epochs, batch_size):
        self.env = env
        self.state_shape = (5, env.height, env.width) # 5 kênh
        self.n_actions = len(env.ACTIONS)
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        
        # Khởi tạo mạng Actor-Critic
        self.network = ActorCritic(
            self.state_shape[0], 
            self.state_shape[1], 
            self.state_shape[2], 
            self.n_actions
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, state_tensor):
        """Chọn hành động dựa trên policy hiện tại"""
        with torch.no_grad():
            policy_logits, state_value = self.network(state_tensor)
        
        # Tạo phân phối xác suất
        dist = Categorical(logits=policy_logits)
        
        # Sample hành động
        action = dist.sample()
        
        # Lấy log_prob của hành động đã chọn
        log_prob = dist.log_prob(action)
        
        return action.item(), state_value.squeeze().item(), log_prob.item()

    def update(self, buffer):
        """
        Update policy và value function
        Đây là phần "Update the policy" và "Update the value function" trong pseudocode.
        """
        # Chuẩn hóa advantage (giúp ổn định)
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(device)

        # Loop K epochs
        for _ in range(self.k_epochs):
            # Lấy các minibatches
            for states, actions, log_probs_old, advs, returns in buffer.get_batch(self.batch_size):
                
                # --- Tính toán loss ---
                # Lấy policy_logits và value mới từ mạng
                policy_logits, state_values = self.network(states)
                state_values = state_values.squeeze() # (batch_size,)
                returns = returns.squeeze()           # (batch_size,)
                
                # Tính log_prob mới cho các hành động cũ
                dist = Categorical(logits=policy_logits)
                log_probs_new = dist.log_prob(actions.squeeze())
                
                # 1. Tính Policy Loss (Actor) - L_CLIP
                # r_t(theta) = exp(log_pi(a|s) - log_pi_old(a|s))
                ratios = torch.exp(log_probs_new - log_probs_old.squeeze())
                
                surr1 = ratios * advs.squeeze()
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advs.squeeze()
                
                # Loss là -E[min(surr1, surr2)] (dấu - vì ta minimize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Tính Value Loss (Critic) - L_VF
                # L_VF = (V(s_t) - R_t)^2
                value_loss = F.mse_loss(state_values, returns)
                
                # 3. Tính Entropy Loss (khuyến khích khám phá)
                entropy_loss = -dist.entropy().mean()
                
                # === (ĐÃ SỬA LỖI) ===
                # Tổng loss
                # Dấu TRỪ (-) entropy_loss để TỐI ĐA HÓA entropy (khuyến khích khám phá)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                # --- Cập nhật ---
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm (giúp ổn định)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

# ---------------------------------
# 4. Hàm Main (Huấn luyện)
# ---------------------------------
def main():
    
    # --- Hyperparameters ---
    NUM_ITERATIONS = 500       # Số "iteration" (tương đương episode trong A2C)
    T_TIMESTEPS = 2048         # (T) Số bước thu thập dữ liệu (Rollout)
    K_EPOCHS = 10              # (K) Số epochs update trên 1 batch dữ liệu
    BATCH_SIZE = 64            # Minibatch size
    GAMMA = 0.99               # Discount factor
    LAMBDA_GAE = 0.95          # GAE parameter
    CLIP_EPSILON = 0.2         # (epsilon) PPO clip range
    LR = 3e-4                  # Learning rate
    
    SAVE_INTERVAL = 25         # Lưu model sau mỗi 25 iterations
    
    # --- Setup Môi trường ---
    # Sử dụng cấu hình map giống server.py
    width, height = 10, 8
    start = (0, 0)
    goal = (9, 7)
    waypoints = [(3, 2), (6, 5)]
    obstacles = [(1, 1), (2, 3), (4, 4), (5, 1), (7, 6)]
    
    env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=100)
    
    # Đồng bộ reward penalties với server.py
    env.step_penalty = -0.5
    env.wall_penalty = -2.0
    env.obstacle_penalty = -5.0
    env.revisit_penalty = -1.0
    env.waypoint_reward = 20.0
    env.goal_reward = 50.0
    env.goal_before_waypoints_penalty = -5.0

    # --- Setup Agent và Buffer ---
    agent = PPOAgent(env, LR, GAMMA, LAMBDA_GAE, CLIP_EPSILON, K_EPOCHS, BATCH_SIZE)
    buffer = RolloutBuffer(
        T_TIMESTEPS, 
        (5, height, width), 
        len(env.ACTIONS), 
        device
    )
    
    # --- Thư mục lưu model ---
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    ppo_model_file = os.path.join(models_dir, "ppo_model.pth")
    
    # --- Vòng lặp huấn luyện chính (theo pseudocode) ---
    print(f"Bắt đầu huấn luyện PPO trên {device}...")
    start_time = time.time()
    
    # `iteration` tương ứng với `iteration = 1, 2, ...` trong pseudocode
    for iteration in range(NUM_ITERATIONS):
        
        # Reset môi trường
        # Sử dụng randomize_map để tăng tính tổng quát hóa
        env.randomize_map(n_obstacles=8, n_waypoints=2)
        state_xy = env.reset()
        state_tensor = env.build_grid_state().unsqueeze(0).to(device)
        
        total_reward_iteration = 0
        
        # `t` tương ứng với `for t = 1, ..., T do` (thu thập dữ liệu)
        for t in range(T_TIMESTEPS):
            # Chọn hành động
            action, value, log_prob = agent.select_action(state_tensor)
            
            # Thực thi hành động
            next_state_xy, reward, done, info = env.step(action)
            total_reward_iteration += reward
            
            # Lưu vào buffer
            buffer.store(state_tensor.squeeze(0), action, reward, value, log_prob, done)
            
            # Chuyển sang state tiếp theo
            next_state_tensor = env.build_grid_state().unsqueeze(0).to(device)
            state_tensor = next_state_tensor
            
            if done:
                # Nếu done, reset môi trường nhưng vẫn tiếp tục thu thập
                # cho đến khi đủ T_TIMESTEPS
                env.randomize_map(n_obstacles=8, n_waypoints=2)
                state_xy = env.reset()
                state_tensor = env.build_grid_state().unsqueeze(0).to(device)
        
        # --- Kết thúc thu thập T bước ---
        
        # Tính value của state cuối cùng
        with torch.no_grad():
            _, last_value = agent.network(state_tensor)
            last_value = last_value.squeeze().item()
            
        # Tính GAE và Returns (Rewards-to-go)
        buffer.compute_advantages_and_returns(last_value, GAMMA, LAMBDA_GAE)
        
        # Cập nhật mạng trong K epochs
        agent.update(buffer)
        
        # Xóa buffer để chuẩn bị cho iteration mới
        buffer.clear()
        
        # --- Logging ---
        print(f"Iteration {iteration+1}/{NUM_ITERATIONS} | "
              f"Reward (avg over {T_TIMESTEPS} steps): {total_reward_iteration/T_TIMESTEPS:.2f} | "
              f"Time: {time.time() - start_time:.1f}s")
        
        # --- Lưu model ---
        if (iteration + 1) % SAVE_INTERVAL == 0:
            print(f"Đang lưu model tại iteration {iteration+1}...")
            torch.save(agent.network.state_dict(), ppo_model_file)
            print(f"✅ Đã lưu PPO model vào: {ppo_model_file}")

    print("--- Huấn luyện PPO hoàn tất! ---")

if __name__ == "__main__":
    main()