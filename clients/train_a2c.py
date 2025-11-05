import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from typing import List, Tuple, Optional
import sys

# --- SỬA LỖI 1: Import Env THẬT, không dùng dummy ---
try:
    from app.robot_env import GridWorldEnv 
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.robot_env import GridWorldEnv 

# --- SỬA LỖI 2: COPY KIẾN TRÚC TỪ TRAIN_PPO.PY VÀO ĐÂY ---
# (Vì các file .pth của bạn được huấn luyện bằng kiến trúc này)
class ActorCritic(nn.Module):
    def __init__(self, in_channels, height, width, n_actions):
        super(ActorCritic, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_actions = n_actions

        # Kiến trúc CNN: 5 -> 32 -> 64 -> 64 (Khớp với train_ppo.py và file .pth)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # conv.0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),          # conv.2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),          # conv.4
            nn.ReLU()
        )
        
        # Kích thước phải là 64 * H * W
        conv_out_size = 64 * height * width
        
        # Tên layer (fc_shared, actor, critic) (Khớp với train_ppo.py và file .pth)
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
# --- KẾT THÚC SỬA LỖI CLASS ---


# --- SỬA LỖI 3: TOÀN BỘ CODE HUẤN LUYỆN GỐC CỦA A2C NẰM TRONG if __name__ == "__main__": ---
# (Phần này không ảnh hưởng đến server, nhưng được giữ lại)
if __name__ == "__main__":
    
    # ---------- Hyperparams (theo mã giả) ----------
    gamma = 0.99              # discount factor
    t_max = 100               # tmax (max steps per rollout)
    E_max = 5000              # Emax (max episodes)
    learning_rate = 1e-3

    # A2C loss weights from pseudocode
    beta_v = 0.5              # Critic loss weight (beta_v in pseudocode's dv)
    beta_entropy = 0.01       # Entropy bonus weight (beta_dH in pseudocode's d_theta)

    # epsilon-greedy (Optional exploration technique)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # actions
    actions = ['up', 'right', 'down', 'left']

    # environment setup
    width, height = 10, 8
    start = (0, 0)
    goal = (9, 7)
    obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
    waypoints = [(3,2), (6,5)]

    # Dùng class GridWorldEnv THẬT đã import
    env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=t_max)
    # env.step_penalty = -0.1 # Thiết lập Env properties (giữ nguyên nếu cần)
    # ... 

    # model & optimizer
    in_channels = 5 # Phải khớp với build_grid_state()
    n_actions = len(env.ACTIONS)
    
    # Dùng class ActorCritic THẬT đã định nghĩa ở trên
    a2c_model = ActorCritic(in_channels, height, width, n_actions)
    a2c_model.train()

    optimizer = optim.Adam(a2c_model.parameters(), lr=learning_rate)

    # save path
    models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
    os.makedirs(models_dir, exist_ok=True)
    a2c_model_file = os.path.join(models_dir, "a2c_model.pth")

    # helper: manhattan and waypoint scheduling (kept from your code)
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def select_next_target(env):
        unvisited_waypoints = set(env.waypoints) - env.visited_waypoints
        if unvisited_waypoints:
            return min(unvisited_waypoints, key=lambda wp: manhattan_distance(env.get_state(), wp))
        else:
            return env.goal

    # ---------- Training loop (Ánh xạ với mã giả Algorithm 1) ----------
    total_rewards = []
    E = 1                        # Initialize episode counter E <- 1
    t_global = 1                 # Initialize step counter t <- 1

    for episode in range(E_max): # repeat until E > Emax
        
        # Reset gradients (Sẽ được zero_grad() lần nữa trước backward)
        # optimizer.zero_grad() 

        t_start = t_global       # t_start = t

        # Initialize state s_t and exploring starts (Mở rộng từ mã giả)
        env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)

        all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
        if episode < 1000:
            env.state = start
            env.visited_waypoints = set()
        else:
            env.state = random.choice(all_cells)
            num_visited = random.randint(0, len(waypoints))
            env.visited_waypoints = set(random.sample(waypoints, num_visited))

        state = env.build_grid_state().unsqueeze(0)  # Get state s_t
        done = False
        episode_reward = 0
        trajectory = []          # Store trajectory: (log_prob, value, reward, entropy)

        # ---------- Inner rollout loop: repeat... until terminal s_t or t - t_start == t_max ----------
        while (not done) and (t_global - t_start < t_max):
            
            target = select_next_target(env) # Waypoint scheduling

            # Perform a_t according to policy pi(a_t|s_t; theta)
            policy_logits, value = a2c_model(state)
            action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
            
            # Action selection (includes epsilon-greedy)
            if random.random() < epsilon:
                action_idx = random.choice(range(n_actions))
            else:
                action_idx = torch.argmax(action_probs).item()

            log_prob = torch.log(action_probs[action_idx] + 1e-10)
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
            
            # Environment Step: Receive reward r_t and new state s_t+1
            next_state_xy, reward, done, _ = env.step(action_idx)
            
            trajectory.append((log_prob, value.squeeze(0), reward, entropy))
            
            episode_reward += reward
            t_global += 1        # t <- t + 1
            state = env.build_grid_state().unsqueeze(0)

        # If no transitions collected, skip update
        if len(trajectory) == 0:
            E += 1
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            total_rewards.append(episode_reward)
            continue

        # ---------- R calculation (Bootstrap if non-terminal) ----------
        R = 0.0
        if not done:
            # R = V(s_t, theta_v) for non-terminal s_t (Bootstrap from last state)
            _, last_value = a2c_model(state)
            R = last_value.item()

        # ---------- Accumulate returns (for i in {t-1, ..., t_start} do R <- r_i + gamma R) ----------
        returns = []
        current_R = R
        rewards = [step[2] for step in trajectory]
        for r in reversed(rewards):
            current_R = r + gamma * current_R
            returns.insert(0, current_R)

        # Prepare data for loss calculation
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack([step[1] for step in trajectory]).squeeze()
        log_probs = torch.stack([step[0] for step in trajectory])
        entropies = torch.stack([step[3] for step in trajectory])

        # Advantage (R - V(s; theta_v))
        advantages = returns - values
        
        # Normalize advantages (Kỹ thuật ổn định)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------- Accumulate losses (Tương đương Accumulate gradients d_theta và d_theta_v) ----------
        
        # Actor Loss: -(log pi * Advantage.detach).sum()
        actor_loss = -(log_probs * advantages.detach()).sum() 

        # Critic Loss: (Advantage.pow(2)).sum() * beta_v
        critic_loss = (advantages.pow(2)).sum() * beta_v

        # Entropy Loss: - beta_entropy * entropies.sum()
        entropy_loss = - beta_entropy * entropies.sum()

        total_loss = actor_loss + critic_loss + entropy_loss

        # ---------- Backprop and Update (Perform update of theta and theta_v) ----------
        optimizer.zero_grad()        # Reset gradients: d_theta <- 0 and d_theta_v <- 0
        total_loss.backward()        # Calculate d_theta and d_theta_v
        
        # optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(a2c_model.parameters(), 0.5) 
        
        optimizer.step()             # Apply updates to theta and theta_v

        # episode bookkeeping
        E += 1                       # E <- E + 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(total_rewards[-500:]) if len(total_rewards) >= 500 else np.mean(total_rewards)
            print(f"Episode {episode + 1}/{E_max} - Reward: {episode_reward:.2f} - Avg(500): {avg_reward:.2f} - Steps: {t_global - t_start} - Epsilon: {epsilon:.3f}")

    # ---------- Save model ----------
    a2c_model.eval()
    torch.save(a2c_model.state_dict(), a2c_model_file)
    print(f"Training complete. Model saved to: {a2c_model_file}")