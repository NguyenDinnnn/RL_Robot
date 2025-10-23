import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from typing import List, Tuple, Optional
# import heapq, permutations bị loại bỏ vì không còn A*
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv
from clients.model import ActorCritic

# Hàm chọn target waypoint chưa thăm gần nhất
def select_next_target(env):
    unvisited_waypoints = set(env.waypoints) - env.visited_waypoints
    if unvisited_waypoints:
        return min(unvisited_waypoints, key=lambda wp: manhattan_distance(env.get_state(), wp))
    else:
        return env.goal
    
# Hàm tính khoảng cách Manhattan
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# NOTE: Các hàm a_star và plan_path_through_waypoints đã được loại bỏ hoàn toàn.

# Tham số huấn luyện
gamma = 0.99  # Discount factor
epsilon = 1.0  # Epsilon khởi đầu
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 5000  # Emax
max_steps_per_episode = 100  # tmax
learning_rate = 0.001

# Danh sách hành động
actions = ['up', 'right', 'down', 'left']

# Khởi tạo môi trường
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
waypoints = [(3,2), (6,5)]

env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=max_steps_per_episode)
env.step_penalty = -0.1
env.wall_penalty = -2.0
env.obstacle_penalty = -5.0
env.revisit_penalty = -1.0
env.waypoint_reward = 20.0
env.goal_reward = 50.0
env.goal_before_waypoints_penalty = -5.0

# Khởi tạo mô hình A2C
in_channels = 5
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels, height, width, n_actions) # therta
a2c_optimizer = optim.Adam(a2c_model.parameters(), lr=learning_rate) # therta

# Thư mục lưu mô hình
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")

# Kiểm tra đường đi tối ưu bằng A* đã bị loại bỏ

# Huấn luyện A2C
total_rewards = []
a2c_model.train()
for episode in range(num_episodes): # khoi tao E
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)
    
    # Exploring starts with controlled initialization
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
    if episode < 1000:
        env.state = start
        env.visited_waypoints = set()
    else:
        env.state = random.choice(all_cells)
        num_visited = random.randint(0, len(waypoints))
        env.visited_waypoints = set(random.sample(waypoints, num_visited))
    
    state = env.build_grid_state().unsqueeze(0) # st get state st
    done = False
    episode_reward = 0
    steps = 0 # khoi tao t
    log_probs = []
    values = []
    rewards = []
    
    # Vòng lặp bước
    while not done and steps < max_steps_per_episode:
        # Waypoint scheduling: chọn target chưa thăm gần nhất
        target = select_next_target(env)
        
        policy_logits, value = a2c_model(state) # value V(st; θv)
        action_probs = F.softmax(policy_logits, dim=-1).squeeze(0) # pi
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.choice(range(n_actions)) # at
        else:
            action_idx = torch.argmax(action_probs).item()
        
        log_prob = torch.log(action_probs[action_idx] + 1e-10)
        action_name = actions[action_idx]
        # Receive reward rt and new state st+1
        next_state_xy, reward, done, _ = env.step(action_idx) # reward rt 
        
        # Store value
        next_state = env.build_grid_state().unsqueeze(0)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        
        episode_reward += reward
        steps += 1
        state = next_state
    
    # Bỏ qua episode nếu không có hành động
    if not log_probs:
        continue
    
    # Tính returns và advantages
    returns = []
    G = 0 # R
    if not done:  # Bootstrap giá trị trạng thái cuối nếu chưa done
        _, final_value = a2c_model(state)
        G = final_value.item()
    # for i ∈ { t - 1, ..., tstart } 
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    # advantage
    advantages = returns - values
    
    # Chuẩn hóa advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Cập nhật mô hình
    actor_loss = -torch.stack(log_probs) * advantages.detach() # Cập nhật Actor Loss
    actor_loss = actor_loss.mean()
    critic_loss = advantages.pow(2).mean() # Cập nhật Critic Loss
    loss = actor_loss + critic_loss
    
    # Cập nhật tham số
    a2c_optimizer.zero_grad() # reset gradient
    loss.backward()
    a2c_optimizer.step()
    
    # Update E và epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    total_rewards.append(episode_reward)
    
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(total_rewards[-500:])
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Avg Reward: {avg_reward:.2f} - Steps: {steps} - Epsilon: {epsilon:.3f}")

# Lưu mô hình
a2c_model.eval()
torch.save(a2c_model.state_dict(), a2c_model_file)
print(f"Huấn luyện hoàn tất. Mô hình A2C được lưu tại: {a2c_model_file}")
