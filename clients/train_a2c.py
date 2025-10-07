import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from typing import List, Tuple, Optional
import heapq
from itertools import permutations
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

# Hàm A* để tìm đường đi ngắn nhất
def a_star(start, goal, obstacles, width, height):
    open_set = []
    heapq.heappush(open_set, (0 + manhattan_distance(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                heapq.heappush(open_set, (g + 1 + manhattan_distance((nx, ny), goal), g + 1, (nx, ny), path + [(nx, ny)]))
    return []

# Hàm lập kế hoạch đi qua các waypoint
def plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height):
    best_path = None
    min_len = float('inf')
    for order in permutations(waypoints):
        path = []
        curr = start
        valid = True
        for wp in order:
            sub_path = a_star(curr, wp, obstacles, width, height)
            if not sub_path:
                valid = False
                break
            path += sub_path[:-1]
            curr = wp
        if not valid:
            continue
        sub_path = a_star(curr, goal, obstacles, width, height)
        if not sub_path:
            continue
        path += sub_path
        if len(path) < min_len:
            min_len = len(path)
            best_path = path
    return best_path or []

# Tham số huấn luyện
gamma = 0.99  # Discount factor
epsilon = 1.0  # Epsilon khởi đầu
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 5000  # Số episode để học tốt
max_steps_per_episode = 100  # Ép đường ngắn
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
env.step_penalty = -0.1  # Giảm penalty để khuyến khích khám phá
env.wall_penalty = -2.0
env.obstacle_penalty = -5.0
env.revisit_penalty = -1.0
env.waypoint_reward = 20.0
env.goal_reward = 50.0
env.goal_before_waypoints_penalty = -5.0

# Khởi tạo mô hình A2C
in_channels = 5
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels, height, width, n_actions)
a2c_optimizer = optim.Adam(a2c_model.parameters(), lr=learning_rate)

# Thư mục lưu mô hình
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")

# Kiểm tra đường đi tối ưu bằng A*
optimal_path = plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height)
if optimal_path:
    print(f"Đường đi tối ưu từ A*: {len(optimal_path)-1} bước")

# Huấn luyện A2C
total_rewards = []
a2c_model.train()
for episode in range(num_episodes):
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)
    
    # Exploring starts with controlled initialization
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
    if episode < 1000:  # Giai đoạn đầu: bắt đầu gần start
        env.state = start
        env.visited_waypoints = set()
    else:
        env.state = random.choice(all_cells)
        num_visited = random.randint(0, len(waypoints))
        env.visited_waypoints = set(random.sample(waypoints, num_visited))
    
    state = env.build_grid_state().unsqueeze(0)
    done = False
    episode_reward = 0
    steps = 0
    log_probs = []
    values = []
    rewards = []
    
    while not done and steps < max_steps_per_episode:
        # Waypoint scheduling: chọn target chưa thăm gần nhất
        target = select_next_target(env)
        
        policy_logits, value = a2c_model(state)
        action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.choice(range(n_actions))
        else:
            action_idx = torch.argmax(action_probs).item()
        
        log_prob = torch.log(action_probs[action_idx] + 1e-10)
        action_name = actions[action_idx]
        next_state_xy, reward, done, _ = env.step(action_idx)
        
        # Fallback to A* if action leads to penalty or no progress
        current_dist = manhattan_distance(env.get_state(), target)
        next_dist = manhattan_distance(next_state_xy, target)
        if reward <= env.obstacle_penalty or reward == env.wall_penalty or (next_dist >= current_dist and reward == env.step_penalty):
            path_to_target = a_star(env.get_state(), target, env.obstacles, env.width, env.height)
            if len(path_to_target) > 1:
                next_pos = path_to_target[1]
                dx, dy = next_pos[0] - env.get_state()[0], next_pos[1] - env.get_state()[1]
                try:
                    action_idx = env.ACTIONS.index((dx, dy))
                    next_state_xy, reward, done, _ = env.step(action_idx)
                    log_prob = torch.log(action_probs[action_idx] + 1e-10)  # Cập nhật log_prob cho action mới
                except ValueError:
                    action_idx = torch.multinomial(action_probs, 1).item()
                    next_state_xy, reward, done, _ = env.step(action_idx)
                    log_prob = torch.log(action_probs[action_idx] + 1e-10)
        
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
    G = 0
    if not done:  # Bootstrap giá trị trạng thái cuối nếu chưa done
        _, final_value = a2c_model(state)
        G = final_value.item()
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    advantages = returns - values
    
    # Chuẩn hóa advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Cập nhật mô hình
    actor_loss = -torch.stack(log_probs) * advantages.detach()
    actor_loss = actor_loss.mean()
    critic_loss = advantages.pow(2).mean()
    loss = actor_loss + critic_loss
    
    a2c_optimizer.zero_grad()
    loss.backward()
    a2c_optimizer.step()
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    total_rewards.append(episode_reward)
    
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(total_rewards[-500:])
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Avg Reward: {avg_reward:.2f} - Steps: {steps} - Epsilon: {epsilon:.3f}")

# Lưu mô hình
a2c_model.eval()
torch.save(a2c_model.state_dict(), a2c_model_file)
print(f"Huấn luyện hoàn tất. Mô hình A2C được lưu tại: {a2c_model_file}")