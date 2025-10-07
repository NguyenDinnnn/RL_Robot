import os
import pickle
from collections import defaultdict
import random
import numpy as np
from typing import List, Tuple, Optional
import heapq
from itertools import permutations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.robot_env import GridWorldEnv

# Hàm tính khoảng cách Manhattan
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Hàm A* để tìm đường đi ngắn nhất (tái sử dụng từ server.py)
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

# Hàm mã hóa trạng thái visited waypoints
def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

# Tham số huấn luyện
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Epsilon khởi đầu
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10000
max_steps_per_episode = 100  # Giảm để ép robot tìm đường ngắn hơn

# Danh sách hành động
actions = ['up', 'right', 'down', 'left']

# Khởi tạo Q-table
ql_Q = defaultdict(lambda: {a: 0.0 for a in actions})

# Map cố định để huấn luyện và đánh giá
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
waypoints = [(3,2), (6,5)]

env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=max_steps_per_episode)

# Điều chỉnh reward parameters để tối ưu số bước
env.step_penalty = -2.0  # Tăng penalty mỗi bước
env.revisit_penalty = -3.0  # Tăng phạt khi quay lại ô cũ
env.waypoint_reward = 30.0  # Tăng thưởng waypoint
env.goal_reward = 100.0  # Tăng thưởng goal
env.goal_before_waypoints_penalty = -10.0  # Tăng phạt nếu đến goal trước

# Hướng dẫn ban đầu bằng A*
optimal_path = plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height)
if optimal_path:
    print(f"Đường đi tối ưu từ A*: {len(optimal_path)-1} bước")
    for i in range(len(optimal_path)-1):
        curr, next_pos = optimal_path[i], optimal_path[i+1]
        dx, dy = next_pos[0] - curr[0], next_pos[1] - curr[1]
        action_idx = env.ACTIONS.index((dx, dy))
        action_name = actions[action_idx]
        visited_code = encode_visited(waypoints, set(waypoints[:i] if optimal_path[i] in waypoints else waypoints[:i-1]))
        state = (curr[0], curr[1], visited_code)
        ql_Q[state][action_name] = 10.0  # Khởi tạo Q-value cao cho đường đi tối ưu

# Huấn luyện Q-Learning
total_rewards = []
for episode in range(num_episodes):
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)
    
    # Exploring starts
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
    env.state = random.choice(all_cells)
    num_visited = random.randint(0, len(waypoints))
    env.visited_waypoints = set(random.sample(waypoints, num_visited))
    
    state_xy = env.get_state()
    # Thêm khoảng cách Manhattan vào trạng thái
    dist_to_next = min([manhattan_distance(state_xy, wp) for wp in waypoints if wp not in env.visited_waypoints] + 
                       [manhattan_distance(state_xy, goal)] if len(env.visited_waypoints) == len(waypoints) else [float('inf')])
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    full_state = (state_xy[0], state_xy[1], visited_code, dist_to_next)
    
    done = False
    episode_reward = 0
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        if random.random() < epsilon:
            action_name = random.choice(actions)
        else:
            if full_state in ql_Q:
                action_name = max(ql_Q[full_state], key=ql_Q[full_state].get)
            else:
                action_name = random.choice(actions)
        
        action_idx = actions.index(action_name)
        
        next_state_xy, reward, done, _ = env.step(action_idx)
        
        # Cập nhật khoảng cách Manhattan cho trạng thái tiếp theo
        dist_to_next = min([manhattan_distance(next_state_xy, wp) for wp in waypoints if wp not in env.visited_waypoints] + 
                           [manhattan_distance(next_state_xy, goal)] if len(env.visited_waypoints) == len(waypoints) else [float('inf')])
        next_visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        next_full_state = (next_state_xy[0], next_state_xy[1], next_visited_code, dist_to_next)
        
        # Tính max Q cho trạng thái tiếp theo
        max_next_q = max(ql_Q[next_full_state].values()) if next_full_state in ql_Q else 0.0
        
        # Cập nhật Q-value
        ql_Q[full_state][action_name] += alpha * (reward + gamma * max_next_q - ql_Q[full_state][action_name])
        
        full_state = next_full_state
        episode_reward += reward
        steps += 1
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    total_rewards.append(episode_reward)
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Epsilon: {epsilon:.3f} - Steps: {steps}")

# Lưu Q-table
models_dir = os.path.join(os.path.dirname(__file__), "models") if '__file__' in globals() else "./models"
os.makedirs(models_dir, exist_ok=True)
QL_QFILE_OFFLINE = os.path.join(models_dir, "qlearning_qtable_offline.pkl")
with open(QL_QFILE_OFFLINE, 'wb') as f:
    pickle.dump(dict(ql_Q), f)

print(f"Huấn luyện hoàn tất. Q-table được lưu tại: {QL_QFILE_OFFLINE}")