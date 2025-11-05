import os
import pickle
from collections import defaultdict
import random
import numpy as np
from typing import List, Tuple, Optional
# import heapq # Không cần A*
# from itertools import permutations # Không cần A*
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.robot_env import GridWorldEnv

# === CÁC HÀM TRỢ GIÚP ===

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# --- ĐÃ XÓA HÀM a_star ---
# --- ĐÃ XÓA HÀM plan_path_through_waypoints ---

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

# === HÀM TRỢ GIÚP MỚI ĐỂ LẤY TRẠNG THÁI PHỨC TẠP ===
def get_full_state(env: GridWorldEnv) -> Tuple:
    """Đóng gói logic để lấy trạng thái 's' phức tạp của bạn."""
    state_xy = env.get_state()
    
    # Tính khoảng cách đến waypoint/đích tiếp theo
    unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
    if unvisited_wps:
        dist_to_next = min([manhattan_distance(state_xy, wp) for wp in unvisited_wps])
    else:
        # Nếu đã thăm hết waypoints, mục tiêu là goal
        dist_to_next = manhattan_distance(state_xy, env.goal)

    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    
    # Trạng thái s = (vị trí x, vị trí y, mã bit waypoints đã thăm, khoảng cách đến mục tiêu)
    return (state_xy[0], state_xy[1], visited_code, dist_to_next)

# === THAM SỐ ===
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
# Bạn có thể cần tăng num_episodes lên nhiều hơn (ví dụ: 50000 hoặc 100000)
num_episodes = 10000 
max_steps_per_episode = 100

actions = ['up', 'right', 'down', 'left']

# === CÀI ĐẶT MÔI TRƯỜNG ===
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
waypoints = [(3,2), (6,5)]

env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=max_steps_per_episode)

env.step_penalty = -2.0
env.revisit_penalty = -3.0
env.waypoint_reward = 30.0
env.goal_reward = 100.0
env.goal_before_waypoints_penalty = -10.0

# === KHỞI TẠO Q-TABLE (Bước 1 trong Mã giả) ===
# 1. Initialize Q(s, a) arbitrarily
# Q-table giờ là một defaultdict trống, tất cả giá trị Q mặc định là 0.0
ql_Q = defaultdict(lambda: {a: 0.0 for a in actions})

# --- ĐÃ XÓA KHỐI CODE HƯỚNG DẪN BẰNG A* ---
print("Bắt đầu huấn luyện Q-learning thuần túy (không có A* guidance)...")

# === VÒNG LẶP HUẤN LUYỆN (Theo Mã giả) ===

total_rewards = []

# 2. Repeat (for each episode):
for episode in range(num_episodes):
    
    # 3. Initialize s
    # Kỹ thuật "Exploring Starts" của bạn (nâng cao hơn mã giả gốc)
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
    env.state = random.choice(all_cells)
    num_visited = random.randint(0, len(waypoints))
    env.visited_waypoints = set(random.sample(waypoints, num_visited))
    
    # Lấy trạng thái 's' ban đầu
    s = get_full_state(env) 
    
    done = False
    episode_reward = 0
    steps = 0
    
    # 4. Repeat (for each step of episode):
    while not done and steps < max_steps_per_episode:
        
        # 5. Choose a from s using policy derived from Q (e.g., ε-greedy)
        if random.random() < epsilon:
            a = random.choice(actions) # Khám phá (Explore)
        else:
            a = max(ql_Q[s], key=ql_Q[s].get) # Khai thác (Exploit)
        
        # 6. Take action a, observe r, s'
        action_idx = actions.index(a)
        # Môi trường tự cập nhật (next_state_xy, visited_waypoints) bên trong
        _, r, done, _ = env.step(action_idx) 
        
        # Lấy trạng thái mới 's_prime' (s') sau khi hành động
        s_prime = get_full_state(env)
        
        # 7. Q(s, a) ← Q(s, a) + α[r + γ maxₐ' Q(s', a') - Q(s, a)]
        
        # Tìm max Q(s', a')
        # defaultdict sẽ trả về 0.0 cho các hành động của s_prime nếu s_prime chưa tồn tại
        max_q_s_prime = max(ql_Q[s_prime].values()) 
        
        # Công thức cập nhật Q-learning
        td_target = r + gamma * max_q_s_prime  # (r + γ maxₐ' Q(s', a'))
        td_error = td_target - ql_Q[s][a]       # (target - Q(s, a))
        ql_Q[s][a] += alpha * td_error         # Q(s, a) + α * td_error
        
        # 8. s ← s'
        s = s_prime
        
        # Ghi nhận phần thưởng và số bước
        episode_reward += r
        steps += 1
    
    # (Kết thúc vòng lặp "step")
    
    # Cập nhật Epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    total_rewards.append(episode_reward)
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Epsilon: {epsilon:.3f} - Steps: {steps}")

# (Kết thúc vòng lặp "episode")

# === LƯU Q-TABLE ===
print("Huấn luyện hoàn tất.")
models_dir = os.path.join(os.path.dirname(__file__), "models") if '__file__' in globals() else "./models"
os.makedirs(models_dir, exist_ok=True)
QL_QFILE_OFFLINE = os.path.join(models_dir, "qlearning_qtable_offline.pkl")
with open(QL_QFILE_OFFLINE, 'wb') as f:
    pickle.dump(dict(ql_Q), f)

print(f"Q-table được lưu tại: {QL_QFILE_OFFLINE}")