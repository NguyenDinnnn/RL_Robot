"""
MC-ES (Explicit Pseudocode Implementation)

Phiên bản này tuân thủ CHÍNH XÁC theo từng chữ của mã giả lý thuyết,
bao gồm cả việc duy trì một đối tượng Policy (pi) riêng biệt 
và cập nhật nó một cách tường minh (explicitly) trong vòng lặp.

*** ĐÃ SỬA LỖI HUẤN LUYỆN ***
"""
import os
import sys
import random
import itertools
import pickle
from collections import defaultdict
from typing import List, Tuple, Set
import numpy as np
import time # Thêm time để theo dõi

# ensure app package importable when running from clients/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.robot_env import GridWorldEnv

# ---------------------- Hyperparameters ----------------------
GAMMA = 0.99

# === SỬA LỖI 1: TĂNG EPISODES ===
# MC-ES với sparse rewards yêu cầu RẤT NHIỀU khám phá.
# 10,000 là quá thấp và sẽ học chính sách "đứng yên".
NUM_EPISODES = 2000000 # 2 Triệu episodes
# ================================

MAX_STEPS_PER_EPISODE = 500 # Giới hạn số bước mỗi tập

# === SỬA LỖI 2: TẮT REWARD SHAPING ===
# Potential-based reward shaping là một kỹ thuật TD (Q-learning),
# nó KHÔNG tương thích về mặt lý thuyết với Monte Carlo (vốn tính G).
# Chúng ta phải học từ sparse rewards (phần thưởng thưa thớt) thực tế.
USE_POTENTIAL_SHAPING = False
# ===================================

WP_WEIGHT = 1.5 # Không còn được sử dụng

# === SỬA LỖI 3: ĐIỀU CHỈNH LOGGING ===
# Điều chỉnh cho phù hợp với 2M episodes
EVAL_INTERVAL = 100000 # Đánh giá mỗi 100k
EVAL_EPISODES = 20
PRINT_INTERVAL = 10000 # In mỗi 10k
# =====================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
# Đặt tên file theo yêu cầu của bạn
Q_FILE = os.path.join(MODELS_DIR, "mc_qtable.pkl")
POLICY_FILE = os.path.join(MODELS_DIR, "mc_policy.pkl")
RETURNS_FILE = os.path.join(MODELS_DIR, "mc_returns.pkl")

# ---------------------- Environment ----------------------
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
waypoints = [(3,2), (6,5)]

env = GridWorldEnv(width=width, height=height, start=start, goal=goal,
                 obstacles=obstacles, waypoints=waypoints, max_steps=MAX_STEPS_PER_EPISODE)

# === SỬA LỖI 4: ĐỒNG BỘ PENALTY ===
# Giảm step_penalty để khuyến khích khám phá (giống server.py)
env.step_penalty = -0.5 
# ==================================
env.revisit_penalty = -2.0
env.waypoint_reward = 30.0
env.goal_reward = 200.0
env.goal_before_waypoints_penalty = -25.0

ACTION_NAMES = GridWorldEnv.ACTION_NAMES[:]

# ---------------------- Helpers ----------------------
# (Toàn bộ các hàm helpers như encode_visited, all_subsets, 
#  build_all_full_states, decode_visited_code, phi_potential, 
#  evaluate_policy, render_policy_as_arrows, safe_load

def encode_visited(wp_list: List[Tuple[int,int]], visited_set: Set[Tuple[int,int]]) -> int:
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code
def all_subsets(lst):
    for r in range(len(lst)+1):
        for comb in itertools.combinations(lst, r):
            yield set(comb)
def build_all_full_states(env: GridWorldEnv):
    all_cells = [(x, y) for x in range(env.width) for y in range(env.height) if (x,y) not in env.obstacles]
    full_states = []
    for cell in all_cells:
        for visited in all_subsets(env.waypoints):
            if cell == env.goal and set(env.waypoints).issubset(visited):
                continue
            code = encode_visited(env.waypoints, visited)
            full_states.append((cell[0], cell[1], code))
    return full_states
def decode_visited_code(code: int, wp_list: List[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    out = set()
    for i, wp in enumerate(wp_list):
        if (code >> i) & 1:
            out.add(wp)
    return out
def phi_potential(state, goal_pos, waypoints, wp_weight=1.0):
    # (Hàm này không còn được dùng nếu USE_POTENTIAL_SHAPING = False)
    x, y, visited_code = state
    pos = (x, y)
    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
    visited_count = 0
    for i in range(len(waypoints)):
        if (visited_code >> i) & 1:
            visited_count += 1
    return -float(dist) - wp_weight * float(visited_count)
def evaluate_policy(policy, n_episodes=20): # Đã sửa: Hàm eval giờ nhận policy
    rewards = []
    steps_list = []
    for _ in range(n_episodes):
        env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints, max_steps=MAX_STEPS_PER_EPISODE)
        env.state = start
        env.visited_waypoints = set()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done and ep_steps < MAX_STEPS_PER_EPISODE:
            s = env.get_state()
            s_code = encode_visited(env.waypoints, env.visited_waypoints)
            full = (s[0], s[1], s_code)
            
            # Đã sửa: Dùng policy được truyền vào
            action = policy[full] 
            
            _, r, done, _ = env.step_by_name(action)
            ep_reward += r
            ep_steps += 1
        rewards.append(ep_reward)
        steps_list.append(ep_steps)
    return float(np.mean(rewards)), float(np.mean(steps_list))
def render_policy_as_arrows(policy): # Đã sửa: Hàm render giờ nhận policy
    arrows = {"up":"↑", "right":"→", "down":"↓", "left":"←"}
    out_lines = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (x,y) in env.obstacles:
                row.append("#")
                continue
            key = (x, y, 0)
            row.append(arrows.get(policy[key], "?")) # Dùng policy[key]
        out_lines.append(" ".join(row))
    return "\n".join(out_lines)
def safe_load(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print("Warning load failed:", path, e)
    return None
# ---------------------- Prepare all states and SA pairs ----------------------
print("Building state-action pairs...")
all_full_states = build_all_full_states(env)
if not all_full_states:
    raise RuntimeError("No startable states (check env configuration)")

all_sa_pairs = [(s, a) for s in all_full_states for a in ACTION_NAMES]
print(f"Total states: {len(all_full_states)}, Total S-A pairs: {len(all_sa_pairs)}")

# ==================================================================
# MÃ GIẢ: Initialize: (BẮT ĐẦU PHẦN KHỞI TẠO)
# ==================================================================

# MÃ GIẢ DÒNG 1: pi(s) in A(s) (arbitrarily)
# Khởi tạo policy (pi) một cách tường minh:
# Gán cho mỗi trạng thái 's' một hành động 'a' ngẫu nhiên.
mc_Policy = defaultdict(lambda: random.choice(ACTION_NAMES))
print("Initialized explicit Policy (pi).")

# MÃ GIẢ DÒNG 2: Q(s, a) in R (arbitrarily)
# Khởi tạo Q-table, gán tất cả Q(s, a) = 0.0
mc_Q = defaultdict(lambda: {a: 0.0 for a in ACTION_NAMES})
print("Initialized Q-table.")

# MÃ GIẢ DÒNG 3: Returns(s, a) <- empty list
# Khởi tạo danh sách Returns, gán cho mỗi (s, a) một danh sách rỗng.
mc_Returns = defaultdict(list)
print("Initialized Returns lists.")

# --- Tải file (nếu có) ---
# Ghi đè các giá trị khởi tạo nếu file tồn tại
loaded_Q = safe_load(Q_FILE)
if loaded_Q:
    for s, v in loaded_Q.items(): mc_Q[tuple(s)] = dict(v)
    print("Loaded existing Q-table.")

loaded_Returns = safe_load(RETURNS_FILE)
if loaded_Returns:
    for k, v in loaded_Returns.items(): mc_Returns[tuple(k)] = list(v)
    print("Loaded existing Returns lists.")

loaded_Policy = safe_load(POLICY_FILE)
if loaded_Policy:
    for k, v in loaded_Policy.items(): mc_Policy[tuple(k)] = v
    print("Loaded existing Policy (pi).")

# ==================================================================
# MÃ GIẢ: Loop forever (for each episode):
# ==================================================================
print(f"Starting MC-ES Training for {NUM_EPISODES} episodes...")
print(f"Rewards: Goal={env.goal_reward}, WP={env.waypoint_reward}, Step={env.step_penalty}")
start_time = time.time()

total_rewards = []
best_eval = -1e9

for episode in range(1, NUM_EPISODES + 1):

    # MÃ GIẢ DÒNG 4: Choose S0 in S, A0 in A(S0) randomly...
    # (Exploring Starts)
    s0, a0 = random.choice(all_sa_pairs)
    visited0 = decode_visited_code(s0[2], env.waypoints)
    
    # Reset env về S0
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints, max_steps=MAX_STEPS_PER_EPISODE)
    env.state = (s0[0], s0[1])
    env.visited_waypoints = set(visited0)
    env.steps = 0

    # MÃ GIẢ DÒNG 5: Generate an episode from S0, A0, following pi:
    trajectory = []
    
    # 5a. Thực hiện hành động ép buộc A0
    next_state, reward, done, info = env.step_by_name(a0)
    
    # (Đã xóa logic Reward Shaping)
    shaped_reward = reward
        
    trajectory.append((s0, a0, shaped_reward))
    ep_reward = shaped_reward
    steps = 1

    # 5b. ... following pi (Đi theo policy 'mc_Policy' đã lưu)
    cur_state_xy = env.get_state()
    cur_code = encode_visited(env.waypoints, env.visited_waypoints)
    cur_full = (cur_state_xy[0], cur_state_xy[1], cur_code)

    while not done and steps < MAX_STEPS_PER_EPISODE:
        
        # LẤY HÀNH ĐỘNG TỪ POLICY (pi)
        action = mc_Policy[cur_full]

        # Thực hiện hành động và lưu trajectory
        next_state, reward, done, info = env.step_by_name(action)
        
        # (Đã xóa logic Reward Shaping)
        shaped_reward = reward

        trajectory.append((cur_full, action, shaped_reward))
        ep_reward += shaped_reward
        steps += 1
        cur_state_xy = env.get_state()
        cur_code = encode_visited(env.waypoints, env.visited_waypoints)
        cur_full = (cur_state_xy[0], cur_state_xy[1], cur_code)

    total_rewards.append(ep_reward)

    # ==================================================================
    # MÃ GIẢ: CẬP NHẬT (Vòng lặp backward)
    # ==================================================================

    # MÃ GIẢ DÒNG 6: G <- 0
    G = 0.0
    seen_sa = set()
    
    # MÃ GIẢ DÒNG 7: Loop for each step of episode, t = T-1, ... 0:
    for t in reversed(range(len(trajectory))):
        s, a, r = trajectory[t]
        
        # MÃ GIẢ DÒNG 8: G <- gamma*G + R_{t+1}
        G = r + GAMMA * G
        
        sa = (s, a)
        
        # MÃ GIẢ DÒNG 9: Unless the pair St, At appears in S0...
        # (Logic của First-Visit)
        if sa not in seen_sa:
            seen_sa.add(sa)
            
            # --- BẮT ĐẦU CẬP NHẬT ---
            
            # MÃ GIẢ DÒNG 10: Append G to Returns(St, At)
            # Thêm G (return) vào danh sách của cặp (s, a) này
            mc_Returns[sa].append(G) 
            
            # MÃ GIẢ DÒNG 11: Q(St, At) <- Average(Returns(St, At))
            # Tính trung bình (Average) của TOÀN BỘ danh sách Returns
            # Và cập nhật giá trị Q-value
            mc_Q[s][a] = float(np.mean(mc_Returns[sa]))
            
            # MÃ GIẢ DÒNG 12: pi(St) <- argmax_a Q(St, a)
            # Cập nhật policy (pi) một cách TƯỜNG MINH
            # Tìm hành động 'a' tốt nhất (argmax) trong Q(s)
            best_action = max(mc_Q[s], key=mc_Q[s].get)
            mc_Policy[s] = best_action
            
            # --- KẾT THÚC CẬP NHẬT ---

    # ... (Phần 'Periodic prints' giữ nguyên) ...
    if episode % PRINT_INTERVAL == 0:
        end_time = time.time()
        eps_per_sec = PRINT_INTERVAL / (end_time - start_time)
        avg_recent = float(np.mean(total_rewards[-PRINT_INTERVAL:])) if len(total_rewards) >= PRINT_INTERVAL else float(np.mean(total_rewards))
        print(f"Episode {episode}/{NUM_EPISODES} | recent_avg={avg_recent:.2f} | {eps_per_sec:.1f} eps/sec")
        start_time = time.time() # Reset timer

    # ... (Phần 'Evaluation' và 'Save' giữ nguyên) ...
    if episode % EVAL_INTERVAL == 0:
        # Đã sửa: Truyền mc_Policy vào hàm đánh giá
        mean_eval, mean_steps = evaluate_policy(mc_Policy, n_episodes=EVAL_EPISODES) 
        print(f"== Eval at {episode}: mean_reward={mean_eval:.2f}, mean_steps={mean_steps:.1f} ==")
        
        if mean_eval > best_eval:
            best_eval = mean_eval
            print(f"Saving new best models (Eval: {best_eval:.2f})...")
            # --- Lưu cả 3 file ---
            with open(Q_FILE, 'wb') as f:
                pickle.dump({tuple(k): v for k,v in mc_Q.items()}, f)
            with open(RETURNS_FILE, 'wb') as f:
                pickle.dump({tuple(k): v for k,v in mc_Returns.items()}, f)
            with open(POLICY_FILE, 'wb') as f:
                pickle.dump({tuple(k): v for k,v in mc_Policy.items()}, f)
            print("Saved improved (EXPLICIT) Q/Returns/Policy.")

# --- Lưu file lần cuối ---
print("MC-ES training finished. Saving final models...")
with open(Q_FILE, 'wb') as f:
    pickle.dump({tuple(k): v for k,v in mc_Q.items()}, f)
with open(RETURNS_FILE, 'wb') as f:
    pickle.dump({tuple(k): v for k,v in mc_Returns.items()}, f)
with open(POLICY_FILE, 'wb') as f:
    pickle.dump({tuple(k): v for k,v in mc_Policy.items()}, f)
print("MC-ES training finished (EXPLICIT version). Models saved.")

# Đã sửa: Truyền mc_Policy vào hàm render
print("\nPolicy (greedy) arrows for visited_code=0:")
print(render_policy_as_arrows(mc_Policy))