from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
from threading import Lock
import os, pickle, torch, numpy as np, time
import heapq
from itertools import permutations
from collections import defaultdict
import torch.nn.functional as F
import random

from app.robot_env import GridWorldEnv
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API", version="1.0.0")
_env_lock = Lock()
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# Environment
# ---------------------------
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
waypoints = [(3, 2), (6, 5)]
obstacles = [(1, 1), (2, 3), (4, 4), (5, 1), (7, 6)]
env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=100)
env.step_penalty = -0.5  # Sync with train_a2c.py
env.wall_penalty = -2.0
env.obstacle_penalty = -5.0
env.revisit_penalty = -1.0
env.waypoint_reward = 20.0
env.goal_reward = 50.0
env.goal_before_waypoints_penalty = -5.0

# ---------------------------
# Models dir
# ---------------------------
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)

# ---------------------------
# Load MC
# ---------------------------
mc_qfile = os.path.join(models_dir, "mc_qtable.pkl")
if os.path.exists(mc_qfile):
    with open(mc_qfile, "rb") as f:
        loaded_mc_Q = pickle.load(f)
    mc_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    mc_Q.update(loaded_mc_Q)
else:
    mc_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})

# ---------------------------
# Load Q-learning
# ---------------------------
QL_QFILE_OFFLINE = os.path.join(models_dir, "qlearning_qtable_offline.pkl")
ql_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
if os.path.exists(QL_QFILE_OFFLINE):
    with open(QL_QFILE_OFFLINE, "rb") as f:
        loaded_ql_Q = pickle.load(f)
    ql_Q.update(loaded_ql_Q)
    print(f"✅ Đã tải Q-table Q-Learning từ file OFFLINE: {QL_QFILE_OFFLINE}")
else:
    print(f"⚠️ File Q-table Q-Learning {QL_QFILE_OFFLINE} không tồn tại. Hãy huấn luyện trước.")

# ---------------------------
# Load SARSA
# ---------------------------
sarsa_qfile = os.path.join(models_dir, "sarsa_qtable.pkl")
if os.path.exists(sarsa_qfile):
    with open(sarsa_qfile, "rb") as f:
        loaded_sarsa_Q = pickle.load(f)
    sarsa_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    sarsa_Q.update(loaded_sarsa_Q)
    print(f"✅ Đã tải Q-table SARSA, tổng số state đã biết = {len(sarsa_Q)}")
else:
    sarsa_Q = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    print("🆕 Không tìm thấy Q-table SARSA, tạo mới.")

# ---------------------------
# Load A2C
# ---------------------------
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")
in_channels = 5
height, width = env.height, env.width
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels, height, width, n_actions)
a2c_model_loaded = False
if os.path.exists(a2c_model_file):
    try:
        a2c_model.load_state_dict(torch.load(a2c_model_file))
        a2c_model.eval()
        a2c_model_loaded = True
        print("✅ A2C model loaded successfully")
    except RuntimeError as e:
        print(f"⚠️ Không load được A2C checkpoint: {str(e)}. Sẽ dùng model mới.")
else:
    print(f"⚠️ File A2C model {a2c_model_file} không tồn tại. Hãy huấn luyện trước bằng train_a2c.py.")

# ---------------------------
# RL params
# ---------------------------
actions = ['up', 'right', 'down', 'left']
alpha, gamma = 0.1, 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# ---------------------------
# Request Models
# ---------------------------
class ResetRequest(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None
    waypoints: Optional[List[Tuple[int, int]]] = None
    obstacles: Optional[List[Tuple[int, int]]] = None
    max_steps: Optional[int] = None

class ActionInput(BaseModel):
    action: Optional[int] = None
    action_name: Optional[str] = None

class AlgorithmRequest(BaseModel):
    algorithm: str

class AStarRequest(BaseModel):
    goal: Optional[Tuple[int, int]] = None

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def select_next_target(env):
    unvisited_waypoints = set(env.waypoints) - env.visited_waypoints
    if unvisited_waypoints:
        return min(unvisited_waypoints, key=lambda wp: manhattan_distance(env.get_state(), wp))
    else:
        return env.goal

# ---------------------------
# A* functions
# ---------------------------
def a_star(start, goal, obstacles, width, height):
    open_set = []
    heapq.heappush(open_set, (0 + abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start, [start]))
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
                heapq.heappush(open_set, (g + 1 + abs(nx - goal[0]) + abs(ny - goal[1]), g + 1, (nx, ny), path + [(nx, ny)]))
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

# ---------------------------
# Extend GridWorldEnv for A* step
# ---------------------------
def step_to_rl(self, target):
    self.state = target
    self.steps += 1
    reward = -0.1  # Đồng bộ với A* reward trong server
    done = False
    info = {"note": "Auto move by A* (RL reward)"}
    if target in self.waypoints and target not in self.visited_waypoints:
        self.visited_waypoints.add(target)
        reward += self.waypoint_reward
        info["event"] = "waypoint"
    if target in self.visited_waypoints and target not in self.waypoints:
        reward += self.revisit_penalty
        info["event"] = "revisit"
    if target == self.goal:
        if set(self.waypoints).issubset(self.visited_waypoints):
            reward += self.goal_reward
            done = True
            info["event"] = "goal"
        else:
            reward += self.goal_before_waypoints_penalty
            info["event"] = "goal_before_waypoints"
    if self.max_steps is not None and self.steps >= self.max_steps and not done:
        done = True
        info["event"] = "timeout"
    return target, reward, done, info

GridWorldEnv.step_to = step_to_rl

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/map")
def get_map():
    with _env_lock:
        return {"map": env.get_map()}

@app.post("/reset")
def reset(req: ResetRequest):
    global env
    with _env_lock:
        w = req.width or env.width
        h = req.height or env.height
        s = req.start or env.start
        g = req.goal or env.goal
        wp = req.waypoints if req.waypoints is not None else list(env.waypoints)
        ob = req.obstacles if req.obstacles is not None else list(env.obstacles)
        ms = req.max_steps if req.max_steps is not None else 100
        env = GridWorldEnv(w, h, s, g, ob, wp, max_steps=ms)
        env.step_penalty = -0.5
        env.wall_penalty = -2.0
        env.obstacle_penalty = -5.0
        env.revisit_penalty = -1.0
        env.waypoint_reward = 20.0
        env.goal_reward = 50.0
        env.goal_before_waypoints_penalty = -5.0
        state = env.reset(max_steps=ms)
        return {"state": state, "map": env.get_map(), "ascii": env.render_ascii()}

@app.post("/reset_all")
def reset_all():
    global env
    with _env_lock:
        w, h = env.width, env.height
        start = (0, 0)
        all_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) != start]
        random.shuffle(all_cells)
        obstacles = all_cells[:8]
        remain = [cell for cell in all_cells if cell not in obstacles]
        waypoints = remain[:2]
        goal = remain[2]
        env = GridWorldEnv(w, h, start, goal, obstacles, waypoints, max_steps=100)
        env.step_penalty = -0.5
        env.wall_penalty = -2.0
        env.obstacle_penalty = -5.0
        env.revisit_penalty = -1.0
        env.waypoint_reward = 20.0
        env.goal_reward = 50.0
        env.goal_before_waypoints_penalty = -5.0
        state = env.reset(max_steps=100)
        return {
            "state": state,
            "map": env.get_map(),
            "ascii": env.render_ascii(),
            "obstacles": obstacles,
            "waypoints": waypoints,
            "goal": goal,
            "rewards_over_time": []
        }

@app.get("/state")
def get_state():
    with _env_lock:
        return {
            "state": env.get_state(),
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "ascii": env.render_ascii()
        }

@app.post("/step")
def step(inp: ActionInput):
    with _env_lock:
        try:
            if inp.action_name is not None:
                s, r, done, info = env.step_by_name(inp.action_name)
            elif inp.action is not None:
                s, r, done, info = env.step(inp.action)
            else:
                return {"error": "No action provided"}
            return {
                "state": s,
                "reward": r,
                "done": done,
                "info": info,
                "steps": env.steps,
                "visited_waypoints": list(env.visited_waypoints),
                "ascii": env.render_ascii()
            }
        except ValueError as e:
            return {"error": str(e)}

@app.post("/run_qlearning_greedy")
def run_qlearning_greedy():
    global ql_Q
    with _env_lock:
        start_time = time.time()
        start_xy = env.reset()
        env.visited_waypoints = set()
        schedule = list(env.waypoints) + [env.goal]
        scheduled_idx = 0
        state_xy = start_xy
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)

        # dist_to_next theo các waypoint chưa thăm → nếu hết thì đến goal
        unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
        if unvisited_wps:
            dist_to_next = min([manhattan_distance(state_xy, wp) for wp in unvisited_wps])
        else:
            dist_to_next = manhattan_distance(state_xy, env.goal)

        full_state = (state_xy[0], state_xy[1], visited_code, dist_to_next)
        done = False
        total_reward = 0
        steps = 0
        rewards_over_time = []
        path = [start_xy]

        # Vòng lặp theo done/steps
        while not done and steps < env.max_steps:
            target = schedule[scheduled_idx] if scheduled_idx < len(schedule) else env.goal

            if full_state in ql_Q and any(ql_Q[full_state].values()):
                max_q = max(ql_Q[full_state].values())
                best_actions = [a for a, q in ql_Q[full_state].items() if q == max_q]
                action_name = random.choice(best_actions)
            else:
                action_name = random.choice(actions)  # fallback

            action_idx = actions.index(action_name)
            next_state_xy, reward, done_env, _ = env.step(action_idx)

            if next_state_xy == target and scheduled_idx < len(schedule) - 1:
                scheduled_idx += 1

            visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
            if unvisited_wps:
                dist_to_next = min([manhattan_distance(next_state_xy, wp) for wp in unvisited_wps])
            else:
                dist_to_next = manhattan_distance(next_state_xy, env.goal)

            full_state = (next_state_xy[0], next_state_xy[1], visited_code, dist_to_next)
            done = done_env
            total_reward += reward
            rewards_over_time.append(total_reward)
            steps += 1
            path.append(next_state_xy)

        elapsed_time = time.time() - start_time
        return {
            "algorithm": "Q-Learning (Offline/Greedy, Waypoint Scheduling)",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": steps,
            "visited_waypoints": list(env.visited_waypoints),
            "ascii": env.render_ascii(),
            "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

@app.post("/run_mc_greedy")
def run_mc_greedy():
    global mc_Q
    with _env_lock:
        start_time = time.time()
        start_xy = env.reset()
        env.visited_waypoints = set()
        schedule = list(env.waypoints) + [env.goal]
        scheduled_idx = 0
        state_xy = start_xy
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)

        unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
        if unvisited_wps:
            dist_to_next = min([manhattan_distance(state_xy, wp) for wp in unvisited_wps])
        else:
            dist_to_next = manhattan_distance(state_xy, env.goal)

        full_state = (state_xy[0], state_xy[1], visited_code, dist_to_next)
        done = False
        total_reward = 0
        steps = 0
        rewards_over_time = []
        path = [start_xy]

        while not done and steps < env.max_steps:
            target = schedule[scheduled_idx] if scheduled_idx < len(schedule) else env.goal

            if full_state in mc_Q and any(mc_Q[full_state].values()):
                max_q = max(mc_Q[full_state].values())
                best_actions = [a for a, q in mc_Q[full_state].items() if q == max_q]
                action_name = random.choice(best_actions)
            else:
                # Fallback A* nếu state chưa biết
                path_to_target = a_star(state_xy, target, env.obstacles, env.width, env.height)
                if len(path_to_target) > 1:
                    next_pos = path_to_target[1]
                    dx, dy = next_pos[0] - state_xy[0], next_pos[1] - state_xy[1]
                    action_idx = env.ACTIONS.index((dx, dy))
                    action_name = actions[action_idx]
                else:
                    action_name = random.choice(actions)

            action_idx = actions.index(action_name)
            next_state_xy, reward, done_env, _ = env.step(action_idx)

            if next_state_xy == target and scheduled_idx < len(schedule) - 1:
                scheduled_idx += 1

            visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
            if unvisited_wps:
                dist_to_next = min([manhattan_distance(next_state_xy, wp) for wp in unvisited_wps])
            else:
                dist_to_next = manhattan_distance(next_state_xy, env.goal)

            full_state = (next_state_xy[0], next_state_xy[1], visited_code, dist_to_next)
            done = done_env
            total_reward += reward
            rewards_over_time.append(total_reward)
            steps += 1
            path.append(next_state_xy)

        elapsed_time = time.time() - start_time
        return {
            "algorithm": "MC (Offline/Greedy, Waypoint Scheduling)",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": steps,
            "visited_waypoints": list(env.visited_waypoints),
            "ascii": env.render_ascii(),
            "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

@app.post("/step_algorithm")
def step_algorithm(req: AlgorithmRequest):
    global epsilon, trajectory, mc_Q
    algo = req.algorithm
    with _env_lock:
        # trajectory (dùng cho MC)
        if 'trajectory' not in globals() or trajectory is None:
            trajectory = []

        state_xy = env.get_state()
        reward = 0
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)

        unvisited_wps = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
        if unvisited_wps:
            dist_to_next = min([manhattan_distance(state_xy, wp) for wp in unvisited_wps])
        else:
            dist_to_next = manhattan_distance(state_xy, env.goal)

        full_state = (state_xy[0], state_xy[1], visited_code, dist_to_next)
        done = False

        if algo == "MC":
            if np.random.rand() > epsilon:
                if full_state in mc_Q and any(mc_Q[full_state].values()):
                    action_name = max(mc_Q[full_state], key=mc_Q[full_state].get)
                else:
                    target = select_next_target(env)
                    path_to_target = a_star(state_xy, target, env.obstacles, env.width, env.height)
                    if len(path_to_target) > 1:
                        next_pos = path_to_target[1]
                        dx, dy = next_pos[0] - state_xy[0], next_pos[1] - state_xy[1]
                        action_idx = env.ACTIONS.index((dx, dy))
                        action_name = actions[action_idx]
                    else:
                        action_name = np.random.choice(actions)
            else:
                action_name = np.random.choice(actions)

            action_idx = actions.index(action_name)
            next_state, r, done, _ = env.step(action_idx)

            # log trajectory + MC first-visit update khi kết thúc episode
            trajectory.append((full_state, action_name, r))
            reward = r
            state_xy = next_state

            if done or env.steps >= env.max_steps:
                G = 0
                visited_state_actions = set()
                for state, action, r in reversed(trajectory):
                    G = r + gamma * G
                    state_action = (state, action)
                    if state_action not in visited_state_actions:
                        visited_state_actions.add(state_action)
                        old_q = mc_Q[state][action]
                        mc_Q[state][action] += alpha * (G - old_q)
                trajectory = []

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        elif algo == "Q-learning":
            # Greedy theo Q-table đã huấn luyện, KHÔNG train online
            if full_state in ql_Q and any(ql_Q[full_state].values()):
                action_name = max(ql_Q[full_state], key=ql_Q[full_state].get)
            else:
                action_name = np.random.choice(actions)

            action_idx = actions.index(action_name)
            next_state, r, done, _ = env.step(action_idx)
            reward = r
            state_xy = next_state
            # (không trajectory/không update ql_Q)

        elif algo == "SARSA":
            sarsa_state = (state_xy[0], state_xy[1], visited_code)
            if sarsa_state in sarsa_Q and any(sarsa_Q[sarsa_state].values()):
                max_q = max(sarsa_Q[sarsa_state].values())
                best_actions = [a for a, q in sarsa_Q[sarsa_state].items() if q == max_q]
                action_name = random.choice(best_actions)
            else:
                action_name = np.random.choice(actions)
            action_idx = actions.index(action_name)
            next_state, r, done, _ = env.step(action_idx)
            state_xy = next_state
            reward = r

        elif algo == "A2C":
            if not a2c_model_loaded:
                raise HTTPException(status_code=400, detail="A2C model not loaded. Please train or load a valid model.")

            target = select_next_target(env)
            state_tensor = env.build_grid_state().unsqueeze(0)
            a2c_model.eval()
            with torch.no_grad():
                policy_logits, _ = a2c_model(state_tensor)
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    action_idx = random.choice(range(n_actions))
                else:
                    action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
                    if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
                        action_idx = random.choice(range(n_actions))
                    else:
                        if random.random() < epsilon:
                            action_idx = random.choice(range(n_actions))
                        else:
                            try:
                                action_idx = torch.multinomial(action_probs, 1).item()
                            except RuntimeError:
                                action_idx = random.choice(range(n_actions))

            next_state, r, done, _ = env.step(action_idx)

            # A* fallback nếu không tiến gần target hoặc ăn reward xấu
            current_dist = manhattan_distance(state_xy, target)
            next_dist = manhattan_distance(next_state, target)
            if r <= env.obstacle_penalty or r == env.wall_penalty or (next_dist >= current_dist and r == env.step_penalty):
                path_to_target = a_star(env.get_state(), target, env.obstacles, env.width, env.height)
                if len(path_to_target) > 1:
                    next_pos = path_to_target[1]
                    dx, dy = next_pos[0] - env.get_state()[0], next_pos[1] - env.get_state()[1]
                    try:
                        action_idx = env.ACTIONS.index((dx, dy))
                        next_state, r, done, _ = env.step(action_idx)
                    except ValueError:
                        pass

            reward = r
            state_xy = next_state
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        return {
            "state": state_xy,
            "reward": reward,
            "done": done or env.steps >= env.max_steps,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints)
        }

@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
    with _env_lock:
        start_time = time.time()
        rewards_over_time = []
        start = env.get_state()
        path = plan_path_through_waypoints(start, env.waypoints, req.goal or env.goal, env.obstacles, env.width, env.height)
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}
        env.reset()
        total_reward = 0
        for node in path[1:]:
            s, r, done, info = env.step_to(node)
            total_reward += r
            rewards_over_time.append(total_reward)
        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))
        elapsed_time = time.time() - start_time
        return {
            "algorithm": "A*",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "info": {},
            "ascii": env.render_ascii(),
            "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

@app.post("/save_qlearning")
def save_qlearning():
    with open(QL_QFILE_OFFLINE, 'wb') as f:
        pickle.dump(dict(ql_Q), f)
    return {"status": "Q-learning Q-table saved to offline file"}

@app.post("/save_mc")
def save_mc():
    with open(os.path.join(models_dir, 'mc_qtable.pkl'), 'wb') as f:
        pickle.dump(dict(mc_Q), f)
    return {"status": "MC Q-table saved"}

@app.post("/save_sarsa")
def save_sarsa():
    with open(os.path.join(models_dir, 'sarsa_qtable.pkl'), 'wb') as f:
        pickle.dump(dict(sarsa_Q), f)
    return {"status": "SARSA Q-table saved"}

@app.post("/save_a2c")
def save_a2c():
    if not a2c_model_loaded:
        raise HTTPException(status_code=400, detail="A2C model not loaded or trained. Cannot save.")
    torch.save(a2c_model.state_dict(), os.path.join(models_dir, 'a2c_model.pth'))
    return {"status": "A2C model saved"}

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
