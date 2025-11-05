# app/robot_env.py

from typing import List, Tuple, Optional
import torch
import random

class GridWorldEnv:
    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ACTION_NAMES = ["up", "right", "down", "left"]

    def __init__(self, width=10, height=10, start=(0,0), goal=(9,9),
                 obstacles=None, waypoints=None, max_steps: Optional[int]=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles: set[Tuple[int,int]] = set(obstacles or [])
        self.waypoints: List[Tuple[int,int]] = waypoints or []
        self.visited_waypoints: set[Tuple[int,int]] = set()
        self.max_steps = max_steps
        self.state = self.start
        self.steps = 0

        # Reward parameters (tối ưu hơn)
        self.step_penalty = -0.5
        self.wall_penalty = -2
        self.obstacle_penalty = -5
        self.revisit_penalty = -1
        self.waypoint_reward = 20
        self.goal_reward = 50
        self.goal_before_waypoints_penalty = -5

    def reset(self, start=None, goal=None, obstacles=None, waypoints=None, max_steps=None):
        if start: self.start = start
        if goal: self.goal = goal
        if obstacles: self.obstacles = set(obstacles)
        if waypoints is not None: self.waypoints = waypoints
        self.visited_waypoints = set()
        self.state = self.start
        self.steps = 0
        if max_steps is not None:
            self.max_steps = max_steps
        return self.state

    def step(self, action: int):
        assert action in [0,1,2,3]
        dx, dy = GridWorldEnv.ACTIONS[action]
        return self.step_vector(dx, dy)

    def step_by_name(self, action_name: str):
        if action_name not in GridWorldEnv.ACTION_NAMES:
            raise ValueError(f"Invalid action_name {action_name}")
        idx = GridWorldEnv.ACTION_NAMES.index(action_name)
        return self.step(idx)

    def step_vector(self, dx: int, dy: int):
        x, y = self.state
        nx, ny = x + dx, y + dy

        reward = self.step_penalty
        done = False
        info = {}

        # Check bounds
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            nx, ny = x, y
            reward = self.wall_penalty
            info["event"] = "wall"

        # Check obstacle
        elif (nx, ny) in self.obstacles:
            nx, ny = x, y
            reward = self.obstacle_penalty
            info["event"] = "obstacle"

        else:
            self.state = (nx, ny)

            # Check waypoint
            if self.state in self.waypoints and self.state not in self.visited_waypoints:
                self.visited_waypoints.add(self.state)
                reward += self.waypoint_reward
                info["event"] = "waypoint"

            # Phạt đi lại ô cũ (chỉ phạt nếu ô đó không phải waypoint)
            # Sửa lỗi logic phạt revisit: Chỉ phạt nếu ô đã thăm *VÀ* không phải là waypoint *chưa thăm* lần đầu
            # (Logic cũ có thể phạt cả khi vừa đến waypoint)
            # => Thực ra logic cũ đã đúng: if self.state in self.visited_waypoints and self.state not in self.waypoints:
            # => Giữ nguyên logic cũ
            if self.state in self.visited_waypoints and self.state not in self.waypoints:
                 reward += self.revisit_penalty
                 # Không đặt info["event"] = "revisit" vì có thể ghi đè "waypoint"

        # Check goal
        if self.state == self.goal:
            if set(self.waypoints).issubset(self.visited_waypoints):
                reward += self.goal_reward
                done = True
                info["event"] = "goal"
            else:
                reward += self.goal_before_waypoints_penalty
                done = False # Vẫn chưa xong nếu chưa đủ waypoint
                info["event"] = "goal_before_waypoints"

        # Timeout
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            done = True
            info["event"] = "timeout"
        
        # Luôn trả về danh sách visited_waypoints trong info
        info["visited_waypoints"] = list(self.visited_waypoints)
        return self.state, reward, done, info

    def get_map(self):
        return {
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "goal": self.goal,
            "obstacles": list(self.obstacles),
            "waypoints": self.waypoints,
            "max_steps": self.max_steps
        }

    def render_ascii(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for ox, oy in self.obstacles:
            if 0 <= ox < self.width and 0 <= oy < self.height:
                grid[oy][ox] = "#"
        for wx, wy in self.waypoints:
            if 0 <= wx < self.width and 0 <= wy < self.height:
                grid[wy][wx] = "v" if (wx, wy) in self.visited_waypoints else "W"
        sx, sy = self.start
        gx, gy = self.goal
        x, y = self.state
        if 0 <= sx < self.width and 0 <= sy < self.height:
            # Chỉ hiển thị 'S' nếu robot không ở ô start
            if (x, y) != (sx, sy):
                grid[sy][sx] = "S"
        if 0 <= gx < self.width and 0 <= gy < self.height:
            # Chỉ hiển thị 'G' nếu robot không ở ô goal
            if (x, y) != (gx, gy):
                grid[gy][gx] = "G"
        if 0 <= x < self.width and 0 <= y < self.height:
            grid[y][x] = "R"
        return "\n".join(" ".join(row) for row in grid)

    def get_state(self):
        return self.state

    def is_done(self):
        done_goal = self.state == self.goal and set(self.waypoints).issubset(self.visited_waypoints)
        done_timeout = self.max_steps is not None and self.steps >= self.max_steps
        return done_goal or done_timeout

    # --- (BẮT ĐẦU SỬA LỖI build_grid_state) ---
    # Thêm tham số current_target
    def build_grid_state(self, current_target=None):
        """Trả về tensor 5 kênh (robot, target, obstacles, waypoint chưa thăm, waypoint đã thăm)"""
        
        # Nếu không cung cấp target, mặc định là goal (cho A*)
        target_pos = current_target if current_target is not None else self.goal

        grid = torch.zeros(5, self.height, self.width, dtype=torch.float32)
        rx, ry = self.state
        # Sửa: Dùng target_pos đã xác định
        tx, ty = target_pos
        
        # Đảm bảo tọa độ nằm trong grid trước khi gán
        if 0 <= ry < self.height and 0 <= rx < self.width:
             grid[0, ry, rx] = 1.0 # Kênh 0: Robot
        if 0 <= ty < self.height and 0 <= tx < self.width:
             grid[1, ty, tx] = 1.0 # Kênh 1: Mục tiêu HIỆN TẠI (waypoint hoặc goal)
        
        for ox, oy in self.obstacles:
            if 0 <= oy < self.height and 0 <= ox < self.width:
                grid[2, oy, ox] = 1.0 # Kênh 2: Obstacles
        for wx, wy in self.waypoints:
             if 0 <= wy < self.height and 0 <= wx < self.width:
                if (wx, wy) not in self.visited_waypoints:
                    grid[3, wy, wx] = 1.0 # Kênh 3: Các waypoint CHƯA thăm
                else:
                    grid[4, wy, wx] = 1.0 # Kênh 4: Các waypoint ĐÃ thăm
        return grid
    # --- (KẾT THÚC SỬA LỖI build_grid_state) ---
    
    def step_to(self, target):
        """Di chuyển trực tiếp robot tới ô target (A* dùng)."""
        # Cần tính reward và done tương tự step() để A* chạy đúng
        prev_state = self.state
        self.state = target
        self.steps += 1
        reward = self.step_penalty # Phạt bước đi cơ bản
        done = False
        info = {"note": "Auto move by A*"}

        if target in self.waypoints and target not in self.visited_waypoints:
            self.visited_waypoints.add(target)
            reward += self.waypoint_reward
            info["event"] = "waypoint"
        # Logic phạt revisit không áp dụng cho A* vì A* không quay lại ô cũ trong 1 sub-path
        # else: # Không cần else ở đây
        if target == self.goal:
            if set(self.waypoints).issubset(self.visited_waypoints):
                reward += self.goal_reward
                done = True
                info["event"] = "goal"
            else:
                reward += self.goal_before_waypoints_penalty
                done = False # Chưa xong nếu chưa đủ waypoint
                info["event"] = "goal_before_waypoints"

        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            done = True
            info["event"] = "timeout"
            
        info["visited_waypoints"] = list(self.visited_waypoints)
        return self.state, reward, done, info # Trả về reward, done, info đầy đủ
    
    def randomize_map(self, n_obstacles=5, n_waypoints=2):
        """Ngẫu nhiên hóa vị trí obstacle, waypoint, goal."""
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) != self.start]
        random.shuffle(all_cells)

        self.obstacles = set(all_cells[:n_obstacles])
        remaining_cells = [cell for cell in all_cells if cell not in self.obstacles]

        if len(remaining_cells) < n_waypoints + 1:
            print(f"Warning: Không đủ ô trống ({len(remaining_cells)}) để chọn {n_waypoints} waypoints và 1 goal. Giảm số lượng.")
            n_waypoints = max(0, len(remaining_cells) - 1)
            # raise ValueError("Không đủ ô trống để chọn waypoint và goal.")

        self.waypoints = remaining_cells[:n_waypoints]
        if len(remaining_cells) > n_waypoints:
            self.goal = remaining_cells[n_waypoints]
        else:
            # Trường hợp cực hiếm: không còn ô nào cho goal -> đặt goal trùng start (không lý tưởng)
            print("Warning: Không còn ô trống cho goal, đặt trùng start.")
            self.goal = self.start
            
        self.reset() # Reset lại trạng thái sau khi random map