# Gym-compliant video recording approach
# Treated as a single agent for teacher network training

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import gymnasium as gym
from gymnasium import spaces

MAP_SIZE = (300, 300)  # Grid size in cells

class Robot:
    def __init__(self, robot_id, pos, goal, robot_type='UGV', battery_limit=100.0, comm_range=20.0):
        self.id = robot_id
        self.pos = np.array(pos, dtype=np.float32)
        self.traj = [self.pos.copy()]
        self.type = robot_type
        self.view_range = 10 if robot_type == 'UGV' else 30
        self.comm_range = comm_range
        self.known_map = np.zeros(MAP_SIZE)
        self.neighbors = []
        self.messages = []
        self.goal = goal # (x, y) for one robot; multiple instances of Robot will be created in initialize_robots
        self.reached_goal = False
        self.battery_limit = battery_limit
        # Task management related attributes
        self.tasks = []  # current task list
        self.task_path = []  # current task path
        self.task_points = None  # task point coordinates (set externally)
        # Distributed architecture related attributes
        self.planner = None  # planner instance
        self.meeting_tags = []  # meeting tags

    def get_state(self):
        # State includes: position, goal, known_map around robot, neighbors' info
        local_map = self.get_local_map()
        neighbor_info = self.get_neighbor_info()
        return np.concatenate([
            self.pos / MAP_SIZE[0],  # Normalized position
            self.goal / MAP_SIZE[0],  # Normalized goal
            local_map.flatten(),  # Local map information
            neighbor_info  # Neighbor information
        ])

    def get_local_map(self, size=10):
        # Get local map around robot
        x, y = int(self.pos[0]), int(self.pos[1])
        local_map = np.zeros((size*2+1, size*2+1))
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                if 0 <= x+i < MAP_SIZE[0] and 0 <= y+j < MAP_SIZE[1]:
                    local_map[i+size, j+size] = self.known_map[x+i, y+j]
        return local_map

    def get_neighbor_info(self, max_neighbors=5):
        # Get information about neighbors (position and goal)
        neighbor_info = np.zeros(max_neighbors * 4)  # 4 values per neighbor (x, y, goal_x, goal_y)
        for i, neighbor in enumerate(self.neighbors[:max_neighbors]):
            neighbor_info[i*4:(i+1)*4] = [
                neighbor.pos[0] / MAP_SIZE[0],
                neighbor.pos[1] / MAP_SIZE[0],
                neighbor.goal[0] / MAP_SIZE[0],
                neighbor.goal[1] / MAP_SIZE[0]
            ]
        return neighbor_info

    def take_action(self, action, enable_obstacle_check=True):
        # Action is a 2D vector representing movement direction
        action = np.clip(action, -1, 1)  # Normalize action to [-1, 1]
        next_pos = self.pos + action * 0.5  # Scale action to reasonable step size
        
        if enable_obstacle_check and self.type == 'UGV':
            x, y = int(next_pos[0]), int(next_pos[1])
            if 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1] and self.known_map[x, y] != 1:
                self.pos = next_pos
        else:  # UAV ignores obstacles or obstacle check is disabled
            self.pos = next_pos
            
        self.traj.append(self.pos.copy())
        
        # Check if goal is reached
        if np.linalg.norm(self.pos - self.goal) < 0.1:
            self.reached_goal = True
            
        return self.get_reward()

    def get_reward(self):
        # Calculate reward based on:
        # 1. Distance to goal
        # 2. Collision with obstacles
        # 3. Goal reached
        # 4. Cooperation with neighbors
        
        reward = 0
        
        # Distance reward
        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reward -= dist_to_goal * 0.1  # Penalize distance to goal
        
        # Collision penalty
        x, y = int(self.pos[0]), int(self.pos[1])
        if 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1]:
            if self.known_map[x, y] == 1:
                reward -= 1.0  # Penalty for collision with obstacles
        
        # Goal reached reward
        if self.reached_goal:
            reward += 10.0
        
        # Cooperation reward (if sharing information with neighbors)
        if len(self.neighbors) > 0:
            reward += 0.1 * len(self.neighbors)
            
        return reward

    def observe(self, obstacles):
        for ox, oy, size in obstacles:
            cx, cy = np.array([ox + size/2, oy + size/2]) # center of obstacle
            dist = np.linalg.norm(self.pos - [cx, cy])
            if dist <= self.view_range:
                for dx in range(size):
                    for dy in range(size):
                        xx, yy = ox + dx, oy + dy
                        if 0 <= xx < MAP_SIZE[0] and 0 <= yy < MAP_SIZE[1]:
                            self.known_map[xx, yy] = 1

    def communicate(self, others):
        self.neighbors = []
        self.messages = []
        for r in others:
            if r.id != self.id:
                dist = np.linalg.norm(r.pos - self.pos)
                if dist <= self.comm_range:
                    self.neighbors.append(r)
                    self.messages.append(r.share_info())

    def share_info(self):
        return {
            'id': self.id, 
            'pos': self.pos.copy(), 
            'goal': self.goal.copy(),
            'tasks': self.tasks.copy()  # share task information
        }

    def set_task_points(self, task_points):
        """Set task point coordinates"""
        self.task_points = task_points

    def set_tasks(self, tasks):
        """Set task list"""
        self.tasks = tasks.copy()

    def set_planner(self, planner):
        """Set planner"""
        self.planner = planner

    def update_meeting_tags(self, tags):
        """Update meeting tags"""
        self.meeting_tags = tags

class MultiRobotEnv(gym.Env):
    metadata = {'render.modes': 'rgb_array', 'render_fps': 20}

    def __init__(self, num_robots=3, enable_obstacle_check=True, render_mode=None):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        #self.observation_space = spaces.Box()
        
        self.num_robots = num_robots
        self.robots = []
        self.num_obstacles = 50
        self.obstacles_size_range = (4, 10)
        #self.obstacles = self._generate_random_obstacles(num_obstacles=50, min_size=4, max_size=10)
        self.enable_obstacle_check = enable_obstacle_check
        self.robot_configs = None
        
        # Gym-standard attributes
        self.render_mode = render_mode  # None, 'human', 'rgb_array'
        self.screen = None
        self.clock = None
        
        # Video recording (gym-style)
        self.video_writer = None
        self.video_frames = []
        
        self.state_dim = 2 + 2 + 21*21 + 5*4
        self.action_dim = 2
        
    def _generate_random_obstacles(self, num_obstacles, min_size, max_size, rng):
        obstacles = []
        for _ in range(num_obstacles):
            x = rng.randint(0, MAP_SIZE[0] - max_size)
            y = rng.randint(0, MAP_SIZE[1] - max_size)
            size = rng.randint(min_size, max_size + 1)
            obstacles.append((x, y, size))
        return obstacles

    def _initialize_robots(self, reset_options, rng, robot_configs=None): # Not used at the moment
        """
        Initialize robots with custom configurations
        robot_configs: list of tuples (position, robot_type, goal, battery_limit, comm_range)
        Example: [((10, 10), 'UAV', (30, 30), 120.0, 25.0), ((20, 20), 'UAV', (40, 40), 90.0, 20.0), ...]
        """
        self.robot_configs = robot_configs  # Store for reset
        # Initialize robots with stored configuration
        self.robots = []
        default_battery_limit = 100.0
        default_comm_range = 20.0
        if self.robot_configs is not None:
            # Use stored robot configurations
            for i, config in enumerate(self.robot_configs):
                if len(config) == 5:
                    pos, robot_type, goal, battery_limit, comm_range = config
                elif len(config) == 4:
                    pos, robot_type, goal, battery_limit = config
                    comm_range = 20.0
                elif len(config) == 3:
                    pos, robot_type, goal = config
                    battery_limit = default_battery_limit
                    comm_range = default_comm_range
                else:
                    raise ValueError("Invalid robot configuration length")
                robot = Robot(i, pos, robot_type, battery_limit, comm_range)
                robot.goal = np.array(goal, dtype=np.float32)
                self.robots.append(robot)
        elif (reset_options is not None) and reset_options != {}:
            # Use random position and goal
            for i in range(self.num_robots):
                rand_init_pos = np.array([rng.randint(0, MAP_SIZE[0]), 
                                    rng.randint(0, MAP_SIZE[1])], dtype=np.float32)
                rand_goal = np.array([rng.randint(0, MAP_SIZE[0]), 
                                    rng.randint(0, MAP_SIZE[1])], dtype=np.float32)
                robot_type = 'UAV' if i == 2 else 'UGV'
                self.robots.append(Robot(robot_id=i, 
                                         pos=rand_init_pos, 
                                         goal=rand_goal,
                                         robot_type=robot_type, 
                                         battery_limit=default_battery_limit, 
                                         comm_range=default_comm_range))
        else:
            # Use default pos and random goals
            init_positions = [(10, 10), (20, 20), (50, 20)]
            rand_goals = []
            for _ in range(self.num_robots):
                rand_goals.append(
                    np.array([rng.randint(0, MAP_SIZE[0]),
                              rng.randint(0, MAP_SIZE[1])], dtype=np.float32))
            
            for i in range(self.num_robots):
                robot_type = 'UAV' if i == 2 else 'UGV'
                self.robots.append(Robot(i, 
                                         init_positions[i], 
                                         rand_goals[i], 
                                         robot_type, 
                                         default_battery_limit, 
                                         default_comm_range))
                
        # Get initial observations
        observations = [robot.get_state() for robot in self.robots]
        return observations

    def reset(self, seed=None, options=None):
        """Gym-compliant reset method"""
        if seed is not None:
            np.random.seed(seed)
        if options is None:
            options = {}
            
        obs_seed = options.get("seed_obstacle", None)
        pos_seed = options.get("seed_position", None)
        if obs_seed is not None:
            obstacle_rng = np.random.RandomState(obs_seed)
        else:
            obstacle_rng = np.random
        if pos_seed is not None:
            position_rng = np.random.RandomState(pos_seed) # random goal and position share the same seed
        else:
            position_rng = np.random
        self.obstacles = self._generate_random_obstacles(
            num_obstacles=self.num_obstacles, 
            min_size=self.obstacles_size_range[0], 
            max_size=self.obstacles_size_range[1], 
            rng=obstacle_rng) # Generate random obstacles with the seed
        observations = self._initialize_robots(
            reset_options=options, 
            rng=position_rng, 
            robot_configs=self.robot_configs) # Initialize robots with the seed
        
        info = {}
        return observations, info # Gym v0.26+ returns (observation, info)

    def step(self, actions):
        """Gym-compliant step method"""
        rewards = []
        dones = []
        infos = []
        
        # Your existing step logic...
        for robot in self.robots:
            robot.observe(self.obstacles)
        for robot in self.robots:
            robot.communicate(self.robots)
        
        for robot, action in zip(self.robots, actions):
            reward = robot.take_action(action, self.enable_obstacle_check)
            rewards.append(reward)
            dones.append(robot.reached_goal)
            infos.append({})
            
        observations = [robot.get_state() for robot in self.robots]
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        # Gym v0.26+ format: obs, reward, terminated, truncated, info
        terminated = dones
        truncated = [False] * len(dones)  # Add truncation logic if needed
        
        return observations, rewards, terminated, truncated, infos

    def render(self):
        """Gym-compliant render method"""
        if self.render_mode is None:
            return
        
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()

    def _render_human(self):
        """Render for human viewing"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()
        
        self.ax.clear()
        self._draw_scene()
        plt.draw()
        plt.pause(0.01)
        
        # Record frame if video recording is active
        if self.video_writer is not None:
            self.video_writer.grab_frame()

    def _render_rgb_array(self):
        """Render and return RGB array (for video recording)"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        self._draw_scene()
        self.fig.canvas.draw()
        
        # Convert to RGB array
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf

    def _draw_scene(self):
        """Draw the simulation scene"""
        self.ax.set_xlim(0, MAP_SIZE[0])
        self.ax.set_ylim(0, MAP_SIZE[1])
        self.ax.grid(False)
        
        # Draw obstacles
        for ox, oy, size in self.obstacles:
            self.ax.add_patch(patches.Rectangle((ox, oy), size, size, color='gray'))
            
        # Draw robots
        for robot in self.robots:
            if robot.type == 'UGV':
                self.ax.add_patch(patches.Rectangle(
                    (robot.pos[0]-1.5, robot.pos[1]-1.5), 
                    3, 3, 
                    color='blue'
                ))
            else:
                # UAV drawing code...
                motor_radius = 0.9
                arm_length = 2.1
                angles = [0, np.pi/2, np.pi, 3*np.pi/2]
                
                for angle in angles:
                    motor_x = robot.pos[0] + arm_length * np.cos(angle)
                    motor_y = robot.pos[1] + arm_length * np.sin(angle)
                    self.ax.add_patch(patches.Circle((motor_x, motor_y), motor_radius, color='red'))
                
                self.ax.add_patch(patches.Circle(robot.pos, 0.6, color='red'))
                
                for angle in angles:
                    motor_x = robot.pos[0] + arm_length * np.cos(angle)
                    motor_y = robot.pos[1] + arm_length * np.sin(angle)
                    self.ax.plot([robot.pos[0], motor_x], [robot.pos[1], motor_y], 'r-', linewidth=1)
            
            # Draw ranges and goals
            self.ax.add_patch(patches.Circle(robot.pos, robot.view_range, fill=False, linestyle='--', alpha=0.3))
            self.ax.add_patch(patches.Circle(robot.pos, robot.comm_range, fill=False, linestyle=':', color='green', alpha=0.2))
            self.ax.plot(robot.goal[0], robot.goal[1], 'kx')

    def close(self):
        """Gym-compliant close method"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        if self.video_writer is not None:
            self.video_writer.finish()

    # Video recording methods (gym-style)
    def start_video_recording(self, filename='simulation.mp4'):
        """Start video recording (gym-style)"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.video_writer = FFMpegWriter(fps=20, bitrate=2400)
        self.video_writer.setup(self.fig, filename, dpi=80)

    def stop_video_recording(self):
        """Stop video recording"""
        if self.video_writer is not None:
            self.video_writer.finish()
            self.video_writer = None
            print("Video saved!")


# Usage examples:

def example_with_video(file_dir):
    """Example using gym-style video recording"""
    # Create environment with human rendering
    env = MultiRobotEnv(num_robots=3, render_mode='human')
    
    # Start video recording
    env.start_video_recording(file_dir)
    
    # Standard gym loop
    observations, info = env.reset(options={"seed_obstacle": 42, "seed_position": 24})
    
    try:
        for step in range(200):
            # Random actions
            actions = [np.random.rand(2) * 2 - 1 for _ in range(env.num_robots)]
            
            # Gym-style step
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Check if done
            if all(terminated):
                break
                
    finally:
        env.stop_video_recording()
        env.close()

def example_with_rgb_arrays():
    """Example collecting RGB arrays (for custom video processing)"""
    env = MultiRobotEnv(num_robots=3, render_mode='rgb_array')
    
    frames = []
    observations, info = env.reset(options={"seed_obstacle": 42, "seed_position": 24})
    
    for step in range(100):
        actions = [np.random.rand(2) * 2 - 1 for _ in range(env.num_robots)]
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # Get RGB frame
        frame = env.render()
        frames.append(frame)
        
        if all(terminated):
            break
    
    env.close()
    
    # Now you can process frames however you want
    print(f"Collected {len(frames)} frames")
    return frames

if __name__ == "__main__":
    example_with_video('hsrg_sim/gym_style_simulation.mp4')