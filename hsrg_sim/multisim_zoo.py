# PettingZoo Parallel API compliant multi-robot environment
# Converted from Gymnasium single-agent wrapper to true multi-agent

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import yaml
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from gymnasium import spaces
import functools

#MAP_SIZE = (300, 300)  # Grid size in cells

class Helper:
    def read_yaml_config(config_dir):
        with open(str(config_dir), 'r') as config_file:
            return yaml.safe_load(config_file)

class Robot:
    def __init__(self, robot_id, pos, goal, robot_type, config, battery_limit, comm_range):
        self.id = robot_id
        self.pos = np.array(pos, dtype=np.float32)
        self.traj = [self.pos.copy()]
        self.type = robot_type
        self.comm_range = comm_range
        self.known_map = np.zeros(MAP_SIZE)
        self.neighbors = []
        self.max_neighbors = config['general_inputs']['max_neighbors']  # Maximum number of neighbors to consider
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
        # View ranges
        view_range_ugv = config['general_inputs']['view_range_ugv']
        view_range_uav = config['general_inputs']['view_range_uav']
        self.view_range = view_range_ugv if robot_type == 'UGV' else view_range_uav

    def get_state(self, obstacles): # NOTE: local_map is obsolete and irrelevant to RL agents
        # State includes: position, goal, known_map around robot, neighbors' info
        #local_map = self.get_local_map()
        neighbor_info = self.get_neighbor_info(self.max_neighbors)
        return np.concatenate([
            self.pos / np.asarray(MAP_SIZE),  # Normalized position
            self.goal / np.asarray(MAP_SIZE),  # Normalized goal
            self.raycast_distances(obstacles, self.pos, self.view_range) / self.view_range,  # Local map information
            neighbor_info  # Neighbor information
        ])

    # def get_local_map(self, size=10):
    #     # Get local map around robot
    #     x, y = int(self.pos[0]), int(self.pos[1])
    #     local_map = np.zeros((size*2+1, size*2+1))
    #     for i in range(-size, size+1):
    #         for j in range(-size, size+1):
    #             if 0 <= x+i < MAP_SIZE[0] and 0 <= y+j < MAP_SIZE[1]:
    #                 local_map[i+size, j+size] = self.known_map[x+i, y+j]
    #     return local_map
    
    def raycast_distances(self, obstacles, agent_pos, max_range, angle_step=20):
        """Cast rays from agent and return distances to nearest obstacles."""
        angles = np.arange(0, 360, angle_step)
        distances = []
        agent_row, agent_col = agent_pos
        obstacle_matrix = np.zeros(MAP_SIZE, dtype=int)
        for x, y, size in obstacles:
            obstacle_matrix[y:y+size, x:x+size] = 1
        rows, cols = obstacle_matrix.shape
        
        for angle in angles:
            rad = np.radians(angle)
            dx, dy = np.cos(rad), np.sin(rad)
            x, y, dist = agent_col, agent_row, 0.0
            
            while dist < max_range:
                x += dx * 0.5
                y += dy * 0.5
                dist += 0.5
                
                row, col = int(round(y)), int(round(x))
                if row < 0 or row >= rows or col < 0 or col >= cols or obstacle_matrix[row, col]:
                    break
            
            distances.append(min(dist, max_range))
        return np.array(distances)

    def get_neighbor_info(self, max_neighbors=5):
        # Get information about neighbors (position and goal)
        neighbor_info = np.zeros(max_neighbors * 4)  # 4 values per neighbor (x, y, goal_x, goal_y)
        for i, neighbor in enumerate(self.neighbors[:max_neighbors]):
            neighbor_info[i*4:(i+1)*4] = [
                neighbor.pos[0] / MAP_SIZE[0],
                neighbor.pos[1] / MAP_SIZE[1],
                neighbor.goal[0] / MAP_SIZE[0],
                neighbor.goal[1] / MAP_SIZE[1]
            ]
        return neighbor_info
    
    @staticmethod
    def check_obstacle_collision(robot_pos, obstacles, robot_clearance=1.0): # (x, y), list of (x, y, size), square size of bot; for UGV only
        # Simple collision detection with obstacles
        x, y = robot_pos
        for ox, oy, size in obstacles:
            # Check if robot overlaps with obstacle rectangle
            if (x - robot_clearance <= ox + size and 
                x + robot_clearance >= ox and
                y - robot_clearance <= oy + size and 
                y + robot_clearance >= oy):
                return True
        return False

    def take_action(self, action, robots, enable_obstacle_check=True, obstacles=None, obstacles_size_range=None):
        # Action is a 2D vector representing movement direction
        action = np.clip(action, -1, 1)  # Normalize action to [-1, 1]
        action_scaling = 2.0
        if action_scaling >= obstacles_size_range[0]:
            raise ValueError("Action scaling must be smaller than the minimum obstacle size, or it can jump across the obstacles")
        next_pos = self.pos + action * action_scaling # Scale action to reasonable step size
        
        x, y = int(next_pos[0]), int(next_pos[1])
        if enable_obstacle_check and self.type == 'UGV':
            if 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1] and not self.check_obstacle_collision(self.pos, obstacles, robot_clearance=1.0):
                self.pos = next_pos
        elif enable_obstacle_check and self.type == 'UAV': # Drones can fly over obstacles but not out of map
            if 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1]:
                self.pos = next_pos
        else:
            self.pos = next_pos
            
        self.traj.append(self.pos.copy())
        
        # Check if goal is reached
        if np.linalg.norm(self.pos - self.goal) < 0.1:
            self.reached_goal = True
            
        return self.get_reward(robots, obstacles)

    def get_reward(self, robots, obstacles):
        # Calculate reward based on:
        # 1. Distance to goal
        # 2. Collision with obstacles
        # 3. Goal reached
        # 4. Cooperation with neighbors
        
        def interdist_to_reward(bot_interdist): # Input: float
            # Reward for distance to other robots
            scaling = 0.001
            interdist_rew_single = - np.exp(- bot_interdist / (scaling * min(MAP_SIZE)))
            return interdist_rew_single
        
        def calculate_total_interdist_reward(robots):
            # Calculate inter-robot distance rewards
            interdist_reward = 0.0
            for i in range(len(robots)):
                for j in range(i + 1, len(robots)):
                    if robots[i].type == robots[j].type:  # Only consider distance between same type robots
                        dist = np.linalg.norm(robots[i].pos - robots[j].pos)
                        interdist_reward += interdist_to_reward(dist) # It's actually a penalty, sign is correct tho
                    else:
                        pass
            return interdist_reward
        
        reward = 0
        
        # Distance reward
        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reward -= dist_to_goal * 0.1  # Penalize distance to goal
        
        # Collision penalty
        x, y = self.pos[0], self.pos[1]
        if 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1]:
            if Robot.check_obstacle_collision(self.pos, obstacles, robot_clearance=1.0):
                reward -= 1.0  # Penalty for collision with obstacles
                
        # Inter-robot distance penalty
        interdist_reward = calculate_total_interdist_reward(robots)
        reward += interdist_reward  # Add inter-robot distance penalty
        
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


class MultiRobotParallelEnv(ParallelEnv):
    """PettingZoo Parallel Environment for Multi-Robot Coordination"""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multirobot_parallel_v0",
        "render_fps": 20,
    }

    def __init__(self, max_steps=1000, enable_obstacle_check=True, render_mode=None):
        """
        Initialize the parallel multi-robot environment
        
        Args:
            enable_obstacle_check (bool): Whether to enable obstacle collision checking
            render_mode (str): Render mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.config = Helper.read_yaml_config("hsrg_sim/setup1.yaml")
        global MAP_SIZE
        MAP_SIZE = self.config['general_inputs']['map_size']
        
        self.robots = []
        self.robot_configs = self.config['robots']
        self.num_robots = len(self.robot_configs)
        self.num_obstacles = self.config['general_inputs']['num_obstacles']
        self.obstacles_size_range = self.config['general_inputs']['obstacles_size_range']
        self.enable_obstacle_check = enable_obstacle_check
        
        # PettingZoo required attributes
        self.possible_agents = [f"robot_{i}" for i in range(self.num_robots)]
        self.agents = self.possible_agents[:]
        max_neighbors = self.config['general_inputs']['max_neighbors']
        
        # Action and observation spaces for each agent
        self._action_spaces = {agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) 
                              for agent in self.possible_agents}
        self._observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(4+4*max_neighbors+18,), dtype=np.float32) 
                                   for agent in self.possible_agents}
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Video recording
        self.video_writer = None
        self.video_frames = []
        
        # Environment state
        self.obstacles = []
        self._step_count = 0
        self.max_steps = max_steps
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for a given agent"""
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for a given agent"""
        return self._action_spaces[agent]
    
    # def _read_yaml_config(self, config_dir):
    #     with open(str(config_dir), 'r') as config_file:
    #         return yaml.safe_load(config_file)
    
    def _generate_random_obstacles(self, num_obstacles, min_size, max_size, rng):
        obstacles = []
        for _ in range(num_obstacles):
            x = rng.integers(0, MAP_SIZE[0] - max_size)
            y = rng.integers(0, MAP_SIZE[1] - max_size)
            size = rng.integers(min_size, max_size + 1)
            obstacles.append((x, y, size))
        return obstacles

    def _initialize_robots(self, rng, robot_configs=None):
        """Initialize robots with custom configurations"""
        self.robots = []
        default_battery_limit = 100.0
        default_comm_range = 20.0

        for robot_id, config in enumerate(robot_configs):
            robot_type = config.get('type')
            if robot_type not in ['UAV', 'UGV']:
                raise ValueError(f"Robot {robot_id} has invalid type: {robot_type}")

            # Extract position (random if not provided)
            if 'initial_position' in config:
                pos = np.array(config['initial_position'], dtype=np.float32)
            else:
                pos = np.array([
                    rng.integers(0, MAP_SIZE[0]),
                    rng.integers(0, MAP_SIZE[1])
                ], dtype=np.float32)

            # Extract goal (random if not provided)
            if 'goal' in config:
                goal = np.array(config['goal'], dtype=np.float32)
            else:
                goal = np.array([
                    rng.integers(0, MAP_SIZE[0]),
                    rng.integers(0, MAP_SIZE[1])
                ], dtype=np.float32)

            # Extract battery & comm range (default if not provided)
            battery_limit = config.get('battery_limit', default_battery_limit)
            comm_range = config.get('communication_range', default_comm_range)

            # Check for obstacle collisions (only for initial position)
            if Robot.check_obstacle_collision(pos, self.obstacles, robot_clearance=1.0):
                #print(f"Warning: Robot {robot_id} initial position collides with obstacles. Retrying...")
                for _ in range(10):  # Try 10 times to find a valid position
                    pos = np.array([rng.integers(0, MAP_SIZE[0]), rng.integers(0, MAP_SIZE[1])], dtype=np.float32)
                    if not Robot.check_obstacle_collision(pos, self.obstacles, robot_clearance=1.0):
                        break
                else:
                    raise ValueError(f"Robot {robot_id} could not find a valid position after 10 tries.")

            # Initialize robot
            self.robots.append(Robot(
                robot_id=robot_id,
                pos=pos,
                goal=goal,
                robot_type=robot_type,
                config=self.config,
                battery_limit=battery_limit,
                comm_range=comm_range
            ))

    def reset(self, seed=None, options=None):
        """
        Reset the environment (PettingZoo Parallel API)
        
        Returns:
            observations (dict): Dictionary mapping agent names to observations
            infos (dict): Dictionary mapping agent names to info dicts
        """
        if options is None:
            options = {}

        main_rng = np.random.default_rng(seed)

        obs_seed = options.get("seed_obstacle", None)
        pos_seed = options.get("seed_position", None)

        obstacle_rng = np.random.default_rng(obs_seed) if obs_seed is not None else main_rng
        position_rng = np.random.default_rng(pos_seed) if pos_seed is not None else main_rng

        # Generate obstacles and initialize robots
        self.obstacles = self._generate_random_obstacles(
            num_obstacles=self.num_obstacles,
            min_size=self.obstacles_size_range[0],
            max_size=self.obstacles_size_range[1],
            rng=obstacle_rng
        )
        
        self._initialize_robots(rng=position_rng, robot_configs=self.robot_configs)
        
        # Reset environment state
        self.agents = self.possible_agents[:]
        self._step_count = 0
        
        # Get initial observations
        observations = {}
        infos = {}
        
        for i, agent in enumerate(self.agents):
            observations[agent] = self.robots[i].get_state(self.obstacles)
            infos[agent] = {}
            
        return observations, infos

    def step(self, actions):
        """
        Execute one step (PettingZoo Parallel API)
        
        Args:
            actions (dict): Dictionary mapping agent names to actions
            
        Returns:
            observations (dict): Dictionary mapping agent names to observations
            rewards (dict): Dictionary mapping agent names to rewards
            terminations (dict): Dictionary mapping agent names to termination flags
            truncations (dict): Dictionary mapping agent names to truncation flags
            infos (dict): Dictionary mapping agent names to info dicts
        """
        # Handle case where not all agents provide actions
        if not actions:
            actions = {agent: np.array([0.0, 0.0]) for agent in self.agents}
        
        # Ensure all active agents have actions
        for agent in self.agents:
            if agent not in actions:
                actions[agent] = np.array([0.0, 0.0])  # Default action
        
        # Environment dynamics: observe and communicate first
        for robot in self.robots:
            if f"robot_{robot.id}" in self.agents:
                robot.observe(self.obstacles)
                
        for robot in self.robots:
            if f"robot_{robot.id}" in self.agents:
                robot.communicate(self.robots)
        
        # Execute actions and collect results
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for i, robot in enumerate(self.robots):
            agent = f"robot_{i}"
            
            if agent in self.agents:
                # Execute action
                action = actions.get(agent, np.array([0.0, 0.0]))
                reward = robot.take_action(
                    action, self.robots, self.enable_obstacle_check, 
                    self.obstacles, self.obstacles_size_range
                )
                
                # Get new observation
                observations[agent] = robot.get_state(self.obstacles)
                rewards[agent] = reward
                terminations[agent] = robot.reached_goal
                truncations[agent] = False  # Can add truncation logic here
                infos[agent] = {"reached_goal": robot.reached_goal}
                
        # Remove terminated agents
        self.agents = [agent for agent in self.agents if not terminations.get(agent, False)]
        
        # Check for truncation (max steps)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncations = {agent: True for agent in self.possible_agents}
        
        # Render if needed
        if self.render_mode == 'human':
            self.render(self.render_mode)
            
        return observations, rewards, terminations, truncations, infos

    def render(self, render_mode=None):
        """Render the environment"""
        if render_mode is None:
            return
        
        if render_mode == 'human':
            return self._render_human()
        elif render_mode == 'rgb_array':
            return self._render_rgb_array()

    def _render_human(self):
        """Render for human viewing"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=np.array(MAP_SIZE)/20)
            plt.ion()
        
        self.ax.clear()
        self._draw_scene()
        plt.draw()
        plt.pause(0.01)
        
        # Record frame if video recording is active
        if self.video_writer is not None:
            self.video_writer.grab_frame()

    def _render_rgb_array(self):
        """Render and return RGB array"""
        raise NotImplementedError("RGB array rendering is not implemented yet. Use human which could be also saved.")

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
                # UAV drawing code
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
        """Close the environment"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        if self.video_writer is not None:
            self.video_writer.finish()

    # Video recording methods
    def start_video_recording(self, filename='simulation.mp4', fps=20, bitrate=2400):
        """Start video recording"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=np.array(MAP_SIZE)/20)
        
        self.video_writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        self.video_writer.setup(self.fig, filename, dpi=80)

    def stop_video_recording(self):
        """Stop video recording"""
        if self.video_writer is not None:
            self.video_writer.finish()
            self.video_writer = None
            print("Video saved!")

    @property
    def max_num_agents(self):
        """Return the maximum number of agents"""
        return len(self.possible_agents)

if __name__ == "__main__":
    
    def example_with_video_pz():
        """Example with video recording using PettingZoo"""
        env = MultiRobotParallelEnv(render_mode='human')
        env.start_video_recording('pettingzoo_simulation.mp4')
        
        observations, infos = env.reset(options={"seed_obstacle": 420, "seed_position": 240})
        
        try:
            for step in range(200):
                actions = {agent: np.random.rand(2) * 2 - 1 for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                if not env.agents:
                    break
                    
        finally:
            env.stop_video_recording()
            env.close()
    
    # Test the environment
    env = MultiRobotParallelEnv()
    print(f"Environment created with agents: {env.possible_agents}")
    print(f"Action spaces: {[env.action_space(agent) for agent in env.possible_agents]}")
    print(f"Observation spaces: {[env.observation_space(agent) for agent in env.possible_agents]}")
    
    # Example with TorchRL wrapper
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper
    
    # Solution 1: Use custom group_map to put all agents under "agents" key
    group_map = {"agents": [f"robot_{i}" for i in range(len(env.possible_agents))]}
    wrapped_env = PettingZooWrapper(env=env, group_map=group_map)
    print("Wrapped environment spec:")
    print(wrapped_env.specs)
    
    # Test reset
    tensordict = wrapped_env.reset()
    print("Reset TensorDict structure:")
    print(tensordict)
    
    # Run example
    example_with_video_pz()