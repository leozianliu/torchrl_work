# multi_robot_simulator.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import os

MAP_SIZE = (300, 300)  # Grid size in cells

def generate_random_obstacles(num_obstacles=10, min_size=3, max_size=8):
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.randint(0, MAP_SIZE[0] - max_size)
        y = np.random.randint(0, MAP_SIZE[1] - max_size)
        size = np.random.randint(min_size, max_size + 1)
        obstacles.append((x, y, size))
    return obstacles

# Generate random obstacles
OBSTACLES = generate_random_obstacles(num_obstacles=100, min_size=4, max_size=10)
TOTAL_STEPS = 200

class Robot:
    def __init__(self, robot_id, pos, robot_type='UGV', battery_limit=100.0, comm_range=20.0):
        self.id = robot_id
        self.pos = np.array(pos, dtype=np.float32)
        self.traj = [self.pos.copy()]
        self.type = robot_type
        self.view_range = 10 if robot_type == 'UGV' else 30
        self.comm_range = comm_range
        self.known_map = np.zeros(MAP_SIZE)
        self.neighbors = []
        self.messages = []
        self.goal = np.array([np.random.randint(0, MAP_SIZE[0]), np.random.randint(0, MAP_SIZE[1])], dtype=np.float32)
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
                reward -= 1.0  # Penalty for collision
        
        # Goal reached reward
        if self.reached_goal:
            reward += 10.0
        
        # Cooperation reward (if sharing information with neighbors)
        if len(self.neighbors) > 0:
            reward += 0.1 * len(self.neighbors)
            
        return reward

    def observe(self):
        for ox, oy, size in OBSTACLES:
            cx, cy = np.array([ox + size/2, oy + size/2])
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

class MultiRobotEnv:
    def __init__(self, num_robots=3, enable_obstacle_check=True):
        self.num_robots = num_robots
        self.robots = []
        self.obstacles = generate_random_obstacles(num_obstacles=50, min_size=4, max_size=10)
        self.enable_obstacle_check = enable_obstacle_check
        self.robot_configs = None  # Store robot configurations for reset
        
        # Define state and action dimensions
        self.state_dim = 2 + 2 + 21*21 + 5*4  # position(2) + goal(2) + local_map(21x21) + neighbor_info(5*4)
        self.action_dim = 2  # 2D movement vector
        
        self.frames = []
        self.recording = False

    def initialize_robots(self, robot_configs):
        """
        Initialize robots with custom configurations
        robot_configs: list of tuples (position, robot_type, goal, battery_limit, comm_range)
        Example: [((10, 10), 'UAV', (30, 30), 120.0, 25.0), ((20, 20), 'UAV', (40, 40), 90.0, 20.0), ...]
        """
        self.robot_configs = robot_configs  # Store for reset
        self.robots = []
        for i, config in enumerate(robot_configs):
            if len(config) == 5:
                pos, robot_type, goal, battery_limit, comm_range = config
            elif len(config) == 4:
                pos, robot_type, goal, battery_limit = config
                comm_range = 20.0
            elif len(config) == 3:
                pos, robot_type, goal = config
                battery_limit = 100.0
                comm_range = 20.0
            else:
                pos, robot_type = config
                goal = None
                battery_limit = 100.0
                comm_range = 20.0
            robot = Robot(i, pos, robot_type, battery_limit, comm_range)
            if goal is not None:
                robot.goal = np.array(goal, dtype=np.float32)
            self.robots.append(robot)
        # Get initial observations
        observations = [robot.get_state() for robot in self.robots]
        return observations

    def reset(self):
        # Initialize robots with stored configuration
        self.robots = []
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
                    battery_limit = 100.0
                    comm_range = 20.0
                else:
                    pos, robot_type = config
                    goal = None
                    battery_limit = 100.0
                    comm_range = 20.0
                robot = Robot(i, pos, robot_type, battery_limit, comm_range)
                if goal is not None:
                    robot.goal = np.array(goal, dtype=np.float32)
                self.robots.append(robot)
        else:
            # Use default configuration
            initial_positions = [(10, 10), (20, 20), (50, 20)]
            for i in range(self.num_robots):
                robot_type = 'UAV' if i == 2 else 'UGV'
                self.robots.append(Robot(i, initial_positions[i], robot_type, 100.0, 20.0))
        # Get initial observations
        observations = [robot.get_state() for robot in self.robots]
        return observations

    def step(self, actions):
        # Execute actions for all robots
        rewards = []
        dones = []
        infos = []
        
        # First, let robots observe and communicate
        for robot in self.robots:
            robot.observe()
        for robot in self.robots:
            robot.communicate(self.robots)
        
        # Then execute actions
        for robot, action in zip(self.robots, actions):
            reward = robot.take_action(action, self.enable_obstacle_check)
            rewards.append(reward)
            dones.append(robot.reached_goal)
            infos.append({})
            
        # Get new observations
        observations = [robot.get_state() for robot in self.robots]
        
        # Render the current state
        self.render()
        
        return observations, rewards, dones, infos

    def render(self):
        # Create visualization
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            plt.ion()  # Turn on interactive mode
        
        # Clear previous frame
        self.ax.clear()
        
        # Draw map
        self.ax.set_xlim(0, MAP_SIZE[0])
        self.ax.set_ylim(0, MAP_SIZE[1])
        
        # Remove grid lines
        self.ax.grid(False)
        
        # Draw obstacles
        for ox, oy, size in self.obstacles:
            self.ax.add_patch(patches.Rectangle((ox, oy), size, size, color='gray'))
            
        # Draw robots
        for robot in self.robots:
            if robot.type == 'UGV':
                # Draw UGV as a square (3x larger)
                self.ax.add_patch(patches.Rectangle(
                    (robot.pos[0]-1.5, robot.pos[1]-1.5), 
                    3, 3, 
                    color='blue'
                ))
            else:
                # Draw UAV (3x larger)
                motor_radius = 0.9  # 3x larger
                arm_length = 2.1    # 3x larger
                angles = [0, np.pi/2, np.pi, 3*np.pi/2]
                
                for angle in angles:
                    motor_x = robot.pos[0] + arm_length * np.cos(angle)
                    motor_y = robot.pos[1] + arm_length * np.sin(angle)
                    self.ax.add_patch(patches.Circle((motor_x, motor_y), motor_radius, color='red'))
                
                # Center point (3x larger)
                self.ax.add_patch(patches.Circle(robot.pos, 0.6, color='red'))
                
                # Connecting lines
                for angle in angles:
                    motor_x = robot.pos[0] + arm_length * np.cos(angle)
                    motor_y = robot.pos[1] + arm_length * np.sin(angle)
                    self.ax.plot([robot.pos[0], motor_x], [robot.pos[1], motor_y], 'r-', linewidth=1)
            
            # Draw ranges and goals
            self.ax.add_patch(patches.Circle(robot.pos, robot.view_range, fill=False, linestyle='--', alpha=0.3))
            self.ax.add_patch(patches.Circle(robot.pos, robot.comm_range, fill=False, linestyle=':', color='green', alpha=0.2))
            self.ax.plot(robot.goal[0], robot.goal[1], 'kx')
        
        # Update the plot
        plt.draw()
        plt.pause(0.01)  # Small pause to allow the plot to update

if __name__ == "__main__":
    # Example usage
    env = MultiRobotEnv(num_robots=3)
    obs = env.reset()
    
    try:
        for _ in range(1000):
            # Random actions for demonstration
            actions = [np.random.rand(2) * 2 - 1 for _ in range(env.num_robots)]
            obs, rewards, dones, infos = env.step(actions)
            
            if all(dones):
                break
    finally:
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close the plot window
