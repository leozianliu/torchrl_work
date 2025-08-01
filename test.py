import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def check_obstacle_collision(robot_pos, obstacles, robot_clearance=0.5): # (x, y), list of (x, y, size), square size of bot; for UGV only
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

print(check_obstacle_collision((3.5, 4.1), [(4, 4, 2), (7, 7, 2)]))  # Example usage