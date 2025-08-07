import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def raycast_distances(obstacles, agent_pos):
    """Cast rays from agent and return distances to nearest obstacles."""
    obstacle_matrix = np.zeros(MAP_SIZE, dtype=int)
    for x, y, size in obstacles:
        obstacle_matrix[y:y+size, x:x+size] = 1
    rows, cols = obstacle_matrix.shape
                
    row, col = int(agent_pos[1]), int(agent_pos[0])
    if row < 0 or row >= rows or col < 0 or col >= cols or obstacle_matrix[row, col]:
        print('On obstacle')
        
if __name__ == "__main__":
    MAP_SIZE = (50, 50)
    obstacles = [(10, 10, 5), (30, 20, 3), (15, 35, 4)]  # (x, y, size)
    raycast_distances(obstacles, (32.9, 22.9))
  
    