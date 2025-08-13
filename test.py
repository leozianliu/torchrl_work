import numpy as np

def interdist_to_reward(bot_interdist): # Input: float
    # Reward for distance to other robots
    scaling = 0.334
    bot_interdict = np.clip(bot_interdist, 0, 300)
    interdist_rew_single = scaling * bot_interdict - 1 # -1 rew when robots collide, >=3 distance no penalty no reward
    interdist_rew_single = min(0, interdist_rew_single)
    # interdist_rew_single = - np.exp(- bot_interdist / (scaling * min(MAP_SIZE)))
    return interdist_rew_single

print(interdist_to_reward(-2))