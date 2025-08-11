import torch

from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy

torch.manual_seed(0)

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

for i, tensordict_data in enumerate(collector):
    print(tensordict_data)
    if i == 4:
        break