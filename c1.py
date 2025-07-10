import tempfile
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer


torch.manual_seed(0)

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

buffer_scratch_dir = tempfile.TemporaryDirectory().name
buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000, scratch_dir=buffer_scratch_dir))

n = 0
for data in collector:
    print(f"Trajectory ID: {data["collector", "traj_ids"]}")
    indices = buffer.extend(data)
    print(len(buffer))

    n += 1
    if n == 2:
        break

sample = buffer.sample(batch_size=30)
print(sample["observation"])

