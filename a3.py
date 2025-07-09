import torch

from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from torchrl.modules import Actor
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule


env = GymEnv("Pendulum-v1")
policy = Actor(MLP(3, 1, num_cells=[32, 64]))
exploration_module = EGreedyModule(spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5)
exploration_policy = TensorDictSequential(policy, exploration_module)

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # Turns off exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
    # Turns on exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)

print(rollout["action"])