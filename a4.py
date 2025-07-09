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
from torchrl.modules import QValueModule

env = GymEnv("CartPole-v1")
#print(env.action_spec)
num_actions = 2
value_net = TensorDictModule(
    MLP(out_features=num_actions, num_cells=[32, 32]),
    in_keys=["observation"],
    out_keys=["action_value"],
)
policy = TensorDictSequential(
    value_net,  # writes action values in our tensordict
    QValueModule(spec=env.action_spec),  # Reads the "action_value" entry by default
)
policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec))

with set_exploration_type(ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)
print(rollout_explore)