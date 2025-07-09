import torch

from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from torchrl.modules import Actor
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor
from torchrl.envs.utils import ExplorationType, set_exploration_type


env = GymEnv("Pendulum-v1")

backbone = MLP(in_features=3, out_features=2)
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True,
)

#with set_exploration_type(ExplorationType.DETERMINISTIC):
with set_exploration_type(ExplorationType.RANDOM):
    rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)