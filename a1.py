import torch

from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from torchrl.modules import Actor
from torchrl.modules import MLP


env = GymEnv("Pendulum-v1")
# module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
module = MLP(
    out_features=env.action_spec.shape[-1],
    num_cells=[32, 64],
    activation_class=torch.nn.Tanh,
)

# policy = TensorDictModule(
#     module,
#     in_keys=["observation"],
#     out_keys=["action"],
# )

policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)