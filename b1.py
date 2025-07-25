from torchrl.envs import GymEnv

from torchrl.modules import Actor, MLP, ValueOperator
from torchrl.objectives import DDPGLoss
from torch.optim import Adam


env = GymEnv("Pendulum-v1")

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(
    MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
    in_keys=["observation", "action"],
)
ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)

rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)

total_loss = 0
print(loss_vals.items())
for key, val in loss_vals.items():
    if key.startswith("loss_"):
        total_loss += val

print(f"Total loss: {total_loss.item()}")
optim = Adam(ddpg_loss.parameters())
total_loss.backward()
optim.step()
optim.zero_grad()
