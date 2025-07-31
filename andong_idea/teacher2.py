# Teacher but with a scalar value function
# Torch
import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing, nn

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueOperator

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators, value

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 8_000  # Number of team frames collected per training iteration
n_iters = 50 # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.1  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 3e-4

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

max_steps = 400  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "discovery"
n_agents = 4
agents_per_target = 2

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    agents_per_target = agents_per_target # A target is considered covered if agents_per_target agents have approached a target
)
# print("action_spec:", env.full_action_spec)
# print("reward_spec:", env.full_reward_spec)
# print("done_spec:", env.full_done_spec)
# print("observation_spec:", env.observation_spec)
# print("action_keys:", env.action_keys)
# print("reward_keys:", env.reward_keys)
# print("observation_keys:", env.observation_keys)
# print("done_keys:", env.done_keys)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)
# check_env_specs(env)

num_cells = 256
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Unflatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], env.n_agents, -1)
policy_net = nn.Sequential(
    Flatten(), # flatten input from [batch_size, n_agents, observation_dim] to [batch_size, observation_dim * n_agents]
    nn.Linear(env.observation_spec["agents", "observation"].shape[-1] * env.n_agents, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, 2 * env.full_action_spec[env.action_key].shape[-1] * env.n_agents, device=device),
    Unflatten(), # unflatten to [batch_size, n_agents, observation_dim]
    NormalParamExtractor(),
)
policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec_unbatched,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.full_action_spec_unbatched[env.action_key].space.low,
        "high": env.full_action_spec_unbatched[env.action_key].space.high,
    },
    return_log_prob=True,
)  # we'll need the log-prob for the PPO loss

global_critic_net = nn.Sequential(
    Flatten(),  # [batch, n_agents, obs_dim] -> [batch, obs_dim * n_agents]
    nn.Linear(env.observation_spec["agents", "observation"].shape[-1] * env.n_agents, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, 1, device=device),  # Single global value
)
class GlobalValueBroadcast(nn.Module): # Custom module to broadcast the single value to all agents
    def __init__(self, base_net, n_agents):
        super().__init__()
        self.base_net = base_net
    def forward(self, obs):
        # obs shape: [batch, n_agents, obs_dim]
        global_value = self.base_net(obs)  # [batch, 1]
        # Broadcast to all agents for compatibility with multi-agent framework
        return global_value.expand(-1, env.n_agents)  # [batch, n_agents]

critic_net_with_broadcast = GlobalValueBroadcast(global_critic_net, env.n_agents)

critic = ValueOperator(
    module=critic_net_with_broadcast,
    in_keys=[("agents", "observation")],
    out_keys=[("state_value")],
)
# print("Running policy:", policy(env.reset()))
# print("Running value:", critic(env.reset()))
# print(env.reward_key)


collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)
loss_module.set_keys(  # We have to tell the loss where to find the keys; since it's multi-agent, it looks like ('agents', '...')
    reward=env.reward_key,
    action=env.action_key,
    value=("state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

# loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda, average_gae=True)  # We build GAE
# GAE = loss_module.value_estimator
# Update advantage module to use the summed rewards
advantage_module = value.GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=critic,
    average_gae=True
)
advantage_module.set_keys(
    reward=("next", "team_reward"),  # Use the summed rewards
    value=("state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

optim = torch.optim.Adam(loss_module.parameters(), lr)

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    ) # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
    
    # Sum rewards across agents and store as 'team_reward'
    # Compute team reward (mean across agents) and store it in next
    team_reward = tensordict_data.get(("next", "agents", "reward")).mean(dim=-2, keepdim=True)
    tensordict_data.set(("next", "team_reward"), team_reward)

    advantage_module(tensordict_data)
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):        
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.savefig("andong_idea/results/teacher.png")

def render_callback(env, *_):
    frames.append(env.render(mode='rgb_array'))

frames = []
with torch.no_grad():
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=render_callback,
        auto_cast_to_device=True,
        break_when_any_done=False,
    )
# Save as video using imageio
import imageio
imageio.mimsave('andong_idea/results/teacher_rollout.mp4', frames, fps=30, macro_block_size=1)