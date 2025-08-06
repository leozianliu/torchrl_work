# Torch
import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from matplotlib import pyplot as plt
from tqdm import tqdm

from multisim_zoo import MultiRobotParallelEnv, Helper

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
data_device = torch.device("cpu")  # Use CPU for data collection

config = Helper.read_yaml_config("hsrg_sim/setup1.yaml")

# Sampling
frames_per_batch = config['train_parameters']['frames_per_batch']
n_iters = config['train_parameters']['n_iters']
total_frames = frames_per_batch * n_iters

# Training
num_epochs = config['train_parameters']['num_epochs']
minibatch_size = config['train_parameters']['minibatch_size']
lr = config['train_parameters']['lr']
max_grad_norm = 1.0

# PPO
clip_epsilon = config['train_parameters']['clip_epsilon']
gamma = config['train_parameters']['clip_epsilon']
lmbda = config['train_parameters']['lmbda']
entropy_eps = config['train_parameters']['entropy_eps']
print(type(entropy_eps))
print('-------------------------')

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

max_steps = config['train_parameters']['max_steps']
#num_vec_envs = frames_per_batch // max_steps

# Policy parameter sharing for same type agents
policy_share_params = config['train_parameters']['policy_share_params']
# Use MAPPO (centralized critic in each type of agent)
mappo = True

reset_options = {"seed_obstacle": 420, "seed_position": 240}

# ==============================================================================
# HETEROGENEOUS AGENT SETUP
# ==============================================================================

# Import custom environment
env = MultiRobotParallelEnv(max_steps=max_steps)

# CREATE HETEROGENEOUS GROUPS
# Method 1: Group by robot type (UAV vs UGV)
def create_heterogeneous_group_map(env):
    """Create group map based on robot types"""
    uav_agents = []
    ugv_agents = []
    
    # Assuming you can access robot configs from env
    for i, robot_config in enumerate(env.robot_configs):
        agent_name = f"robot_{i}"
        if robot_config['type'] == 'UAV':
            uav_agents.append(agent_name)
        elif robot_config['type'] == 'UGV':
            ugv_agents.append(agent_name)
    
    group_map = {}
    if uav_agents:
        group_map["uav"] = uav_agents
    if ugv_agents:
        group_map["ugv"] = ugv_agents
        
    return group_map

# Choose your grouping method
group_map = create_heterogeneous_group_map(env)
print(f"Heterogeneous group map: {group_map}")

# Wrap with PettingZoo using heterogeneous group map
env = PettingZooWrapper(env=env, group_map=group_map)

# FIX: Handle multiple reward keys for heterogeneous groups
# print("Available reward keys:", env.full_reward_spec)

# Create RewardSum transform for each group's reward
reward_transforms = []
for group_name in group_map.keys():
    reward_key = (group_name, "reward")
    episode_reward_key = (group_name, "episode_reward")
    reward_transforms.append(
        RewardSum(in_keys=[reward_key], out_keys=[episode_reward_key])
    )

# Apply all reward transforms
env = TransformedEnv(env, *reward_transforms)

# print("Environment specs after grouping:")
# print(f"Observation spec: {env.observation_spec}")
# print(f"Action spec: {env.full_action_spec}")
# print(f"Reward spec: {env.full_reward_spec}")
# print(f"Group keys: {list(env.observation_spec.keys())}")

# ==============================================================================
# HETEROGENEOUS NETWORKS SETUP
# ==============================================================================

# Create separate networks for each agent type
networks = {}
policies = {}
critics = {}

for group_name in group_map.keys():
    # Get specs for this group
    obs_shape = env.observation_spec[group_name, "observation"].shape
    action_shape = env.full_action_spec[group_name, "action"].shape
    n_agents_in_group = obs_shape[0]  # First dimension is number of agents
    obs_per_agent = obs_shape[-1]     # Last dimension is observation per agent
    action_per_agent = action_shape[-1]  # Last dimension is action per agent
    
    print(f"\nGroup '{group_name}':")
    print(f"  Number of agents: {n_agents_in_group}")
    print(f"  Obs per agent: {obs_per_agent}")
    print(f"  Action per agent: {action_per_agent}")
    
    # POLICY NETWORK for this group
    # Key insight: share_params=True means all agents in this group share the SAME parameters
    if group_name == "uav":
        # All UAVs share the same policy network parameters
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_per_agent,
                n_agent_outputs=2 * action_per_agent,  # loc and scale
                n_agents=n_agents_in_group,
                centralised=False,
                share_params=policy_share_params,  # All UAVs use SAME parameters
                device=device,
                depth=2,  # Deeper network for UAVs
                num_cells=256,  # Larger network for UAVs
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        print(f"  UAV policy parameters: {sum(p.numel() for p in policy_net.parameters())}")
    else:  # UGV
        # All UGVs share the same policy network parameters (different from UAV parameters)
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_per_agent,
                n_agent_outputs=2 * action_per_agent,  # loc and scale
                n_agents=n_agents_in_group,
                centralised=False,
                share_params=policy_share_params,  # All UGVs use SAME parameters (but different from UAVs)
                device=device,
                depth=2,  # Shallower network for UGVs
                num_cells=256,  # Smaller network for UGVs
                activation_class=torch.nn.ReLU,  # Different activation
            ),
            NormalParamExtractor(),
        )
        print(f"  UGV policy parameters: {sum(p.numel() for p in policy_net.parameters())}")
    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(group_name, "observation")],
        out_keys=[(group_name, "loc"), (group_name, "scale")],
    )

    # Updated policy creation:
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched[group_name, "action"],  # Group-specific spec
        in_keys=[(group_name, "loc"), (group_name, "scale")],
        out_keys=[(group_name, "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[group_name, "action"].space.low,
            "high": env.full_action_spec_unbatched[group_name, "action"].space.high,
        },
        return_log_prob=True,
    )
    
    # CRITIC NETWORK for this group
    # Key insight: share_params=True means all agents in this group share the SAME critic parameters
    if group_name == "uav":
        # All UAVs share the same critic network parameters
        critic_net = MultiAgentMLP(
            n_agent_inputs=obs_per_agent,
            n_agent_outputs=1,
            n_agents=n_agents_in_group,
            centralised=mappo,
            share_params=True,  # All UAVs use SAME critic parameters
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )
        print(f"  UAV critic parameters: {sum(p.numel() for p in critic_net.parameters())}")
    else:  # UGV
        # All UGVs share the same critic network parameters (different from UAV parameters)
        critic_net = MultiAgentMLP(
            n_agent_inputs=obs_per_agent,
            n_agent_outputs=1,
            n_agents=n_agents_in_group,
            centralised=mappo,
            share_params=True,  # All UGVs use SAME critic parameters (but different from UAVs)
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.ReLU,
        )
        print(f"  UGV critic parameters: {sum(p.numel() for p in critic_net.parameters())}")
    
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[(group_name, "observation")],
        out_keys=[(group_name, "state_value")],
    )
    
    # Store networks
    policies[group_name] = policy
    critics[group_name] = critic

# ==============================================================================
# COMBINE HETEROGENEOUS NETWORKS
# ==============================================================================

# Combine all policies into one module
combined_policy_modules = {}
combined_critic_modules = {}

for group_name in group_map.keys():
    combined_policy_modules[group_name] = policies[group_name]
    combined_critic_modules[group_name] = critics[group_name]

# Create combined policy and critic
from tensordict.nn import TensorDictSequential

# Method 1: Simple combination - each policy acts independently
class HeterogeneousPolicy(torch.nn.Module):
    def __init__(self, policies):
        super().__init__()
        self.policies = torch.nn.ModuleDict(policies)
    
    def forward(self, tensordict):
        out = tensordict.empty()
        for group_name, policy in self.policies.items():
            if (group_name, "observation") in tensordict.keys(True):
                group_output = policy(tensordict.select(group_name))
                out.update(group_output)
        return out

class HeterogeneousCritic(torch.nn.Module):
    def __init__(self, critics):
        super().__init__()
        self.critics = torch.nn.ModuleDict(critics)
    
    def forward(self, tensordict):
        out = tensordict.empty()
        for group_name, critic in self.critics.items():
            if (group_name, "observation") in tensordict.keys(True):
                group_output = critic(tensordict.select(group_name))
                out.update(group_output)
        return out

# Create combined modules
combined_policy = HeterogeneousPolicy(policies)
combined_critic = HeterogeneousCritic(critics)

# print("\nTesting heterogeneous networks:")
reset_data = env.reset(options=reset_options).to(device)
# print("Reset data keys:", reset_data.keys(True))

policy_output = combined_policy(reset_data)
print("Policy output keys:", policy_output.keys(True))

critic_output = combined_critic(reset_data)
print("Critic output keys:", critic_output.keys(True))

# ==============================================================================
# TRAINING SETUP WITH PARAMETER SHARING WITHIN GROUPS
# ==============================================================================

print("\n" + "="*60)
print("PARAMETER SHARING SUMMARY:")
print("="*60)
total_params = 0
for group_name in group_map.keys():
    policy_params = sum(p.numel() for p in policies[group_name].parameters())
    critic_params = sum(p.numel() for p in critics[group_name].parameters()) 
    group_total = policy_params + critic_params
    total_params += group_total
    
    print(f"{group_name.upper()} group:")
    print(f"  - Number of agents: {len(group_map[group_name])}")
    print(f"  - Policy parameters: {policy_params:,} (shared among all {group_name} agents)")
    print(f"  - Critic parameters: {critic_params:,} (shared among all {group_name} agents)")
    print(f"  - Total {group_name} parameters: {group_total:,}")
print(f"\nTOTAL MODEL PARAMETERS: {total_params:,}")
print("="*60)

collector = SyncDataCollector(
    env,
    combined_policy,
    device=data_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

# Create separate loss modules for each group
loss_modules = {}
optimizers = {}

for group_name in group_map.keys():
    loss_module = ClipPPOLoss(
        actor_network=policies[group_name],
        critic_network=critics[group_name],
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,
    )
    
    # Set keys for this specific group
    loss_module.set_keys(
        reward=(group_name, "reward"),
        action=(group_name, "action"),
        value=(group_name, "state_value"),
        done=(group_name, "done"),
        terminated=(group_name, "terminated"),
    )
    
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    
    # Separate optimizer for each group
    optimizer = torch.optim.Adam(loss_module.parameters(), lr)
    
    loss_modules[group_name] = loss_module
    optimizers[group_name] = optimizer

print(f"\nCreated {len(loss_modules)} separate loss modules for groups: {list(loss_modules.keys())}")

# ==============================================================================
# TRAINING LOOP WITH HETEROGENEOUS AGENTS
# ==============================================================================

def train_heterogeneous_agents():
    pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
    episode_reward_mean_list = []
    
    for tensordict_data in collector:
        # Prepare data for each group
        for group_name in group_map.keys():
            # Expand done and terminated for this group
            if ("next", group_name, "done") not in tensordict_data.keys(True):
                tensordict_data.set(
                    ("next", group_name, "done"),
                    tensordict_data.get(("next", "done"))
                    .unsqueeze(-1)
                    .expand(tensordict_data.get_item_shape(("next", group_name, "reward"))),
                )
            if ("next", group_name, "terminated") not in tensordict_data.keys(True):
                tensordict_data.set(
                    ("next", group_name, "terminated"),
                    tensordict_data.get(("next", "terminated"))
                    .unsqueeze(-1)
                    .expand(tensordict_data.get_item_shape(("next", group_name, "reward"))),
                )
        
        # Compute GAE for each group
        for group_name in group_map.keys():
            loss_module = loss_modules[group_name]
            with torch.no_grad():
                loss_module.value_estimator(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )
        
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        
        # Training epochs
        for epoch in range(num_epochs):
            for batch in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample().to(device)
                
                # Train each group separately (but parameters are shared within each group)
                total_loss = 0
                for group_name in group_map.keys():
                    loss_module = loss_modules[group_name]
                    optimizer = optimizers[group_name]
                    
                    # Extract data for this group
                    group_data = subdata.select(group_name)
                    
                    if group_data.numel() > 0:  # Check if group has data
                        loss_vals = loss_module(subdata)  # Pass full data, loss module will extract what it needs
                        
                        loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"] 
                            + loss_vals["loss_entropy"]
                        )
                        
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                        
                        optimizer.step()  # Updates shared parameters for this group
                        optimizer.zero_grad()
                        
                        total_loss += loss_value.item()
                        
                        # Print parameter sharing info (first iteration only)
                        if epoch == 0 and batch == 0 and not hasattr(loss_module, '_first_update'):
                            n_agents_in_group = len(group_map[group_name])
                            print(f"    {group_name} group: {n_agents_in_group} agents sharing {sum(p.numel() for p in loss_module.parameters()):,} parameters")
                            setattr(loss_module, '_first_update', True)
        
        collector.update_policy_weights_()
        
        # Logging - aggregate rewards across all groups
        episode_rewards = []
        for group_name in group_map.keys():
            episode_reward_key = (group_name, "episode_reward")
            done_key = ("next", group_name, "done")
            
            if done_key in tensordict_data.keys(True) and episode_reward_key in tensordict_data.keys(True):
                done = tensordict_data.get(done_key)
                group_rewards = tensordict_data.get(episode_reward_key)[done]
                if group_rewards.numel() > 0:
                    episode_rewards.append(group_rewards.mean().item())
        
        if episode_rewards:
            episode_reward_mean = sum(episode_rewards) / len(episode_rewards)
        else:
            episode_reward_mean = 0.0
            
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.2f}", refresh=False)
        pbar.update()
    
    return episode_reward_mean_list

# Run training
if __name__ == "__main__":
    print("="*80)
    print("HETEROGENEOUS MULTI-AGENT SETUP WITH POLICY PARAMETER SHARING WITHIN GROUPS")
    print("="*80)
    
    print("\n1. PARAMETER SHARING EXPLANATION:")
    print("   - All UAV agents share the SAME neural network parameters")
    print("   - All UGV agents share the SAME neural network parameters") 
    
    print("\n2. Starting heterogeneous multi-agent training...")
    episode_rewards = train_heterogeneous_agents()
    
    print("\n3. TRAINING COMPLETED")
    print("="*80)
    
    # Demonstrate parameter sharing
    print("\n4. PARAMETER SHARING VERIFICATION:")
    env_test = MultiRobotParallelEnv(max_steps=max_steps)
    #group_map_test = create_heterogeneous_group_map(env_test)
    env_test = PettingZooWrapper(env=env_test, group_map=group_map)
    env_test.reset(options={"seed_obstacle": 42, "seed_position": 100})
    
    # for group_name, agent_list in group_map.items():
    #     print(f"\n{group_name.upper()} group agents: {agent_list}")
    #     if len(agent_list) > 1:
    #         print(f"  All {len(agent_list)} {group_name} agents use the SAME {sum(p.numel() for p in policies[group_name].parameters()):,} policy parameters")
    #         print(f"  All {len(agent_list)} {group_name} agents use the SAME {sum(p.numel() for p in critics[group_name].parameters()):,} critic parameters")
    #     else:
    #         print(f"  Single {group_name} agent uses {sum(p.numel() for p in policies[group_name].parameters()):,} policy parameters")
    #         print(f"  Single {group_name} agent uses {sum(p.numel() for p in critics[group_name].parameters()):,} critic parameters")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel("Training iterations")
    plt.ylabel("Average Episode Reward")
    plt.title("Heterogeneous Multi-Agent Training with Parameter Sharing")
    plt.grid(True)
    plt.show()
    
    # Render final policy
    def render_callback(env, *_):
        frames.append(env.render(render_mode='human'))
    
    env_test.start_video_recording('hsrg_sim/marl_simulation.mp4')
    frames = []
    print("\n5. Rendering final policy...")
    with torch.no_grad():
        env_test.rollout(
            max_steps=100,
            policy=combined_policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
    env_test.stop_video_recording()
    env_test.close()
    
