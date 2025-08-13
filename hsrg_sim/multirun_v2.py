# Torch
import torch

# Tensordict modules
from tensordict import TensorDict
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv, ParallelEnv, EnvCreator
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

# Utils
from matplotlib import pyplot as plt
from tqdm import tqdm

from multisim_zoo import MultiRobotParallelEnv, Helper

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(3)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
#data_device = torch.device("cpu")  # Use CPU for data collection

config = Helper.read_yaml_config("hsrg_sim/setup1.yaml")

# Sampling
frames_per_batch = config['train_parameters']['frames_per_batch']
n_iters = config['train_parameters']['n_iters']
total_frames = frames_per_batch * n_iters
num_envs = config['train_parameters']['num_envs']

# Training
num_epochs = config['train_parameters']['num_epochs']
minibatch_size = config['train_parameters']['minibatch_size']
lr = config['train_parameters']['lr']
max_grad_norm = 1.0

# PPO
clip_epsilon = config['train_parameters']['clip_epsilon']
gamma = config['train_parameters']['gamma']
lmbda = config['train_parameters']['lmbda']
entropy_eps = config['train_parameters']['entropy_eps']

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

max_steps = config['train_parameters']['max_steps']
#num_vec_envs = frames_per_batch // max_steps

# Policy parameter sharing for same type agents
policy_share_params = config['train_parameters']['policy_share_params']
# Use MAPPO (centralized critic in each type of agent)
mappo = True

rng_seed = config['train_parameters']['rng_seed']
eval_seed = config['train_parameters']['eval_seed']

if frames_per_batch // num_envs > max_steps:
    raise ValueError(
        f"Frames_per_batch divided by num_envs must be greater than max_steps, or each environment will run more than one rollout, which cannot be handled by GAE properly."
    )

# ==============================================================================
# HETEROGENEOUS AGENT SETUP
# ==============================================================================

# Import custom environment
#env = MultiRobotParallelEnv(seed=rng_seed, max_steps=max_steps)

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

# Create group mapping
group_map = None
def make_env(worker_id):
    def _make():
        env_seed = rng_seed + worker_id
        single_env = MultiRobotParallelEnv(seed=env_seed, max_steps=max_steps)
        global group_map
        if group_map is None:
            group_map = create_heterogeneous_group_map(single_env)
        wrapped_env = PettingZooWrapper(env=single_env, group_map=group_map)
        return wrapped_env
    return _make

# Need to put the main code in __main__ to avoid issues with multiprocessing (main calls itself)
if __name__ == "__main__":

    make_env_lst = [make_env(worker) for worker in range(num_envs)]

    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_env_lst,
        shared_memory=False,
    )

    # Wrap with PettingZoo using heterogeneous group map
    #env = PettingZooWrapper(env=env, group_map=group_map)

    # FIX: Handle multiple reward keys for heterogeneous groups
    # print("Available reward keys:", env.full_reward_spec)

    # Create RewardSum transform for each group's reward
    reward_transform1 = RewardSum(in_keys=[("uav", "reward")], out_keys=[("uav", "episode_reward")])
    reward_transform2 = RewardSum(in_keys=[("ugv", "reward")], out_keys=[("ugv", "episode_reward")])

    # Apply all reward transforms
    env = TransformedEnv(env, reward_transform1)
    env = TransformedEnv(env, reward_transform2)

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
        n_agents_in_group = obs_shape[1]  # Second dimension is number of agents
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

    # Create combined policy and critic
    def HeterogeneousPolicy(policies):
        """Combines multiple policies into a single module."""
        return TensorDictSequential(*[policies[group_name] for group_name in policies.keys()])

    def HeterogeneousCritic(critics):
        """Combines multiple critics into a single module."""
        return TensorDictSequential(*[critics[group_name] for group_name in critics.keys()])

    # Create combined modules
    combined_policy = HeterogeneousPolicy(policies)
    combined_critic = HeterogeneousCritic(critics)

    # print("\nTesting heterogeneous networks:")
    reset_data = env.reset().to(device)

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

    def actor_critic(tensordict): # Use this instead of combined_policy to use value modules too
        tensordict = combined_policy(tensordict)
        tensordict = combined_critic(tensordict)
        return tensordict

    collector = SyncDataCollector(
        env,
        actor_critic,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device='cpu'),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Create separate loss modules for each group
    loss_modules = {}
    optimizers = {}
    advantage_modules = {}

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
            done=(group_name, "done"),
            terminated=(group_name, "terminated"),
            advantage=(group_name, "advantage"),
            value_target=(group_name, "value_target"),
            value=(group_name, "state_value"),
            sample_log_prob=(group_name, "action_log_prob"),
        )
        
        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=gamma, 
            lmbda=lmbda,
            average_gae=True,  # Average GAE across agents in this group
            time_dim=1,  # Time dimension is the first one in the tensordict
        )
    
        # Separate optimizer for each group
        optimizer = torch.optim.Adam(loss_module.parameters(), lr)
        
        loss_modules[group_name] = loss_module
        optimizers[group_name] = optimizer

    # ==============================================================================
    # TRAINING LOOP WITH HETEROGENEOUS AGENTS
    # ==============================================================================
    
    def split_tensordict(original_td, agent_type):
        """Split TensorDict with UAV/UGV data into separate agent TensorDict."""
        
        # Create the next TensorDict using only keyword arguments
        next_dict = TensorDict(
            batch_size=original_td["next"].batch_size,
            device=original_td["next"].device,
            is_shared=original_td["next"].is_shared
        )
        next_dict["done"] = original_td["next"]["done"]
        next_dict["terminated"] = original_td["next"]["terminated"]
        next_dict["truncated"] = original_td["next"]["truncated"]
        next_dict[agent_type] = original_td["next"][agent_type]
        
        # Create the main TensorDict using only keyword arguments
        agent_td = TensorDict(
            batch_size=original_td.batch_size,
            device=original_td.device,
            is_shared=original_td.is_shared
        )
        agent_td["collector"] = original_td["collector"]
        agent_td["done"] = original_td["done"]
        agent_td["terminated"] = original_td["terminated"]
        agent_td["truncated"] = original_td["truncated"]
        agent_td[agent_type] = original_td[agent_type]
        agent_td["next"] = next_dict
        
        return agent_td

    def train_heterogeneous_agents():
        pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
        episode_reward_mean_list = []
        
        for tensordict_data in collector:
            # print(tensordict_data)
            # print('-'*80)
            tensordict_split_dict = {}
            tensordict_split_dict['uav'] = split_tensordict(tensordict_data, 'uav')
            tensordict_split_dict['ugv'] = split_tensordict(tensordict_data, 'ugv')
            # print(tensordict_split_dict['uav'])
            # print('-'*80)
            # print(tensordict_split_dict['ugv'])
            # print('-'*80)
            # Prepare data for each group
            for group_name in group_map.keys():
            #     # Expand done and terminated for this group
                if ("next", group_name, "state_value") not in tensordict_data.keys(True):
                    tensordict_data.set(
                        ("next", group_name, "state_value"),
                        tensordict_data.get((group_name, "state_value"))
                    )
                # if ("next", group_name, "terminated") not in tensordict_data.keys(True):
                #     tensordict_data.set(
                #         ("next", group_name, "terminated"),
                #         tensordict_data.get(("next", "terminated"))
                #         .unsqueeze(-1)
                #         .expand(tensordict_data.get_item_shape(("next", group_name, "reward"))),
                #     )
                
                with torch.no_grad():
                    GAE = loss_modules[group_name].value_estimator
                    GAE(
                        tensordict_split_dict[group_name],
                        params=loss_modules[group_name].critic_network_params,
                        target_params=loss_modules[group_name].target_critic_network_params,
                    )  # Compute GAE and add it to the data
                
                tensordict_split_dict[group_name] = tensordict_split_dict[group_name].reshape(-1)  # Flatten the batch size [n_envs, t_steps] -> [n_frames] to shuffle data
            
            data_view = TensorDict(uav=tensordict_split_dict['uav'], ugv=tensordict_split_dict['ugv'], batch_size=[frames_per_batch])
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
                        group_subdata = subdata[group_name] #.select(group_name)
                        # group_data[(group_name, "reward")] = group_data[("next", group_name, "reward")]
                        print(group_subdata)
                        # print("Expected reward key:", loss_module.value_estimator.reward_key)
   
                        
                        if group_subdata.numel() > 0:  # Check if group has data
                            loss_vals = loss_module(group_subdata)  # Pass full data, loss module will extract what it needs
                            
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
    env_test = MultiRobotParallelEnv(seed=eval_seed, max_steps=max_steps)
    #group_map_test = create_heterogeneous_group_map(env_test)
    env_test = PettingZooWrapper(env=env_test, group_map=group_map)
    env_test.reset()
    
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
    
