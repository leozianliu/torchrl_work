from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    ParallelEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
# from torchrl.record import CSVLogger, VideoRecorder

# Hyperparameters
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 1000
total_frames = 1_000_000  # For a complete training, bring the number of frames up to 1M
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = 0.2  # clip value for PPO loss: see the equation in the intro for more context.
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# Parallel environment setup
num_envs = 5  # Number of parallel environments
env_name = "InvertedDoublePendulum-v4"
env_device = torch.device("cpu")  # Device for the environment

# Function to create a single environment with transforms
def make_env(base_env=None, norm_loc=None, norm_scale=None):
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"], loc=norm_loc, scale=norm_scale),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    return env


# Need to put the main code in a function to avoid issues with multiprocessing (main calls itself)
if __name__ == "__main__":
    # Check multiprocessing start method and set device accordingly
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    if torch.cuda.is_available() and not is_fork:
        print("Using GPU")
    else:
        print("Using CPU")


    # First, create a single environment to get normalization stats
    base_env = GymEnv(env_name, device="cpu")
    single_env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    # Initialize normalization stats on the single environment
    single_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # Get the normalization parameters
    norm_loc = single_env.transform[0].loc.clone()
    norm_scale = single_env.transform[0].scale.clone()

    # # Delete the single environment
    # single_env.close()

    # Create parallel environment
    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_env,
        create_env_kwargs={"base_env":base_env, "norm_loc":norm_loc, "norm_scale":norm_scale},
        shared_memory=False,
        # pin_memory=False,
    )

    env.reset()

    # print("normalization constant shape:", env.get_env_transform(0)[0].loc.shape)
    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)

    # Check environment specs
    check_env_specs(env)

    # Create actor network
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    # Create policy module
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low,
            "high": env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )

    # Create value network
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    # Initialize networks
    policy_module(env.reset().to(device))
    value_module(env.reset().to(device))

    # Create data collector with parallel environment
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    # Create replay buffer (only used for sampling, still on-policy for PPO)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Create advantage module
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Create optimizer and scheduler
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    # Training loop
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        # Training loop over epochs
        for epoch in range(num_epochs):
            # Compute advantage
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            # Mini-batch training
            for batch in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size).to(device) # Sample from replay buffer and move to GPU
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                # Optimization step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        # Log training metrics
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        
        # Evaluation every 10 iterations
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                single_env.to(device) # Send single_env to GPU
                eval_rollout = single_env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        
        #pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        scheduler.step()

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("Training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Total return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

    # Uncomment for video recording
    # path = "./training_loop"
    # logger = CSVLogger(exp_name="ppo", log_dir=path, video_format="mp4")
    # video_recorder = VideoRecorder(logger, tag="video")
    # record_env = TransformedEnv(GymEnv("InvertedDoublePendulum-v4", from_pixels=True, pixels_only=False), video_recorder)
    # record_env.rollout(max_steps=1000, policy=policy_module)
    # video_recorder.dump()