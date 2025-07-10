import torch
import time
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torch.optim import Adam
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder


# Set up the environment
torch.manual_seed(0)
env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
env.set_seed(0)

# Define the policy
value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(env.action_spec, annealing_num_steps=100_000, eps_init=0.5)
policy_explore = Seq(policy, exploration_module)

# Set up the collector and replay buffer
init_rand_steps = 50000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

# Create the optimizer and loss function
loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.90)

# Log the training process
path = "./training_loop"
logger = CSVLogger(exp_name="dqn", log_dir=path, video_format="mp4")
video_recorder = VideoRecorder(logger, tag="video")
record_env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True, pixels_only=False), video_recorder
)

# Training loop
total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad() # Clears gradients for the next iteration
            exploration_module.step(data.numel()) # Update exploration factor
            updater.step() # Update target params
            if i % 10:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length >= 500:
        break

t1 = time.time()

torchrl_logger.info(f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s.")

# Render the video
record_env.rollout(max_steps=1000, policy=policy)
video_recorder.dump()

