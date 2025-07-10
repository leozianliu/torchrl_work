from torchrl.record import CSVLogger
from torchrl.envs import GymEnv
from torchrl.envs import TransformedEnv
from torchrl.record import VideoRecorder
import imageio
import numpy as np

env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)
env_rollout = env.rollout(max_steps=100)
print(env_rollout)

logger = CSVLogger(exp_name="my_exp")


recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)

record_rollout = record_env.rollout(max_steps=200)
frames = [step["pixels"] for step in record_rollout]
imageio.mimsave("rollout.mp4", [np.array(frame) for frame in frames], fps=10)