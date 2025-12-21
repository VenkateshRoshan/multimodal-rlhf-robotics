import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.gym_wrapper import RobotGraspEnv

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create environment
def make_env():
    env = RobotGraspEnv(render=True)
    env = Monitor(env, "logs/")
    return env

env = DummyVecEnv([make_env])

# Custom policy network
policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 512],
        vf=[512, 512]
    ),
    activation_fn=nn.ReLU,
)

# Create PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="logs/tensorboard/",
    policy_kwargs=policy_kwargs,
    device="cuda"
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="checkpoints/",
    name_prefix="ppo_robot"
)

# Train
print("🚀 Starting training...")
model.learn(
    total_timesteps=500000,
    callback=checkpoint_callback,
    progress_bar=True
)

# Save
model.save("checkpoints/ppo_final")
print("✅ Done!")