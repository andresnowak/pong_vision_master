import gymnasium as gym

# import ale_py # Often not explicitly needed if gymnasium[atari] is installed
import torch
import numpy as np
import os
import yaml
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecTransposeImage

from stable_baselines3.common.evaluation import evaluate_policy
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import time

from src.config_loader import load_config
from src.create_env import make_full_vision_env


NUM_EPISODES = 1  # Number of games to play
RENDER_DELAY = 0.05  # Delay between frames (in seconds) for better visualization
N_STACK = 4
SEED=42
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


gym.register_envs(ale_py)


def load_model(model_path: str, env):
    # Load the trained model
    model = DQN.load(model_path, env=env, device=device)
    return model


def watch_agent_play(env_id: str, model_path: str):
    # Create environment

    if "-ram" in env_id:
        print("osidjfosidf")
        env = gym.make(env_id, render_mode="human")
        env = DummyVecEnv([lambda: env])  # Wrap in vectorized environment
        env = VecFrameStack(env, n_stack=N_STACK)
    else:
        env = make_atari_env(
            env_id=env_id,
            n_envs=1,
            seed=SEED,
            env_kwargs={"render_mode": "human"},
            # wrapper_kwargs={"frame_stack": N_STACK},
        )
        env = VecTransposeImage(env)

    # Load the trained model
    model = load_model(model_path, env)

    for episode in range(1, NUM_EPISODES + 1):
        # obs, _ = env.reset()
        obs = env.reset()
        done = False
        total_reward = 0
        frames = 0

        while not done:
            # Show the game screen
            env.render()

            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)

            # Take the action
            # obs, reward, done, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)
            # done = done or truncated

            total_reward += reward
            frames += 1

            # Add small delay to make the game watchable
            time.sleep(RENDER_DELAY)

        print(f"Episode {episode}: Total reward: {total_reward}, Frames: {frames}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DQN model")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Name of the environment to visualize on like (PongNoFrameskip-v4)",
    )
    args = parser.parse_args()

    watch_agent_play(args.env, args.model)