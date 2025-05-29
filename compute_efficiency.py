"""
Measure a model's inference time
"""

import gymnasium as gym
import torch
import numpy as np
import argparse
from itertools import product
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import time
from active_gym.atari_env import AtariFixedFovealEnv, AtariEnvArgs

from src.utils import visualization
from src.config_loader import load_config
from src.create_env import make_fovea_env
from src.dqn_sugarl import QNetwork
from src.pvm_buffer import PVMBuffer

from torch import nn

N_FRAMES = 200

NUM_EPISODES = 10
RENDER_DELAY = 0.05
N_STACK = 4
SEED = 42

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#device = "cpu"
print(f"Using device: {device}")


def load_model(model_path: str, env: gym.Env, sensory_action_set):
    model = QNetwork(env, sensory_action_set=sensory_action_set).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["q"])
    model.eval()
    return model


def count_parameters(model: nn.Module, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Efficiency")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn_fovea_pong_config.yml",
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    env_args = AtariEnvArgs(
        config["environment"]["env_id"],
        config["seed"],
        obs_size=(84, 84),
        frame_stack=config["environment"]["frame_stack"],
        action_repeat=config["environment"]["action_repeat"],
        fov_size=(
            config["environment"]["fov_size"],
            config["environment"]["fov_size"],
        ),
        fov_init_loc=(
            config["environment"]["fov_init_loc_x"],
            config["environment"]["fov_init_loc_y"],
        ),
        sensory_action_mode=config["environment"]["sensory_action_mode"],
        sensory_action_space=(
            -config["environment"]["sensory_action_space"],
            config["environment"]["sensory_action_space"],
        ),
        resize_to_full=True,
        clip_reward=config["environment"]["clip_reward"],
        mask_out=True,
    )
        
    env = AtariFixedFovealEnv(env_args)

    vector_env = gym.vector.SyncVectorEnv([lambda: env])


    OBSERVATION_SIZE = (84, 84)
    fov_size = config["environment"]["fov_size"]
    observ_x_max = OBSERVATION_SIZE[0] - fov_size
    observ_y_max = OBSERVATION_SIZE[1] - fov_size

    # Fix: Use correct y_size parameter
    sensory_action_step = (
        observ_x_max // config["environment"]["sensory_action_x_size"],
        observ_y_max // config["environment"]["sensory_action_y_size"],
    )
    sensory_action_x_set = list(range(0, observ_x_max, sensory_action_step[0]))[
        : config["environment"]["sensory_action_x_size"]
    ]
    sensory_action_y_set = list(
        range(0, observ_y_max, sensory_action_step[1])
    )[
        : config["environment"][
            "sensory_action_y_size"
        ]  # Fixed typo here
    ]
    sensory_action_set = [
        np.array(a) for a in list(product(sensory_action_x_set, sensory_action_y_set))
    ]

    model_path = args.model
    print(model_path)

    model = load_model(model_path, vector_env, sensory_action_set)

    # Fix: Correct buffer shape for vectorized env
    pvm_buffer = PVMBuffer(
        config["environment"]["pvm_stack"],
        (config["environment"]["frame_stack"],
         ) + OBSERVATION_SIZE,
    )

    resize = Resize(OBSERVATION_SIZE)


    print("Starting episodes")
    frames = 0
    for episode in range(1, NUM_EPISODES + 1):

        obs, _ = env.reset()
        done = False
        total_reward = 0

        total_inference_time = 0.0  # initialize timer

        while not done:

            if frames >= N_FRAMES:
                break

            pvm_buffer.append(obs)
            input_obs = (
                resize(torch.from_numpy(pvm_buffer.get_obs(mode="stack_max")))
                .unsqueeze(0)
                .float()
                .to(device)
            )

            # Inference timing
            start_time = time.perf_counter()
            with torch.no_grad():
                motor_q, sensory_q = model(input_obs)
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_inference_time += inference_time

            motor_action = motor_q.argmax().item()
            sensory_action_idx = sensory_q.argmax().item()
            fovea_loc = sensory_action_set[sensory_action_idx]

            action = {
                "motor_action": motor_action,
                "sensory_action": fovea_loc
            }

            obs, reward, done, truncated, info = env.step(action)
            #print("Step")
            total_reward += reward
            frames += 1

        avg_inference_time = total_inference_time / frames if frames > 0 else None
        print(f"Episode {episode}: Total reward: {total_reward}, Frames: {frames}")
        print(f"Total inference time: {total_inference_time:.4f} sec")
        print(f"Average inference time per frame: {avg_inference_time:.6f} sec")


        print(f"Trainable parameters: {count_parameters(model):,}")
        print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")



    






