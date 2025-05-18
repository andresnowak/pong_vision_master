import gymnasium as gym
import torch
import numpy as np
import argparse
from itertools import product
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import cv2
import time
from active_gym.atari_env import AtariFixedFovealPeripheralEnv, AtariEnvArgs

from src.config_loader import load_config
from src.dqn import QNetwork
from src.pvm_buffer import PVMBuffer

NUM_EPISODES = 1
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
device = "cpu"
print(f"Using device: {device}")


def load_model(model_path: str, env: gym.Env, sensory_action_set):
    vision_model = QNetwork(env, sensory_action_set=sensory_action_set).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    vision_model.load_state_dict(checkpoint["q"])
    vision_model.eval()

    return vision_model


def visualization(env, obs, fovea_loc, fov_size, fig=None, axs=None):
    frame = obs[0]  # First channel of the observation
    full_frame = env.render()

    # Create figure if it doesn't exist
    if fig is None or axs is None:
        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.show()

    # Clear previous content
    for ax in axs:
        ax.clear()

    # Scale coordinates from observation space to render space
    render_h, render_w = full_frame.shape[:2]
    scale_x = render_w / 84
    scale_y = render_h / 84

    x, y = fovea_loc
    x_scaled = int(x * scale_x)
    y_scaled = int(y * scale_y)
    w_scaled = int(fov_size * scale_x)
    h_scaled = int(fov_size * scale_y)

    frame_with_box = full_frame.copy()
    cv2.rectangle(
        frame_with_box,
        (x_scaled, y_scaled),
        (x_scaled + w_scaled, y_scaled + h_scaled),
        (255, 0, 0),
        2,
    )

    axs[0].imshow(frame_with_box)
    axs[0].set_title("Full Environment Frame")
    axs[0].axis("off")

    axs[1].imshow(frame, cmap="gray")
    axs[1].set_title("Foveated Observation")
    axs[1].axis("off")

    plt.pause(0.001)  # Reduced pause time for smoother updates
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, axs  # Return the figure and axes for reuse


def watch_agent_play(env, vector_env, model_path: str, config):
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

    vision_model = load_model(model_path, vector_env, sensory_action_set)

    # Fix: Correct buffer shape for vectorized env

    resize = Resize(OBSERVATION_SIZE)

    # Initialize visualization
    plt.ion()
    fig, axs = None, None
    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        frames = 0

        while not done:
            input_obs = (
                resize(torch.from_numpy(obs))
                .unsqueeze(0)
                .float()
            )
            # Fix: Handle vectorized observations properly
            # Predict actions
            with torch.no_grad():
                sensory_q = vision_model(input_obs)
                actions = vector_env.single_action_space.sample()
                motor_action = np.array([actions["motor_action"]])
                sensory_action_idx = sensory_q.argmax().item()

            # Lookup the 2D fovea position
            fovea_loc = sensory_action_set[sensory_action_idx]

            action = {
                "motor_action": motor_action[0],
                "sensory_action": fovea_loc
            }
    
            obs, reward, done, truncated, info = env.step(action)
            # done = terminated or truncated
            total_reward += reward
            frames += 1

            # Update visualization
            fig, axs = visualization(env, obs, fovea_loc, fov_size, fig, axs)

            time.sleep(RENDER_DELAY)

        print(f"Episode {episode}: Total reward: {total_reward}, Frames: {frames}")

    plt.ioff()  # Turn off interactive mode when done
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DQN model")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn_fovea_peripheral_pong_vision_only_config.yml",
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    env_args = AtariEnvArgs(
        config["environment"]["env_id"],
        20,
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
        peripheral_res=(config["environment"]["peripheral_res"], config["environment"]["peripheral_res"]),
        resize_to_full=True,
        clip_reward=config["environment"]["clip_reward"],
        mask_out=True,
    )
        
    env = AtariFixedFovealPeripheralEnv(env_args)
    print(dir(env.ale))
    # env.ale.setMode(1)
    # env.ale.setDifficulty(1)

    def reset_with_random_ball(env):
        obs = env.reset()

        if hasattr(env, "ale"):
            # Save initial state
            initial_state = env.ale.cloneState()

            # Advance 1 frame with a random action to randomize direction
            env.step(env.action_space.sample())

            # Save this randomized state
            randomized_state = env.ale.cloneState()

            # Restore initial state but keep RNG seed
            env.ale.restoreState(initial_state)

    vector_env = gym.vector.SyncVectorEnv([lambda: env])
    

    # Create environment
    watch_agent_play(env, vector_env, args.model, config)
