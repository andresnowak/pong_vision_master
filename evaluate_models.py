import gymnasium as gym
import torch
import numpy as np
import argparse
from itertools import product
from torchvision.transforms import Resize
import cv2
from active_gym.atari_env import AtariEnvArgs, AtariFixedFovealPeripheralEnv, AtariFixedFovealEnv
import torch.nn.utils.prune as prune
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import json


from src.config_loader import load_config
from src.dqn_efficient import QNetwork as LCAQNetwork
from src.dqn import QNetwork
from src.dqn_sugarl import QNetwork as SugaRLQNetwork
from src.pvm_buffer import PVMBuffer
from src.extra_envs import CVAtariFixedFovealEnv, CVAtariFixedFovealPeripheralEnv
from src.utils import set_global_seed

# Evaluation parameters
NUM_EPISODES = 30  # Changed from 1 to 100
N_STACK = 4
SEED = 42

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"
print(f"Using device: {device}")


def load_model(model_path: str, env: gym.Env, sensory_action_set, model_type: str):
    # Load the saved state dictionary

    # Create and load the dense model
    if model_type == "dqn_fovea_motor_separate_pong_lca":
        state_dict = torch.load(model_path, map_location=device)["motor_q"]
        model_dense = LCAQNetwork(env).to(device)
    elif model_type == "dqn_fovea_motor_separate_pong": # this one is without peripheral in reality
        state_dict = torch.load(model_path, map_location=device)["motor_q"]
        model_dense = QNetwork(env).to(device)
    elif model_type == "dqn_fovea_peripheral_motor_separate_pong":
        state_dict = torch.load(model_path, map_location=device)["motor_q"]
        model_dense = QNetwork(env).to(device)
    elif model_type == "dqn_fovea_peripheral_pong":
        state_dict = torch.load(model_path, map_location=device)["q"]
        model_dense = SugaRLQNetwork(env, sensory_action_set=sensory_action_set).to(
            device
        )
    elif model_type == "dqn_fovea_pong":
        state_dict = torch.load(model_path, map_location=device)["q"]
        model_dense = SugaRLQNetwork(env, sensory_action_set=sensory_action_set).to(
            device
        )
    else:
        raise NameError
    model_dense.load_state_dict(state_dict)
    model_dense.eval()
    compute_sparsity(model_dense)

    return model_dense


def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    if total_params > 0:
        sparsity = zero_params / total_params
        print(f"Model sparsity: {zero_params}/{total_params} ({sparsity:.1%})")
    else:
        print("No weight parameters found")
    return sparsity


def load_env(config):
    env_args = AtariEnvArgs(
        config["environment"]["env_id"],
        20,
        obs_size=(84, 84),
        peripheral_res=config["environment"].get("peripheral_res", 30), # we put 30 as default, but if the env doesn't use it won't be used
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

    if config["exp_name"] in ["dqn_fovea_motor_separate_pong_lca", "dqn_fovea_motor_separate_pong"] :
        env = CVAtariFixedFovealEnv(env_args)
        vector_env = gym.vector.SyncVectorEnv([lambda: env])
    elif config["exp_name"] in ["dqn_fovea_peripheral_motor_separate_pong"]:
        env = CVAtariFixedFovealPeripheralEnv(env_args)
        vector_env = gym.vector.SyncVectorEnv([lambda: env])
    elif config["exp_name"] in ["dqn_fovea_peripheral_pong"]:
        env = AtariFixedFovealPeripheralEnv(env_args)
        vector_env = gym.vector.SyncVectorEnv([lambda: env])
    elif config["exp_name"] in ["dqn_fovea_pong"]:
        env = AtariFixedFovealEnv(env_args)
        vector_env = gym.vector.SyncVectorEnv([lambda: env])
    else:
        raise NameError

    return env, vector_env

def evaluate_agent(env, vector_env, model_path: str, config, model_type: str):
    """Evaluate agent for NUM_EPISODES and return average reward"""
    OBSERVATION_SIZE = (84, 84)
    fov_size = config["environment"]["fov_size"]
    observ_x_max = OBSERVATION_SIZE[0] - fov_size
    observ_y_max = OBSERVATION_SIZE[1] - fov_size

    # Setup sensory action space
    sensory_action_step = (
        observ_x_max // config["environment"]["sensory_action_x_size"],
        observ_y_max // config["environment"]["sensory_action_y_size"],
    )
    sensory_action_x_set = list(range(0, observ_x_max, sensory_action_step[0]))[
        : config["environment"]["sensory_action_x_size"]
    ]
    sensory_action_y_set = list(range(0, observ_y_max, sensory_action_step[1]))[
        : config["environment"]["sensory_action_y_size"]
    ]
    sensory_action_set = [
        np.array(a) for a in list(product(sensory_action_x_set, sensory_action_y_set))
    ]

    def evaluate(motor_model, model_type: str = "normal"):
        # Setup buffer
        pvm_buffer = PVMBuffer(
            config["environment"]["pvm_stack"],
            (config["environment"]["frame_stack"],) + OBSERVATION_SIZE,
        )
        resize = Resize(OBSERVATION_SIZE)

        # Track statistics
        episode_rewards = []
        episode_lengths = []

        print(f"Starting evaluation for {model_type} for {NUM_EPISODES} episodes...")

        for episode in range(1, NUM_EPISODES + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            frames = 0

            while not done:
                pvm_buffer.append(obs)
                input_obs = (
                    resize(torch.from_numpy(pvm_buffer.get_obs(mode="stack_max")))
                    .unsqueeze(0)
                    .float()
                )

                # Predict actions
                with torch.no_grad():
                    if model_type in ["dqn_fovea_peripheral_pong", "dqn_fovea_pong"]:
                        motor_q, sensory_q = motor_model(input_obs.to(device))
                        motor_action = motor_q.argmax().item()
                        sensory_action_idx = sensory_q.argmax().item()
                        # Lookup the 2D fovea position
                        sensory_action = sensory_action_set[sensory_action_idx]
                    else:
                        motor_q = motor_model(input_obs.to(device))
                        motor_action = motor_q.argmax().item()
                        sensory_action = [54]

                # Lookup the 2D fovea position
                action = {"motor_action": motor_action, "sensory_action": sensory_action}

                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                frames += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(frames)

            # Print progress every 10 episodes
            if episode % 10 == 0:
                current_avg = np.mean(episode_rewards)
                print(
                    f"Episode {episode}/{NUM_EPISODES}: Current average reward: {current_avg:.2f}"
                )

        # Calculate final statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        avg_length = np.mean(episode_lengths)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Episodes: {NUM_EPISODES}")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min Reward: {min_reward}")
        print(f"Max Reward: {max_reward}")
        print(f"Average Episode Length: {avg_length:.1f} frames")
        print("=" * 50)

        # Save results to file
        results = {
            "fov_size": config["environment"]["fov_size"],
            "peripheral_res": config["environment"].get("peripheral_res", None),
            "model_name": model_type,
            "num_episodes": NUM_EPISODES,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "avg_length": avg_length,
            "num_parameters": sum(param.numel() for param in motor_model.parameters())
        }

        env.close()
        return results

    # Load model
    motor_model = load_model(
        model_path, vector_env, sensory_action_set, model_type
    )
    results = evaluate(motor_model, model_type)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN model performance")
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes to evaluate (default: 100)",
    )

    args = parser.parse_args()

    # Update NUM_EPISODES from command line argument
    NUM_EPISODES = args.episodes

    model_list = [
        "models/dqn_pong_yml/dqn_fovea_pong_motor_separate_lca_best/run_20250521_020159/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_motor_separate_best/run_20250518_202609/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_peripheral_real_perihperal_pong_motor_separate_best/run_20250520_112403/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250519_234526/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_pong_best/run_20250518_202140/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250517_183532/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250509_230000/trained_models/pong_seed1_step10000000_model.pt",
        "models/dqn_pong_yml/dqn_fovea_pong_best/run_20250509_153631/trained_models/pong_seed1_step5000000_model.pt",
    ]
    config_list = [
        "models/dqn_pong_yml/dqn_fovea_pong_motor_separate_lca_best/run_20250521_020159/config.yml",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_motor_separate_best/run_20250518_202609/config.yml",
        "models/dqn_pong_yml/dqn_fovea_peripheral_real_perihperal_pong_motor_separate_best/run_20250520_112403/config.yml",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250519_234526/config.yml",
        "models/dqn_pong_yml/dqn_fovea_pong_best/run_20250518_202140/config.yml",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250517_183532/config.yml",
        "models/dqn_pong_yml/dqn_fovea_peripheral_pong_best/run_20250509_230000/config.yml",
        "models/dqn_pong_yml/dqn_fovea_pong_best/run_20250509_153631/config.yml",
    ]

    all_results = {}
    set_global_seed(42)
    for i, value in enumerate(zip(model_list, config_list)):
        model_path, config_path = value

        config = load_config(config_path)

        # Run evaluation
        env, vector_env = load_env(config)
        results = evaluate_agent(env, vector_env, model_path, config, config["exp_name"])

        all_results[f"{config['exp_name']}_{i}"] = results

    print(all_results)


    # Save to JSON file
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to Python type
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj


    os.makedirs("evaluations", exist_ok=True)
    with open(f'evaluations/evaluation_results.json', 'w') as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)
    print(f"Results saved to evaluation_results.json")


