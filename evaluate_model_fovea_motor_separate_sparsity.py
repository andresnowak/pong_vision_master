import gymnasium as gym
import torch
import numpy as np
import argparse
from itertools import product
from torchvision.transforms import Resize
import cv2
from active_gym.atari_env import AtariFixedFovealPeripheralEnv, AtariEnvArgs
import torch.nn.utils.prune as prune
from copy import deepcopy
import matplotlib.pyplot as plt
import os


from src.config_loader import load_config
from src.dqn import QNetwork
from src.pvm_buffer import PVMBuffer
from src.extra_envs import CVAtariFixedFovealEnv
from src.sparse_utils import plot_results_for_paper, save_results
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



def load_model(model_path: str, env: gym.Env, temp, sparse_amount=0.8):
    # Load the saved state dictionary
    state_dict = torch.load(model_path, map_location=device)["motor_q"]

    # Create and load the dense model
    model_dense = QNetwork(env).to(device)
    model_dense.load_state_dict(state_dict)
    model_dense.eval()
    compute_sparsity(model_dense)

    # Create a separate instance for the sparse model
    model_sparse = QNetwork(env).to(device)
    model_sparse.load_state_dict(state_dict)
    model_sparse.eval()
    model_sparse = make_model_sparse(
        model_sparse, sparse_amount
    )  # Apply sparsification
    compute_sparsity(model_sparse)

    return model_dense, model_sparse


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


def make_model_sparse(model, amount=0.8):
    # Apply global unstructured L1 pruning to all Linear weights
    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # Make pruning permanent (remove mask and reparam hooks)
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    return model


def evaluate_agent(env, vector_env, model_path: str, config):
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
                    motor_q = motor_model(input_obs.to(device))
                    motor_action = motor_q.argmax().item()

                # Lookup the 2D fovea position
                action = {"motor_action": motor_action, "sensory_action": [54]}

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
            "num_episodes": NUM_EPISODES,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "avg_length": avg_length,
        }

        # # Optionally save to file
        # import json
        # with open(f'evaluation_results_{model_type}.json', 'w') as f:
        #     json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
        #             for k, v in results.items()}, f, indent=2)
        # print(f"Results saved to evaluation_results_{model_type}.json")

        env.close()

        return results

    sparse_results = []
    # Load model
    motor_model, motor_model_sparse = load_model(
        model_path, vector_env, sensory_action_set
    )
    dense_result = evaluate(motor_model)
    for sparse_amount in [0.2, 0.4, 0.6, 0.8]:
        motor_model, motor_model_sparse = load_model(
        model_path, vector_env, sensory_action_set, sparse_amount
        )
        sparse_result = evaluate(motor_model_sparse, "sparse")
        sparse_results.append(sparse_result)

    plot_results_for_paper(dense_result, sparse_results, "outputs_figures/fovea_motor_separate")
    save_results(dense_result, sparse_results, "outputs_figures/fovea_motor_separate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN model performance")
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes to evaluate (default: 100)",
    )

    args = parser.parse_args()

    # Update NUM_EPISODES from command line argument
    NUM_EPISODES = args.episodes

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
        resize_to_full=True,
        clip_reward=config["environment"]["clip_reward"],
        mask_out=True,
    )

    set_global_seed(42)
    env = CVAtariFixedFovealEnv(env_args)
    vector_env = gym.vector.SyncVectorEnv([lambda: env])

    # Run evaluation
    results = evaluate_agent(env, vector_env, args.model, config)
