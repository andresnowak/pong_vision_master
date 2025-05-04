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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py

from src.config_loader import load_config
from src.create_env import make_no_vision_env
from src.save_model_helpers import create_run_directory

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# here cpu is faster because we use Mplpolicy, and it is more costly to copy the data to gpu here (even trough on mac this shouldn't be a problem so ii don't know why with mps is slower than cpu if data leaves in same memory)
device = "cpu"
print(f"Using device: {device}")

gym.register_envs(ale_py)


def main(config):
    # Create unique run directory
    run_path = create_run_directory(config["paths"]["model_save_path"])
    print(f"Run directory created at: {run_path}")

    # Update paths to use our run directory
    config["paths"]["model_save_path"] = str(run_path)
    config["paths"]["log_dir"] = str(run_path / "logs")

    config_save_path = str(run_path / "config.yml")
    print(f"Saving configuration used to: {config_save_path}")
    try:
        with open(config_save_path, "w") as f:
            # Use yaml.dump to write the dictionary to the file
            # default_flow_style=False makes it more readable (block style)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error saving config file: {e}")

    """Main training and evaluation function."""

    # --- Environment Setup ---
    print(f"Creating training environment: {config['env_id']} (x{config['environment']['n_envs_train']})")
    env = make_no_vision_env(
        config["env_id"], config["environment"]["n_envs_train"], config["seed"]
    )

    print(f"Creating evaluation environment: {config['env_id']} (x{config['environment']['n_envs_eval']})")
    # We use a different seed for the evaluation env
    eval_env = make_no_vision_env(
        config["env_id"], config["environment"]["n_envs_train"], config["seed"] + 10
    )

    # --- Callbacks ---
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=config['evaluation']['reward_threshold'],
        verbose=1
    )

    # Adjust eval_freq based on the number of training environments
    eval_freq_adjusted = config['evaluation']['eval_freq']
    print(f"Evaluation frequency (adjusted for {config['environment']['n_envs_train']} envs): {eval_freq_adjusted} steps")

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=config['paths']['model_save_path'], # Save best model here
        log_path=config['paths']['log_dir'],
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        deterministic=config['evaluation']['deterministic'],
        render=False,
        verbose=1,
    )

    # --- Training ---
    print("Starting training...")
    algo_config = config['algorithm']
    model = DQN(
        policy=algo_config["policy"],
        env=env,
        verbose=0,
        buffer_size=algo_config["buffer_size"],
        learning_starts=algo_config["learning_starts"],
        batch_size=algo_config["batch_size"],
        gamma=algo_config["gamma"],
        train_freq=algo_config["train_freq"],
        gradient_steps=algo_config["gradient_steps"],
        target_update_interval=algo_config["target_update_interval"],
        exploration_fraction=algo_config["exploration_fraction"],
        exploration_final_eps=algo_config["exploration_final_eps"],
        learning_rate=algo_config["learning_rate"],
        seed=config["seed"],
        tensorboard_log=config["paths"]["log_dir"],
        device=device,
        policy_kwargs=dict(net_arch=[128, 256, 128, 128]),
    )

    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        progress_bar=True
    )

    # Save the final model
    final_model_path = os.path.join(run_path, "model_final.zip")
    print(f"Training finished. Saving final model to: {final_model_path}")
    model.save(final_model_path)

    # Load the best model saved by the callback for final evaluation
    best_model_path = os.path.join(run_path, "best_model.zip")
    print(f"Loading best model from: {best_model_path} for final evaluation...")
    model = DQN.load(best_model_path, env=env, device=device)


    print("Evaluating the loaded model...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        deterministic=config['evaluation']['deterministic']
    )
    print(f"Evaluation results -> Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Cleanup ---
    print("Closing environments.")
    env.close()
    eval_env.close()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pong using config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn_no_vision_pong_config.yml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories if they don't exist based on loaded config
    # os.makedirs(config['paths']['log_dir'], exist_ok=True)

    # os.path.dirname gets the directory part of the path
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)

    # Run the main function
    main(config)