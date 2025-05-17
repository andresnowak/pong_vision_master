"""
# This module implements methods from (with modifications):
# Jinghuan Shang and Michael S. Ryoo. "Active Reinforcement Learning under Limited Visual Observability" (2023).
# arXiv:2306.00975 [cs.LG].
https://github.com/elicassion/sugarl
"""

import argparse
import os, sys
import os.path as osp
import random
import time
from itertools import product
from distutils.util import strtobool

sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter
import time
import os
import yaml
from tqdm import tqdm
import wandb

from src.buffer import ReplayBuffer
from src.pvm_buffer import PVMBuffer
from src.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from src.dqn import QNetwork, linear_schedule
from src.create_env import make_fovea_peripheral_vision_separate_env
from src.config_loader import load_config
from src.save_model_helpers import create_run_directory


def train(config):
    wandb.init(
        project=config["project"],
        name=config["exp_name"],
        config=config,  # Pass the entire config dictionary
    )
    writer = SummaryWriter(config["paths"]["log_dir"])


    # TRY NOT TO MODIFY: seeding
    seed_everything(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    resize = Resize((84, 84))

    # get a discrete observ action space
    OBSERVATION_SIZE = (84, 84)
    observ_x_max, observ_y_max = (
        OBSERVATION_SIZE[0] - config["environment"]["fov_size"],
        OBSERVATION_SIZE[1] - config["environment"]["fov_size"],
    )
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

    # Make Env
    envs = make_fovea_peripheral_vision_separate_env(
        config["environment"]["env_id"],
        config["seed"],
        (sensory_action_step[0], sensory_action_step[1]),
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
        resize_to_full=config["environment"]["resize_to_full"],
        clip_reward=config["environment"]["clip_reward"],
        mask_out=True,
    )
    envs = gym.vector.SyncVectorEnv([envs])

    # Make network
    q_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    optimizer = optim.Adam(
        q_network.parameters(), lr=config["algorithm"]["learning_rate"]
    )
    target_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    target_network.load_state_dict(q_network.state_dict())


    rb = ReplayBuffer(
        config["algorithm"]["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space["motor_action"],
        device,
        n_envs=envs.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset()
    global_transitions = 0

    progress_bar = tqdm(
        initial=global_transitions,
        total=config["algorithm"]["total_timesteps"],
        desc="Training",
        unit="transitions",
    )


    # Training
    while global_transitions < config["algorithm"]["total_timesteps"]:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            config["algorithm"]["start_e"],
            config["algorithm"]["end_e"],
            config["algorithm"]["exploration_fraction"]
            * config["algorithm"]["total_timesteps"],
            global_transitions,
        )
    
        if random.random() < epsilon:
            sensory_actions = np.array([random.randint(0, len(sensory_action_set) - 1)])
        else:
            sensory_q_values = q_network(
                resize(torch.from_numpy(obs)).to(device)
            )
            sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

        actions = envs.single_action_space.sample()
        motor_actions = np.array([actions["motor_action"]])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step(
            {
                "motor_action": motor_actions,
                "sensory_action": [sensory_action_set[a] for a in sensory_actions],
            }
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    # print(
                    #     f"[T: {time.time() - start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]"
                    # )
                    progress_bar.set_postfix(
                        {
                            "transitions": global_transitions,
                            "Reward": rewards,
                        }
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        rewards,
                        global_transitions,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        infos["final_info"][idx]["ep_len"],
                        global_transitions,
                    )
                    writer.add_scalar("charts/epsilon", epsilon, global_transitions)

                    # wandb
                    wandb.log(
                        {
                            "episodic_return": rewards,
                            "episodic_length": infos["final_info"][idx]["ep_len"],
                            "epsilon": epsilon,
                            "global_transitions": global_transitions,
                        }
                    )

                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(
            obs,
            real_next_obs,
            sensory_actions,
            rewards,
            dones,
            {},
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # INC total transitions
        global_transitions += config["environment"]["env_num"]
        progress_bar.update(config["environment"]["env_num"])

        obs_backup = obs  # back obs

        if global_transitions < config["algorithm"]["batch_size"]:
            continue

        # Training
        if global_transitions % config["algorithm"]["train_frequency"] == 0:
            data = rb.sample(
                config["algorithm"]["batch_size"] // config["environment"]["env_num"]
            )  # counter-balance the true global transitions used for training

            # observ_r = (
            #     F.softmax(pred_motor_actions)
            #     .gather(1, data.motor_actions)
            #     .squeeze()
            #     .detach()
            # )  # 0-1

            # Q network learning
            if global_transitions > config["algorithm"]["learning_starts"]:
                with torch.no_grad():
                    sensory_target = target_network(
                        resize(data.next_observations)
                    )
                    sensory_target_max, _ = sensory_target.max(dim=1) # we grab the biggest Q value (Remember from the Q value function)
                    # scale step-wise reward with observ_r
    

                    # DQN loss formula (but they have the differnce of rewards - everything else instead of +)
                    # Standard DQN TD Target: R + gamma * max_a' Q_target(s', a')
                    # data.rewards should now be your foveal_tracking_reward
                    td_target = (
                        data.rewards.flatten()  # These are your foveal_tracking_rewards
                        + config["algorithm"]["gamma"]
                        * sensory_target_max
                        * (
                            1 - data.dones.flatten()
                        )  # (1 - done) handles terminal states
                    )

                # 2. Get Current Q-values for actions taken
                # Assuming q_network now only outputs sensory Q-values
                old_sensory_q_val = q_network(
                    resize(data.observations).to(
                        device
                    )  # Ensure .to(device) if not already
                )
                # Get the Q-value for the specific sensory/foveal action that was taken
                old_sensory_val = old_sensory_q_val.gather(
                    1, data.actions
                ).squeeze()

                # 3. Calculate Loss
                # The loss is between the TD target and the Q-value of the action taken
                loss = F.mse_loss(td_target, old_sensory_val)

                if global_transitions % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_transitions)
                    writer.add_scalar(
                        "losses/sensor_q_values",
                        old_sensory_val.mean().item(),
                        global_transitions,
                    )
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_transitions / (time.time() - start_time)),
                        global_transitions,
                    )
                    writer.add_scalar(
                        "losses/observ_r",
                        data.rewards.mean().item(),
                        global_transitions,
                    )
                    writer.add_scalar(
                        "losses/td_target",
                        td_target.mean().item(),
                        global_transitions,
                    )

                    wandb.log(
                        {
                            "td_loss": loss,
                            "td_target": td_target.mean().item(),
                            "sensor_q_values": old_sensory_val.mean().item(),
                            "SPS": int(global_transitions / (time.time() - start_time)),
                            "observ_r": data.rewards.mean(),
                            "td_target": td_target.mean().item(),
                        }
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if (global_transitions // config["environment"]["env_num"]) % config[
                "algorithm"
            ]["target_network_frequency"] == 0:
                target_network.load_state_dict(q_network.state_dict())

            # evaluation
            if (
                global_transitions % config["evaluation"]["eval_frequency"] == 0
                and config["evaluation"]["eval_frequency"] > 0
            ) or (global_transitions >= config["algorithm"]["total_timesteps"]):
                q_network.eval()

                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(config["evaluation"]["eval_num"]):
                    eval_env = make_fovea_peripheral_vision_separate_env(
                        config["environment"]["env_id"],
                        config["seed"] + eval_ep,
                        (sensory_action_step[0], sensory_action_step[1]),
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
                        sensory_action_mode=config["environment"][
                            "sensory_action_mode"
                        ],
                        sensory_action_space=(
                            -config["environment"]["sensory_action_space"],
                            config["environment"]["sensory_action_space"],
                        ),
                        peripheral_res=(config["environment"]["peripheral_res"], config["environment"]["peripheral_res"]),
                        resize_to_full=config["environment"]["resize_to_full"],
                        clip_reward=config["environment"]["clip_reward"],
                        mask_out=True,
                        training=False,
                        record=config["capture_video"],
                    )
                    eval_env = gym.vector.SyncVectorEnv([eval_env])

                    obs_eval, _ = eval_env.reset()
                    done = False

                    while not done:
                        sensory_q_values = q_network(
                            resize(torch.from_numpy(obs_eval)).to(device)
                        )
            
                        actions = envs.single_action_space.sample()
                        motor_actions = np.array([actions["motor_action"]])
                        sensory_actions = (
                            torch.argmax(sensory_q_values, dim=1).cpu().numpy()
                        )
                        next_obs_eval, rewards, dones, _, infos = eval_env.step(
                            {
                                "motor_action": motor_actions,
                                "sensory_action": [
                                    sensory_action_set[a] for a in sensory_actions
                                ],
                            }
                        )
                        obs_eval = next_obs_eval
                        done = dones[0]
                        if done:
                            eval_episodic_returns.append(
                                rewards # to use our custom reward
                            )
                            eval_episodic_lengths.append(
                                infos["final_info"][0]["ep_len"]
                            )
                            if config["capture_video"]:
                                record_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "recordings"
                                )
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(
                                    os.path.join(record_file_dir, record_file_fn)
                                )

                            if eval_ep == 0:
                                model_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "trained_models"
                                )
                                os.makedirs(model_file_dir, exist_ok=True)
                                model_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_model.pt"
                                torch.save(
                                    {
                                        "q": q_network.state_dict(),
                                    },
                                    os.path.join(model_file_dir, model_fn),
                                )

                writer.add_scalar(
                    "charts/eval_episodic_return",
                    np.mean(eval_episodic_returns),
                    global_transitions,
                )
                writer.add_scalar(
                    "charts/eval_episodic_return_std",
                    np.std(eval_episodic_returns),
                    global_transitions,
                )
                # writer.add_scalar("charts/eval_episodic_length", np.mean(), global_transitions)
                print(
                    f"[T: {time.time() - start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]"
                )
                progress_bar.set_postfix(
                    {
                        "eval_episodic_reward": eval_episodic_returns,
                    }
                )

                wandb.log(
                    {
                        "eval_episodic_return": np.mean(eval_episodic_returns),
                        "eval_episodic_return_std": np.std(eval_episodic_returns),
                    }
                )

                q_network.train()

        obs = obs_backup  # restore obs if eval occurs

    envs.close()
    eval_env.close()
    writer.close()
    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pong using config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn_fovea_peripheral_pong_vision_only_config.yml",
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    run_name = f"{config['environment']['env_id']}__{os.path.basename(__file__)}__{config['seed']}__{get_timestr()}"
    run_dir = os.path.join("runs", config["exp_name"])
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    run_path = create_run_directory(config["paths"]["model_save_path"])
    print(f"Run directory created at: {run_path}")

    # Update paths to use our run directory
    config["paths"]["model_save_path"] = str(run_path)
    config["paths"]["log_dir"] = str(run_path / "logs")

    os.makedirs(os.path.dirname(config["paths"]["model_save_path"]), exist_ok=True)

    config_save_path = str(run_path / "config.yml")
    print(f"Saving configuration used to: {config_save_path}")
    try:
        with open(config_save_path, "w") as f:
            # Use yaml.dump to write the dictionary to the file
            # default_flow_style=False makes it more readable (block style)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error saving config file: {e}")

    train(config)
