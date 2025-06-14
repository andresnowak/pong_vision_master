"""
# This module implements methods from (with modifications):
# Jinghuan Shang and Michael S. Ryoo. "Active Reinforcement Learning under Limited Visual Observability" (2023).
# arXiv:2306.00975 [cs.LG].
https://github.com/elicassion/sugarl

In this "Full Vision" option, we don't have a fovea and the agent gets a full unblurred 84x84 view of the environment.
The q_network has no sensory policy head and outputs None for the sensory q values. This script should be paired with dqn_fullvis_pong.yml

We leave the SUGARL specific code commented instead of erasing it to see where SUGARL is involved

"""

print("-#- Training script starts")

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

from src.buffer import DoubleActionReplayBuffer
from src.pvm_buffer import PVMBuffer
from src.utils import get_timestr, seed_everything 
from src.dqn_fullvis import QNetwork, linear_schedule
from src.create_env import make_fovea_peripheral_env
from src.config_loader import load_config
from src.save_model_helpers import create_run_directory

print("-#- Imports done")


def train(config):

    # With no possibility to move the fovea we can hardcode the sensory_actions as a singleton
    SENSORY_ACTIONS_FIXED = np.array([0])

    wandb.init(
        project=config["project"],
        name=config["exp_name"],
        config=config,  # Pass the entire config dictionary
    )
    writer = SummaryWriter(config["paths"]["log_dir"])
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    seed_everything(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("-#- Training on:", device)

    envs = make_fovea_peripheral_env(
        config["environment"]["env_id"],
        config["seed"],
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

    #sugarl_r_scale = get_sugarl_reward_scale_atari(config["environment"]["env_id"])

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
    

    # Full vision => no sensory action set
    #sensory_action_set = None
    #print(sensory_action_set)

    q_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    optimizer = optim.Adam(
        q_network.parameters(), lr=config["algorithm"]["learning_rate"]
    )
    target_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    target_network.load_state_dict(q_network.state_dict())

    #sfn = SelfPredictionNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    #sfn_optimizer = optim.Adam(
    #    sfn.parameters(), lr=config["algorithm"]["learning_rate"]
    #)

    # print(envs.single_observation_space, envs.num_envs)
    # obs, _ = envs.reset()
    # print("Observation type:", obs.dtype)  # Should be uint8
    # print("Min/Max values:", obs.min(), obs.max())  # Should be 0 and 255 if uint8

    rb = DoubleActionReplayBuffer(
        config["algorithm"]["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space["motor_action"],
        Discrete(len(sensory_action_set)),
        device,
        n_envs=envs.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset()
    global_transitions = 0
    pvm_buffer = PVMBuffer(
        config["environment"]["pvm_stack"],
        (
            envs.num_envs,
            config["environment"]["frame_stack"],
        )
        + OBSERVATION_SIZE,
    )

    progress_bar = tqdm(
        initial=global_transitions,
        total=config["algorithm"]["total_timesteps"],
        desc="Training",
        unit="transitions",
    )

    while global_transitions < config["algorithm"]["total_timesteps"]:
        # print(sensory_action_set) # [array([0, 0])]
        pvm_buffer.append(obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            config["algorithm"]["start_e"],
            config["algorithm"]["end_e"],
            config["algorithm"]["exploration_fraction"]
            * config["algorithm"]["total_timesteps"],
            global_transitions,
        )
        if random.random() < epsilon:
            # actions = np.array(
            #     [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            # )
            actions = envs.single_action_space.sample()
            motor_actions = np.array([actions["motor_action"]])
            sensory_actions = SENSORY_ACTIONS_FIXED # np.array([random.randint(0, len(sensory_action_set) - 1)])
            #print(SENSORY_ACTIONS_FIXED)
            #print(np.array([random.randint(0, len(sensory_action_set) - 1)]))
        else:
            motor_q_values, sensory_q_values = q_network(
                resize(torch.from_numpy(pvm_obs)).to(device)
            )
            motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
            # The q_networks outputs None for the sensory action in this full vision version
            #sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()
        
        sensory_actions = SENSORY_ACTIONS_FIXED #np.array([random.randint(0, len(sensory_action_set) - 1)])
        #print(sensory_actions) # [0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step(
            {
                "motor_action": motor_actions,
                "sensory_action": [sensory_action_set[a] for a in sensory_actions],
            }
        )
        # print (global_step, infos)

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
                            "Reward": infos["final_info"][idx]["reward"],
                        }
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        infos["final_info"][idx]["reward"],
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
                            "episodic_return": infos["final_info"][idx]["reward"],
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
        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        rb.add(
            pvm_obs,
            real_next_pvm_obs,
            motor_actions,
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

            # sfn learning
            concat_observation = torch.concat(
                [data.next_observations, data.observations], dim=1
            )  # concat at dimension T
            #pred_motor_actions = sfn(resize(concat_observation))
            # print (pred_motor_actions.size(), data.motor_actions.size())
            #sfn_loss = sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())
            #sfn_optimizer.zero_grad()
            #sfn_loss.backward()
            #sfn_optimizer.step()
            """
            observ_r = (
                F.softmax(pred_motor_actions)
                .gather(1, data.motor_actions)
                .squeeze()
                .detach()
            )  # 0-1
            """

            # Q network learning
            if global_transitions > config["algorithm"]["learning_starts"]:
                with torch.no_grad():
                    motor_target, sensory_target = target_network(
                        resize(data.next_observations)
                    )
                    motor_target_max, _ = motor_target.max(dim=1)
                    #sensory_target_max, _ = sensory_target.max(dim=1)
                    # scale step-wise reward with observ_r
                    #observ_r_adjusted = observ_r.clone()
                    #observ_r_adjusted[data.rewards.flatten() > 0] = (
                    #    1 - observ_r_adjusted[data.rewards.flatten() > 0]
                    #)
                    # DQN loss formula (but they have the differnce of rewards - everything else instead of +)
                    
                    """
                    td_target = (
                        data.rewards.flatten()
                        - (1 - observ_r) * sugarl_r_scale
                        + config["algorithm"]["gamma"]
                        * (motor_target_max) # + sensory_target_max)
                        * (1 - data.dones.flatten())
                    )
                
                    # DQN loss formula
                    original_td_target = data.rewards.flatten() + config["algorithm"][
                        "gamma"
                    ] * (motor_target_max) * ( # + sensory_target_max) * (
                        1 - data.dones.flatten()
                    )
                    """
                    # DQN algorithm target value computation
                    # We replace the SUGARL block with a vanilla DQN td_target = reward + gamma * target_q * (1 - done)
                    td_target = (
                    data.rewards.flatten()
                    + config["algorithm"]["gamma"]
                    * motor_target_max
                    * (1 - data.dones.flatten())
                    )

                old_motor_q_val, old_sensory_q_val = q_network(
                    resize(data.observations)
                )
                old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
                """
                old_sensory_val = old_sensory_q_val.gather(
                    1, data.sensory_actions
                ).squeeze()
                """
                old_val = old_motor_val #+ old_sensory_val

                loss = F.mse_loss(td_target, old_val)

                if global_transitions % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_transitions)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_transitions
                    )
                    writer.add_scalar(
                        "losses/motor_q_values",
                        old_motor_val.mean().item(),
                        global_transitions,
                    )
                    #writer.add_scalar(
                    #    "losses/sensor_q_values",
                    #    old_sensory_val.mean().item(),
                    #    global_transitions,
                    #)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_transitions / (time.time() - start_time)),
                        global_transitions,
                    )

                    #writer.add_scalar(
                    #    "losses/sfn_loss", sfn_loss.item(), global_transitions
                    #)
                    #writer.add_scalar(
                    #    "losses/observ_r", observ_r.mean().item(), global_transitions
                    #)
                    #writer.add_scalar(
                    #    "losses/original_td_target",
                    #    original_td_target.mean().item(),
                    #    global_transitions,
                    #)
                    writer.add_scalar(
                        #"losses/sugarl_r_scaled_td_target",
                        # Change the name since no longer SUGARL
                        "losses/target_q_value",
                        td_target.mean().item(),
                        global_transitions,
                    )

                    wandb.log(
                        {
                            "td_loss": loss,
                            "q_values": old_motor_val.mean().item(),
                            #"sensor_q_values": old_sensory_val.mean().item(),
                            "SPS": int(global_transitions / (time.time() - start_time)),
                            #"sfn_loss": sfn_loss.item(),
                            #"observ_r": observ_r.mean(),
                            #"original_td_target": original_td_target.mean().item(),
                            #"sugarl_r_scaled_td_target": td_target.mean().item(),
                            "target_q_value": td_target.mean().item(),
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
                #sfn.eval()

                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(config["evaluation"]["eval_num"]):
                    eval_env = make_fovea_peripheral_env(
                        config["environment"]["env_id"],
                        config["seed"] + eval_ep,
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
                    pvm_buffer_eval = PVMBuffer(
                        config["environment"]["pvm_stack"],
                        (
                            eval_env.num_envs,
                            config["environment"]["frame_stack"],
                        )
                        + OBSERVATION_SIZE,
                    )
                    while not done:
                        pvm_buffer_eval.append(obs_eval)
                        pvm_obs_eval = pvm_buffer_eval.get_obs(mode="stack_max")
                        motor_q_values, sensory_q_values = q_network(
                            resize(torch.from_numpy(pvm_obs_eval)).to(device)
                        )
                        motor_actions = (
                            torch.argmax(motor_q_values, dim=1).cpu().numpy()
                        )
                        """
                        sensory_actions = (
                            torch.argmax(sensory_q_values, dim=1).cpu().numpy()
                        )
                        """
                        sensory_actions = SENSORY_ACTIONS_FIXED #np.array([random.randint(0, len(sensory_action_set) - 1)])
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
                                infos["final_info"][0]["reward"]
                            )
                            eval_episodic_lengths.append(
                                infos["final_info"][0]["ep_len"]
                            )
                            if config["capture_video"]:
                                # record_file_dir = os.path.join(
                                #     "recordings",
                                #     config["exp_name"],
                                #     os.path.basename(__file__).rstrip(".py"),
                                #     config["environment"]["env_id"],
                                # )
                                record_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "recordings"
                                )
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(
                                    os.path.join(record_file_dir, record_file_fn)
                                )

                            if eval_ep == 0:
                                # model_file_dir = os.path.join(
                                #     "trained_models",
                                #     config['exp_name'],
                                #     os.path.basename(__file__).rstrip(".py"),
                                #     config["environment"]["env_id"],
                                # )
                                model_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "trained_models"
                                )
                                os.makedirs(model_file_dir, exist_ok=True)
                                model_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_model.pt"
                                torch.save(
                                    {
                                        #"sfn": sfn.state_dict(),
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
                #sfn.train()

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
        default= "config/dqn_fullvis_pong_config.yml", #"config/dqn_fovea_peripheral_pong_config.yml",
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
