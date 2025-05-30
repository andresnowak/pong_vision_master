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
from gymnasium.spaces import Discrete, Dict, Box
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


from src.separated_vision_buffer import SeparatedVisionFoveaBuffer
from src.separated_buffer import StructuredDoubleActionReplayBuffer
from src.separated_vision_network import SeparatedVisionFoveaNetwork, SeparatedFoveaSFN
from src.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from src.dqn_sugarl import linear_schedule
from src.create_env import make_fovea_env
from src.config_loader import load_config
from src.save_model_helpers import create_run_directory


def train(config):
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

    # NOTE: env setup (How many environments we will use but here we will always use one really, so this isn't necessary)
    # envs = []
    # for i in range(config["environment"]["env_num"]):
    #     envs.append(
    #         make_fovea_env(
    #             config["environment"]["env_id"],
    #             config["seed"] + i,
    #             frame_stack=config["environment"]["frame_stack"],
    #             action_repeat=config["environment"]["action_repeat"],
    #             fov_size=(
    #                 config["environment"]["fov_size"],
    #                 config["environment"]["fov_size"],
    #             ),
    #             fov_init_loc=(
    #                 config["environment"]["fov_init_loc_x"],
    #                 config["environment"]["fov_init_loc_y"],
    #             ),
    #             sensory_action_mode=config["environment"]["sensory_action_mode"],
    #             sensory_action_space=(
    #                 -config["environment"]["sensory_action_space"],
    #                 config["environment"]["sensory_action_space"],
    #             ),
    #             resize_to_full=config["environment"]["resize_to_full"],
    #             clip_reward=config["environment"]["clip_reward"],
    #             mask_out=True,
    #         )
    #     )
    # # envs = gym.vector.AsyncVectorEnv(envs)
    env_wrapper = make_fovea_env(
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
        resize_to_full=config["environment"]["resize_to_full"],
        clip_reward=config["environment"]["clip_reward"],
        mask_out=True,
        return_separated=True,  # Option ajoutée pour retourner les observations séparées
    )
    envs = gym.vector.SyncVectorEnv([env_wrapper])

    # Créer l'espace d'observation structuré si l'environnement ne le fait pas déjà
    # Pour le cas où l'environnement ne fournit pas d'observations séparées
    if not isinstance(envs.single_observation_space, Dict):
        # Déterminer les valeurs min/max et le type de données à partir de l'espace d'observation existant
        original_obs_space = envs.single_observation_space
        low_val = original_obs_space.low.min()
        high_val = original_obs_space.high.max()
        obs_dtype = original_obs_space.dtype
        
        # Créer un espace d'observation structuré
        fov_size = config["environment"]["fov_size"]
        frame_stack = config["environment"]["frame_stack"]
        
        obs_space = Dict({
            "fovea": Box(
                low=low_val, 
                high=high_val, 
                shape=(frame_stack, fov_size, fov_size), 
                dtype=obs_dtype
            ),
            "position": Box(
                low=0,
                high=84,  # Supposons que l'espace d'observation est 84x84
                shape=(2,),
                dtype=np.float32
            )
        })
    else:
        obs_space = envs.single_observation_space
    
    # Facteur d'échelle de récompense (si utilisé)
    sugarl_r_scale = get_sugarl_reward_scale_atari(config["environment"]["env_id"])
    
    # Configurer l'espace d'action sensorielle discrète
    OBSERVATION_SIZE = (84, 84)  # Taille de l'observation complète
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
    
    # Créer les réseaux pour l'agent
    q_network = SeparatedVisionFoveaNetwork(
        obs_space, 
        envs.single_action_space["motor_action"], 
        sensory_action_set
    ).to(device)
    
    optimizer = optim.Adam(
        q_network.parameters(), 
        lr=config["algorithm"]["learning_rate"]
    )
    
    target_network = SeparatedVisionFoveaNetwork(
        obs_space, 
        envs.single_action_space["motor_action"], 
        sensory_action_set
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    # Initialiser le SFN pour les observations séparées
    sfn = SeparatedFoveaSFN(obs_space, envs.single_action_space["motor_action"]).to(device)
    sfn_optimizer = optim.Adam(
    sfn.parameters(),
    lr=config["algorithm"]["learning_rate"]
    )
    
    # Initialiser le buffer de replay
    rb = StructuredDoubleActionReplayBuffer(
        config["algorithm"]["buffer_size"],
        obs_space,
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
    pvm_buffer = SeparatedVisionFoveaBuffer(
        config["environment"]["pvm_stack"],
        observation_structure=obs_space,
    )

    progress_bar = tqdm(
        initial=global_transitions,
        total=config["algorithm"]["total_timesteps"],
        desc="Training",
        unit="transitions",
    )

    while global_transitions < config["algorithm"]["total_timesteps"]:
        # Convertir les observations si nécessaire
        if not isinstance(obs, dict):
            # Si l'environnement ne retourne pas directement des observations séparées,
            # vous devrez les extraire/convertir ici
            # Cette partie dépend beaucoup de comment votre environnement est configuré
            structured_obs = convert_to_structured_obs(obs, config)
        else:
            structured_obs = obs
                # À ajouter après avoir créé une observation structurée
        # Ajouter l'observation au buffer
        pvm_buffer.append(structured_obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
        
        # Sélection d'action (epsilon-greedy)
        epsilon = linear_schedule(
            config["algorithm"]["start_e"],
            config["algorithm"]["end_e"],
            config["algorithm"]["exploration_fraction"] * config["algorithm"]["total_timesteps"],
            global_transitions,
        )
        
        if random.random() < epsilon:
            # Exploration: actions aléatoires
            actions = envs.single_action_space.sample()
            motor_actions = np.array([actions["motor_action"]])
            sensory_actions = np.array([random.randint(0, len(sensory_action_set) - 1)])
        else:
            # Exploitation: utiliser le réseau
            # Convertir chaque composante en tenseur
            obs_tensors = {
                key: torch.from_numpy(pvm_obs[key]).to(device)
                for key in pvm_obs
            }
            
            motor_q_values, sensory_q_values = q_network(obs_tensors)
            motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
            sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

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
        # Convertir next_obs si nécessaire
        if not isinstance(next_obs, dict):
            structured_next_obs = convert_to_structured_obs(next_obs, config)
        else:
            structured_next_obs = next_obs
        
        # Stocker dans le buffer de replay
        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(structured_next_obs)
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
        
        # Mettre à jour pour la prochaine itération
        obs = next_obs
        global_transitions += envs.num_envs
        progress_bar.update(envs.num_envs)

        obs_backup = obs  # back obs

        if global_transitions < config["algorithm"]["batch_size"]:
            continue

        # Training
        # Remplacer la partie d'entraînement du SFN
        if global_transitions % config["algorithm"]["train_frequency"] == 0:
            data = rb.sample(
                config["algorithm"]["batch_size"] // config["environment"]["env_num"]
            )
            
            # Préparer les données pour le SFN
            sfn_input = {
                "current": data.observations,
                "next": data.next_observations
            }
            
            # Apprentissage du SFN
            pred_motor_actions = sfn(sfn_input)
            sfn_loss = sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())
            sfn_optimizer.zero_grad()
            sfn_loss.backward()
            sfn_optimizer.step()
            
            # Calcul du reward observationnel
            observ_r = (
                F.softmax(pred_motor_actions)
                .gather(1, data.motor_actions)
                .squeeze()
                .detach()
            )  # 0-1
        
            # Apprentissage du réseau Q
            if global_transitions > config["algorithm"]["learning_starts"]:
                with torch.no_grad():
                    motor_target, sensory_target = target_network(data.next_observations)
                    motor_target_max, _ = motor_target.max(dim=1)
                    sensory_target_max, _ = sensory_target.max(dim=1)
                    
                    # Ajuster la récompense d'observation
                    observ_r_adjusted = observ_r.clone()
                    observ_r_adjusted[data.rewards.flatten() > 0] = (
                        1 - observ_r_adjusted[data.rewards.flatten() > 0]
                    )
                    
                    # Calculer la cible TD (avec et sans ajustement)
                    td_target = (
                        data.rewards.flatten()
                        - (1 - observ_r) * sugarl_r_scale
                        + config["algorithm"]["gamma"]
                        * (motor_target_max + sensory_target_max)
                        * (1 - data.dones.flatten())
                    )
                    
                    original_td_target = data.rewards.flatten() + config["algorithm"][
                        "gamma"
                    ] * (motor_target_max + sensory_target_max) * (
                        1 - data.dones.flatten()
                    )
        
                # Calculer les valeurs Q actuelles
                old_motor_q_val, old_sensory_q_val = q_network(data.observations)
                old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
                old_sensory_val = old_sensory_q_val.gather(
                    1, data.sensory_actions
                ).squeeze()
                old_val = old_motor_val + old_sensory_val
        
                # Calculer la perte et mettre à jour le réseau
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
                        "losses/sfn_loss", sfn_loss.item(), global_transitions
                    )
                    writer.add_scalar(
                        "losses/observ_r", observ_r.mean().item(), global_transitions
                    )
                    writer.add_scalar(
                        "losses/original_td_target",
                        original_td_target.mean().item(),
                        global_transitions,
                    )
                    writer.add_scalar(
                        "losses/sugarl_r_scaled_td_target",
                        td_target.mean().item(),
                        global_transitions,
                    )

                    wandb.log(
                        {
                            "td_loss": loss,
                            "q_values": old_motor_val.mean().item(),
                            "sensor_q_values": old_sensory_val.mean().item(),
                            "SPS": int(global_transitions / (time.time() - start_time)),
                            "sfn_loss": sfn_loss.item(),
                            "observ_r": observ_r.mean(),
                            "original_td_target": original_td_target.mean().item(),
                            "sugarl_r_scaled_td_target": td_target.mean().item(),
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
            # Remplacer la partie d'évaluation
            
            if (global_transitions % config["evaluation"]["eval_frequency"] == 0
                and config["evaluation"]["eval_frequency"] > 0) or (global_transitions >= config["algorithm"]["total_timesteps"]):
                q_network.eval()
                sfn.eval()
            
                eval_episodic_returns, eval_episodic_lengths = [], []
            
                for eval_ep in range(config["evaluation"]["eval_num"]):
                    eval_env = make_fovea_env(
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
                        sensory_action_mode=config["environment"]["sensory_action_mode"],
                        sensory_action_space=(
                            -config["environment"]["sensory_action_space"],
                            config["environment"]["sensory_action_space"],
                        ),
                        resize_to_full=config["environment"]["resize_to_full"],
                        clip_reward=config["environment"]["clip_reward"],
                        mask_out=True,
                        return_separated=True,  # Important: obtenir des observations séparées
                        training=False,
                        record=config["capture_video"],
                    )
                    eval_env = gym.vector.SyncVectorEnv([eval_env])
            
                    obs_eval, _ = eval_env.reset()
                    done = False
                    
                    # Créer un buffer d'observations séparé pour l'évaluation
                    pvm_buffer_eval = SeparatedVisionFoveaBuffer(
                        config["environment"]["pvm_stack"],
                        observation_structure=obs_space
                    )
                    
                    while not done:
                        # Convertir les observations si nécessaire
                        if not isinstance(obs_eval, dict):
                            structured_obs_eval = convert_to_structured_obs(obs_eval, config)
                        else:
                            structured_obs_eval = obs_eval
                        
                        # Stocker l'observation
                        pvm_buffer_eval.append(structured_obs_eval)
                        pvm_obs_eval = pvm_buffer_eval.get_obs(mode="stack_max")
                        
                        # Convertir en tenseurs
                        obs_tensors_eval = {
                            key: torch.from_numpy(pvm_obs_eval[key]).to(device)
                            for key in pvm_obs_eval
                        }
                        
                        # Obtenir les actions via le réseau
                        motor_q_values, sensory_q_values = q_network(obs_tensors_eval)
                        motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
                        sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()
                        
                        # Exécuter l'action
                        next_obs_eval, rewards, dones, _, infos = eval_env.step(
                            {
                                "motor_action": motor_actions,
                                "sensory_action": [sensory_action_set[a] for a in sensory_actions],
                            }
                        )
                        
                        obs_eval = next_obs_eval
                        done = dones[0]
                        
                        # Gérer la fin d'un épisode
                        if done:
                            eval_episodic_returns.append(infos["final_info"][0]["reward"])
                            eval_episodic_lengths.append(infos["final_info"][0]["ep_len"])
                            
                            # Enregistrer la vidéo si nécessaire
                            if config["capture_video"]:
                                record_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "recordings"
                                )
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(
                                    os.path.join(record_file_dir, record_file_fn)
                                )
                            
                            # Sauvegarder le modèle pour le premier épisode d'évaluation
                            if eval_ep == 0:
                                model_file_dir = os.path.join(
                                    config["paths"]["model_save_path"], "trained_models"
                                )
                                os.makedirs(model_file_dir, exist_ok=True)
                                model_fn = f"{config['environment']['env_id']}_seed{config['seed']}_step{global_transitions:07d}_model.pt"
                                torch.save(
                                    {
                                        "sfn": sfn.state_dict(),
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
                sfn.train()

        obs = obs_backup  # restore obs if eval occurs

    envs.close()
    eval_env.close()
    writer.close()
    progress_bar.close()

def convert_to_structured_obs(obs, config):
    """
    Convertit une observation brute en observations structurées avec fovéa et périphérie.
    """
    # Vérifier la forme de l'observation
    batch_size, frames, height, width = obs.shape
    
    # Paramètres de configuration
    fov_size = config["environment"]["fov_size"]
    
    # Afficher des informations sur les valeurs pour déboguer
    #print(f"Plage de valeurs obs: min={obs.min()}, max={obs.max()}, type={obs.dtype}")
    
    
    # Normaliser les données si nécessaire
    # Si les valeurs sont entre -1 et 1, les ramener entre 0 et 1 pour l'affichage
    if obs.min() < 0:
        normalized_obs = (obs + 1) / 2
    else:
        normalized_obs = obs.copy()
    
    # Calculer les coordonnées du centre de l'image
    center_x, center_y = width // 2, height // 2
    
    # Calculer les coordonnées de la fovéa
    fov_x1 = max(0, center_x - fov_size // 2)
    fov_y1 = max(0, center_y - fov_size // 2)
    fov_x2 = min(width, fov_x1 + fov_size)
    fov_y2 = min(height, fov_y1 + fov_size)
    
    # Initialiser les structures pour les observations séparées
    fovea = np.zeros((batch_size, frames, fov_size, fov_size), dtype=normalized_obs.dtype)
    
    # Extraire la région fovéale (vision haute résolution)
    for b in range(batch_size):
        for f in range(frames):
            fovea_region = normalized_obs[b, f, fov_y1:fov_y2, fov_x1:fov_x2]
            if fovea_region.shape[0] < fov_size or fovea_region.shape[1] < fov_size:
                y_pad = max(0, fov_size - fovea_region.shape[0])
                x_pad = max(0, fov_size - fovea_region.shape[1])
                fovea_region = np.pad(fovea_region, ((0, y_pad), (0, x_pad)), mode='constant')
            fovea[b, f] = fovea_region
    
    
    # Position de la fovéa
    position = np.array([[center_x - fov_size // 2, center_y - fov_size // 2]], dtype=np.float32)
    
    # Validation supplémentaire pour l'affichage
    
    return {
        "fovea": fovea,  
        "position": position
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pong using config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn_fovea_pong_light_config.yml",
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
