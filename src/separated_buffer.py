import numpy as np
import torch as th
import copy
import warnings
from typing import Dict, Any, Optional, Union
from gymnasium.spaces import Space, Dict as DictSpace
from gymnasium.vector import VectorEnv

from src.utils import get_action_dim, get_obs_shape, get_device
from src.buffer import BaseBuffer, DoubleActionReplayBufferSamples

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class StructuredDoubleActionReplayBuffer(BaseBuffer):
    """
    Buffer de replay adapté pour stocker des observations structurées avec vision fovéale et périphérique
    séparées, ainsi que les actions motrices et sensorielles.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: DictSpace,
        motor_action_space: Space,
        sensory_action_space: Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.motor_action_space = motor_action_space
        self.sensory_action_space = sensory_action_space
        
        # Extraire les formes pour chaque composant de l'observation
        self.obs_shapes = {
            key: get_obs_shape(space) for key, space in observation_space.spaces.items()
        }
        
        self.motor_action_dim = get_action_dim(motor_action_space)
        self.sensory_action_dim = get_action_dim(sensory_action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        
        # Ajuster la taille du buffer
        self.buffer_size = max(buffer_size // n_envs, 1)
        
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer ne supporte pas optimize_memory_usage = True "
                "et handle_timeout_termination = True simultanément."
            )
        self.optimize_memory_usage = optimize_memory_usage
        
        # Créer des arrays pour chaque composant de l'observation
        self.observations = {}
        for key, shape in self.obs_shapes.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs) + shape, 
                dtype=observation_space.spaces[key].dtype
            )
        
        if optimize_memory_usage:
            # Les observations contiennent aussi les prochaines observations
            self.next_observations = None
        else:
            self.next_observations = {}
            for key, shape in self.obs_shapes.items():
                self.next_observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs) + shape, 
                    dtype=observation_space.spaces[key].dtype
                )
        
        # Arrays pour les actions et récompenses
        self.motor_actions = np.zeros((self.buffer_size, self.n_envs, self.motor_action_dim), dtype=motor_action_space.dtype)
        self.sensory_actions = np.zeros((self.buffer_size, self.n_envs, self.sensory_action_dim), dtype=sensory_action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Gestion des timeouts
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Stockage d'informations supplémentaires
        self.infos = [None] * self.buffer_size
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        motor_action: np.ndarray,
        sensory_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """
        Ajoute une transition au buffer
        """
        # Copier les observations pour éviter les modifications par référence
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key]).copy()
            
            if self.optimize_memory_usage:
                next_pos = (self.pos + 1) % self.buffer_size
                self.observations[key][next_pos] = np.array(next_obs[key]).copy()
            else:
                self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()
        
        # Redimensionner les actions si nécessaire
        motor_action = motor_action.reshape((self.n_envs, self.motor_action_dim))
        sensory_action = sensory_action.reshape((self.n_envs, self.sensory_action_dim))
        
        # Stocker les actions, récompenses et dones
        self.motor_actions[self.pos] = np.array(motor_action).copy()
        self.sensory_actions[self.pos] = np.array(sensory_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.infos[self.pos] = copy.deepcopy(infos)
        
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([False])
        
        # Mettre à jour la position et vérifier si le buffer est plein
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> DoubleActionReplayBufferSamples:
        """
        Échantillonne des éléments du buffer
        """
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> DoubleActionReplayBufferSamples:
        # Échantillonner aléatoirement les indices d'environnement
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        # Préparer les observations structurées
        observations = {}
        next_observations = {}
        
        for key in self.observations.keys():
            observations[key] = self.observations[key][batch_inds, env_indices]
            
            if self.optimize_memory_usage:
                next_observations[key] = self.observations[key][(batch_inds + 1) % self.buffer_size, env_indices]
            else:
                next_observations[key] = self.next_observations[key][batch_inds, env_indices]
        
        # Convertir en tenseurs PyTorch
        torch_observations = {key: self.to_torch(obs) for key, obs in observations.items()}
        torch_next_observations = {key: self.to_torch(obs) for key, obs in next_observations.items()}
        
        # Préparer les autres données
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
        
        # Retourner l'échantillon complet
        return DoubleActionReplayBufferSamples(
            observations=torch_observations,
            motor_actions=self.to_torch(self.motor_actions[batch_inds, env_indices]),
            sensory_actions=self.to_torch(self.sensory_actions[batch_inds, env_indices]),
            next_observations=torch_next_observations,
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
        )