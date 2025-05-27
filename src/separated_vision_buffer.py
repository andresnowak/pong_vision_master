   
import numpy as np
import torch
from collections import deque


class SeparatedVisionBuffer:
    """
    Buffer adapté pour stocker et récupérer des observations structurées avec vision fovéale et périphérique
    """
    def __init__(self, max_size, observation_structure):
        """
        Args:
            max_size: nombre maximum d'observations dans le buffer
            observation_structure: structure des observations (formes pour fovea, peripheral, etc.)
        """
        self.max_size = max_size
        self.fovea_buffer = deque(maxlen=max_size)
        self.peripheral_buffer = deque(maxlen=max_size)
        self.position_buffer = deque(maxlen=max_size)
        self.observation_structure = observation_structure
    
    def append(self, observation):
        """
        Ajoute une observation structurée au buffer
        
        Args:
            observation: dict contenant 'fovea', 'peripheral' et 'position'
        """
        self.fovea_buffer.append(observation["fovea"].copy())
        self.peripheral_buffer.append(observation["peripheral"].copy())
        self.position_buffer.append(observation["position"].copy())
    
    def get_obs(self, mode="latest"):
        """
        Récupère les observations du buffer selon le mode spécifié
        
        Args:
            mode: 'latest' pour la dernière observation, 'stack_max' pour max pooling, etc.
        
        Returns:
            Un dictionnaire contenant les observations structurées
        """
        if mode == "latest":
            return {
                "fovea": self.fovea_buffer[-1],
                "peripheral": self.peripheral_buffer[-1],
                "position": self.position_buffer[-1]
            }
        elif mode == "stack_max":
            # Pour les images (fovea et peripheral), faire max pooling
            fovea_stack = np.stack(list(self.fovea_buffer))
            peripheral_stack = np.stack(list(self.peripheral_buffer))
            
            return {
                "fovea": np.max(fovea_stack, axis=0),
                "peripheral": np.max(peripheral_stack, axis=0),
                "position": self.position_buffer[-1]  # Pour position, juste prendre la dernière
            }
        
        # Autres modes possibles...
    
    def copy(self):
        """Crée une copie du buffer"""
        new_buffer = SeparatedVisionBuffer(self.max_size, self.observation_structure)
        new_buffer.fovea_buffer = self.fovea_buffer.copy()
        new_buffer.peripheral_buffer = self.peripheral_buffer.copy()
        new_buffer.position_buffer = self.position_buffer.copy()
        return new_buffer


