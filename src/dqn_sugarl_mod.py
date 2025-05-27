"""
# This module implements methods from:
# Jinghuan Shang and Michael S. Ryoo. "Active Reinforcement Learning under Limited Visual Observability" (2023).
# arXiv:2306.00975 [cs.LG].
https://github.com/elicassion/sugarl
"""

from torch import nn
import torch
from gymnasium.spaces import Discrete, Dict


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, fovea_shape, peripheral_shape, sensory_action_set=None):
        super().__init__()
        self.sensory_action_set = sensory_action_set  # Store the sensory action set

        # Fovea processing
        self.fovea_net = nn.Sequential(
            nn.Conv2d(fovea_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Peripheral processing
        self.peripheral_net = nn.Sequential(
            nn.Conv2d(peripheral_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Fovea location processing
        self.fov_loc_net = nn.Linear(2, 16)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # Example output size
        )

    def forward(self, fovea, peripheral, fov_loc):
        # Use the sensory action set if needed
        if self.sensory_action_set is not None:
            # Example: Log the sensory action set size
            print(f"Sensory action set size: {len(self.sensory_action_set)}")
    
        fovea_features = self.fovea_net(fovea)
        peripheral_features = self.peripheral_net(peripheral)
        fov_loc_features = self.fov_loc_net(fov_loc)
        combined = torch.cat([fovea_features, peripheral_features, fov_loc_features], dim=1)
        return self.fc(combined)

class SelfPredictionNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        motor_action_space_size = None
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory_action"].n

        self.backbone = nn.Sequential(
            nn.Conv2d(8, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, motor_action_space_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
