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
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            #sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n

        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, motor_action_space_size)


    def forward(self, x):
        x = self.backbone(x)
        motor_action = self.motor_action_head(x)

        # Keeping a second dummy head (None) for compatibility
        return motor_action, None



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
