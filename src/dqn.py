import torch.nn as nn
from gymnasium.spaces import Discrete, Dict

class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if sensory_action_set == None:
            if isinstance(env.single_action_space, Discrete):
                action_space_size = env.single_action_space.n
            else:
                action_space_size = env.single_action_space[
                    "motor_action"
                ].n
        else:
            action_space_size = len(sensory_action_set)

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_size),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)