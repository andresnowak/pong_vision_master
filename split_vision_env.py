"""
The goal of this script is to take the fovea peripheral env observations and feed them separatly to the policy network.
By not stacking them, it is possible to get the input image from (84x84) down to 2x(21x21) + 2 for instance, which is 8 times lighter.
(fovea + peripheral + fovea localisation coordinates).

There are multiple ways to input the fovea, peripheral and fovea localisation:
1) Allow fovea and peripheral to have different shapes, and have 2 CNN towers in the policy networks.
2) Stack them to keep one CNN tower and feed the fovea location coordinates separately.
3) Simplest but naive way: Stack them and add a third 21x21 channel indicating where the fovea is on the peripheral view.

- 1 is clearly best, but was hard to make work because of code dependencies (buffer for example). The code we are building on was to built for this and 
I didn't manage to adapt the code.
- 2 is easier, since we don't need to change the code too much. However, the periph and fov channels are stacked but the same pixels don't represent the 
same spot on the image, which is not the assumption CNNs typically make. Also, the fov_loc still needs to be fed separatly, but a 
DoubleActionWithFovlocReplayBuffer is already implemented in SUGARL, which makes it easier.
- 3 Is easiest, but still breaks the CNN assumptions and the gain goes from 8x down to 5x because of the third channel for the fovea loc.
"""

import numpy as np
from active_gym import AtariBaseEnv, AtariEnvArgs
import gymnasium as gym
from active_gym.fov_env import FixedFovealPeripheralEnv
from gymnasium.spaces import Box, Dict
import torch

from torchvision.transforms import Resize



class NaiveSplitFovealPeripheralEnv(FixedFovealPeripheralEnv):

    """
    Start with solution 3
    """

    def __init__(self, env, args):
        super().__init__(env, args)

        self.fov_shape = (self.env.frame_stack, self.fov_size[0], self.fov_size[1])
        self.periph_shape = (self.env.frame_stack, self.peripheral_res[0], self.peripheral_res[1])

        # Define observation space as a dictionary
        self.observation_space = Dict({
            "fovea": Box(low=-1., high=1., shape=self.fov_shape, dtype=np.float32),
            "peripheral": Box(low=-1., high=1., shape=self.periph_shape, dtype=np.float32),
            "fov_loc": Box(low=0., high=1., shape=(2,), dtype=np.float32),
        })


    """
    Override the _get_fov_state method to get a lightweight observation
    """
    def _get_fov_state(self, full_state) -> dict:
        """
        Extracts the fovea, peripheral, and normalized fovea location from the full state.
        Returns a dictionary compatible with the DoubleActionWithFovlocReplayBuffer.
        """
        # Convert full_state to tensor if it's not already
        full_state = torch.from_numpy(full_state) if isinstance(full_state, np.ndarray) else full_state
    
        # Crop the fovea (high-resolution region)
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
                                    self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1]]
    
        # Downsample the peripheral view (low-resolution context)
        peripheral_state = self._squeeze_to_peripheral_size(full_state)
    
        # Normalize the fovea location to [0, 1] range
        fov_loc_normalized = np.array(self.fov_loc) / np.array([self.obs_size[0], self.obs_size[1]])
    
        # Combine the fovea and peripheral into a single observation
        # This is necessary for compatibility with the replay buffer
        combined_observation = {
            "fovea": fov_state.detach().cpu().numpy().astype(np.float32),
            "peripheral": peripheral_state.detach().cpu().numpy().astype(np.float32),
            "fov_loc": fov_loc_normalized.astype(np.float32),
        }
    
        return combined_observation
    

    def _squeeze_to_peripheral_size(self, s) -> torch.Tensor:
        # Downsample the observation to peripheral resolution
        resizer = Resize(self.peripheral_res)
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s)
        return resizer(s)
    

def NaiveSplitAtariFovealPeripheralEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = NaiveSplitFovealPeripheralEnv(base_env, args)
    return wrapped_env



