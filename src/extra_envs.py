import cv2
import numpy as np
import gymnasium as gym
from gymnasium import wrappers


import copy
from enum import IntEnum
from typing import Tuple, Union

import cv2
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import numpy as np

from active_gym import (
    AtariBaseEnv,
    AtariEnvArgs,
    FixedFovealEnv,
    FixedFovealPeripheralEnv,
)


class CVFidexFovealEnv(FixedFovealEnv):
    def __init__(self, env, args):
        super().__init__(env, args)

    def step(self, action):
        """
        action : {"motor_action":
                  "sensory_action": }
        """
        # print ("in env", action, action["motor_action"], action["sensory_action"])
        state, reward, done, truncated, info = self.env.step(action=action["motor_action"])
        # NOTE it seems action is literally the position when we are using absolute, so lets consider we are always using absolute
        sensory_action = self.detect_ball_position(state)
        sensory_action = action["sensory_action"] if sensory_action is None else sensory_action
        sensory_action = np.array(sensory_action)
        fov_state = self._fov_step(full_state=state, action=sensory_action)

        info["fov_loc"] = self.fov_loc.copy()

        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
        return fov_state, reward, done, truncated, info
    
    def detect_ball_position(self, observation):
        # gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        gray = observation[0]
        gray = np.clip(gray * 255, 0, 255)
        gray = gray.astype(np.uint8)

        # Use a low threshold since the ball is small and possibly dim
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.1 < area < 15: # The area of the ball seems to be 3 (but it says w and h are 2 and 4 of the bounding rect, so seems strange to me)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h != 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    candidates.append([x + w // 2, y + h // 2])

        # Return the first candidate found

        # NOTE: Because the environment uses fov_loc as top_left we have from the center of teh ball to get the top left position
        if candidates:
            value = candidates[0]
            value[0] = value[0] - self.fov_size[0] // 2
            value[1] = value[1] - self.fov_size[1] // 2
            return value
        return None
    
def CVAtariFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = CVFidexFovealEnv(base_env, args)
    return wrapped_env