import gymnasium as gym
import numpy as np
import cv2


# Specific reward for ram environment
class PongRamHitRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.ale = self.unwrapped.ale
        self.last_ball_x = None
        self.old_ram = None
    
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array(
            [21, 21, 255, 255, 255, 255], dtype=np.float32
        )  # Use reasonable upper bounds
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            # The base gymnasium env handles seeding on reset
            pass  # Let the base env handle it

        obs, info = self.env.reset(**kwargs)
        self.last_ball_x = None
        # Convert initial RAM observation to our custom state
        state = self.decode_ram()
        obs_custom = np.array(
            [
                state["cpu_score"],
                state["player_score"],
                state["opponent_y"],
                state["player_y"],
                state["ball_x"],
                state["ball_y"],
            ],
            dtype=np.float32,
        )
        return obs_custom, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self.decode_ram()

        obs = np.array([state["cpu_score"], state["player_score"], state["opponent_y"], state["player_y"], state["ball_x"], state["ball_y"]], dtype=np.float32)

        hit_reward = self.detect_hit(state)
        # reward += hit_reward
        return obs, reward, terminated, truncated, info

    # For determinstic V4 env
    def decode_ram(self):
        ram = self.ale.getRAM()
        # if self.old_ram is not None:
        #     print("Ram:", np.where(ram == self.old_ram)[0])
        #     print(ram)
        self.old_ram = ram.copy()
        # return {
        #     "ball_x": ram[49],
        #     "ball_y": ram[50],
        #     "player_y": ram[51],
        #     "player_x": ram[52],
        #     "opponent_x": ram[53],
        #     "opponent_y": ram[54],
        # }

        # is this the real correct information
        return  {
            "cpu_score": ram[13],
            "player_score": ram[14],
            "opponent_y": ram[50],
            "player_y": ram[51],
            "ball_x": ram[49],
            "ball_y": ram[54]
        }

    def detect_hit(self, state):
        current_ball_x = state["ball_x"]
        hit_reward = 0

        OPPONENT_PADDLE_LEFT_X = 74
        PLAYER_PADDLE_RIGHT_X = 187
        # NOTE: Verify these RAM interpretations for your specific Pong version!

        # Margin to account for discrete steps and ball speed
        # Might need tuning based on observation.
        HIT_MARGIN_PLAYER = 2
        HIT_MARGIN_OPPONENT = 2  # Use a separate margin if needed

        # Debug prints (remove or make conditional for training)
        # print(f"Ball y: {state['ball_y']}")
        # print(f"Ball X: {self.last_ball_x} -> {current_ball_x}")
        # print(f"Player X: {PLAYER_PADDLE_RIGHT_X}, Opponent X: {OPPONENT_PADDLE_LEFT_X}")
        # print(f"player y: {state['player_y']}")
        # print(f"Oponent y: {state['opponent_y']}")

        # NOTE: It seems the ifs happen two times
        if self.last_ball_x is not None:
            # Player Hit: Ball was near/at player paddle, now moving right
            if (
                self.last_ball_x <= OPPONENT_PADDLE_LEFT_X + HIT_MARGIN_OPPONENT
                and current_ball_x > self.last_ball_x  # Check ball is moving right
                # Optional stricter check: ensure it cleared the paddle
                # and current_ball_x > PLAYER_PADDLE_RIGHT_X + HIT_MARGIN_PLAYER
            ):
                # Ensure the player paddle is actually near the ball's y coordinate for a more robust check (optional)
                hit_reward = -0.1

            # Opponent Hit: Ball was near/at opponent paddle, now moving left
            elif (
                self.last_ball_x >= PLAYER_PADDLE_RIGHT_X - HIT_MARGIN_PLAYER
                and current_ball_x < self.last_ball_x  # Check ball is moving left
            ):
                hit_reward = 0.25  # Give reward to player for hitting the ball

        self.last_ball_x = current_ball_x
        return hit_reward


# Specific reward for ram environment
class PongFollowRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, fov_step_size=(10, 10)):
        super().__init__(env)

        self.fov_step_size = fov_step_size
        self.ale = self.unwrapped.ale
        self.last_ball_x = None
        self.old_ram = None

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            # The base gymnasium env handles seeding on reset
            pass  # Let the base env handle it

        obs, info = self.env.reset(**kwargs)
        self.last_ball_x = None

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self.decode_ram()

        ball_position = np.array(
            [
                state["ball_x"],
                state["ball_y"],
            ],
            dtype=np.float32,
        )

        # NOTE: It seems fov_loc is like resize so it isn't the same as ram position
        fov_loc = info["fov_loc"]

        # reward = self.calculate_foveal_reward_external(fov_loc[0], fov_loc[1], ball_position[0], ball_position[1], self.fov_size[0], self.fov_step_size)
        reward = self.calculate_foveal_reward_external_2(fov_loc[0], fov_loc[1], ball_position[0], ball_position[1], self.fov_size[0],
        )
        info["foveal_reward"] = float(reward)

        return obs, reward, terminated, truncated, info

    # For determinstic V4 env
    def decode_ram(self):
        ram = self.ale.getRAM()
        self.old_ram = ram.copy()

        # is this the real correct information
        return {
            "cpu_score": ram[13],
            "player_score": ram[14],
            "opponent_y": ram[50],
            "player_y": ram[51],
            "ball_x": ram[49],
            "ball_y": ram[54],
        }
    
    def calculate_foveal_reward_external(
        self,
        fovea_center_x,
        fovea_center_y,
        ball_x_ram,
        ball_y_ram,
        fovea_size,
        fov_step_size: tuple[int, int] = (10, 10),
    ):
        if ball_x_ram is None or ball_y_ram is None: # Should not happen if RAM state is valid
            return -0.1 # Or some other default

        # Convert RAM ball coordinates to screen/pixel coordinates if necessary
        # For Pong, RAM coordinates often map somewhat directly to pixel rows/columns
        # but might need scaling or offset depending on the screen resolution
        # and how the original game mapped them.
        # Let's assume for now ball_x_ram and ball_y_ram are usable as pixel coordinates
        # This is a HUGE assumption and often requires calibration.
        # Example: If RAM Y is 0-255 but screen playable area is 34-194, you might need to adjust.

        # print(ball_x_ram, ball_y_ram, fovea_center_x, fovea_center_y)
        ball_x_ram, ball_y_ram = self.rescale_ball_position(ball_x_ram, ball_y_ram)

        # print(ball_x_ram, ball_y_ram, fovea_center_x, fovea_center_y)

        dist_x = abs(fovea_center_x - ball_x_ram)
        dist_y = abs(fovea_center_y - ball_y_ram)

        print(dist_x, dist_y)


        leniency_pixels = 4
        central_threshold_x = (
            fov_step_size[0] / 2.0
        ) + leniency_pixels  # e.g., 10/2 + 2 = 7 pixels
        central_threshold_y = (fov_step_size[1] / 2.0) + leniency_pixels

        # Within fovea threshold
        half_fov = fovea_size / 2.0

        if dist_x < central_threshold_x and dist_y < central_threshold_y:
            return 0.01
        elif dist_x < half_fov and dist_y < half_fov:
            return -0.05
        else:
            return -0.1
        
    def calculate_foveal_reward_external_2(
        self,
        fovea_center_x,
        fovea_center_y,
        ball_x_ram,
        ball_y_ram,
        fovea_size,
        max_penalty=0.2,
    ):
        if ball_x_ram is None or ball_y_ram is None:
            return -max_penalty  # fallback default
        
        # NOTE: Position of the fovea is top left of the fovea square it seems not its center
        real_fovea_center_x = fovea_center_x + (fovea_size / 2)
        real_fovea_center_y = fovea_center_y + (fovea_size / 2)
        radius = fovea_size / 2

        # Rescale RAM coordinates to screen coordinates
        ball_x, ball_y = self.rescale_ball_position(ball_x_ram, ball_y_ram)

        # Euclidean distance to center
        dx = real_fovea_center_x - ball_x
        dy = real_fovea_center_y - ball_y

        dist_to_center = np.sqrt(dx**2 + dy**2)
        dist_to_edge = dist_to_center - radius
        # print(dist_to_center, dist_to_edge, radius, ball_x, ball_y, real_fovea_center_x, real_fovea_center_y)

        if dist_to_edge <= 0:
            # print("yeah")
            max_reward = 4.0
            # Normalized distance (0=center, 1=edge)
            norm_dist = dist_to_center / radius  
            # Higher reward when closer to center (quadratic decay)
            # print(max_reward * (1 - norm_dist**2))
            # return max_reward * (1 - norm_dist**2) 
            return 10
        else:
            # Define max distance for normalization (e.g., half the diagonal of 84x84 image)
            max_dist = np.sqrt(84**2 + 84**2) / 2 - radius  # ~59.4
            normalized_dist = min(dist_to_edge / max_dist, 1.0)  # clamp to [0, 1]

            # Reward is less negative the closer the ball is
            shaped_reward = -max_penalty * (normalized_dist**2)

            # return shaped_reward
            return -max_penalty


    def rescale_ball_position(
        self, x_ram, y_ram, original_size=(210, 160), resized_size=(84, 84)
    ):
        orig_w, orig_h = original_size
        resized_w, resized_h = resized_size

        x_resized = int(x_ram * resized_w / orig_w)
        y_resized = int(y_ram * resized_h / orig_h)

        return x_resized, y_resized


class PongCVFovealRewardWrapper(gym.Wrapper): 
    def __init__(
        self,
        env: gym.Env,
        ball_color_lower_rgb=np.array(
            [200, 200, 200], dtype=np.uint8
        ),  # Adjusted to common Pong ball white-ish
        ball_color_upper_rgb=np.array([255, 255, 255], dtype=np.uint8),
        min_ball_area=5,
        max_ball_area=100,
        fov_step_size=(10, 10)
    ):
        super().__init__(env)
    
        self.ball_color_lower_rgb = ball_color_lower_rgb
        self.ball_color_upper_rgb = ball_color_upper_rgb
        self.min_ball_area = min_ball_area
        self.max_ball_area = max_ball_area
        self.fov_step_size = fov_step_size

        # This wrapper does not change the observation space from the wrapped environment
        self.observation_space = self.env.observation_space
        # The action space is also inherited.

        # print("PongCVFovealRewardWrapper initialized. Using OpenCV for ball detection.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Pass through original visual observation

        # Optionally, detect ball in initial obs and add to info
        ball_x_cv, ball_y_cv = self._find_ball_cv(obs)  # obs is the visual frame
        info["cv_ball_position"] = (ball_x_cv, ball_y_cv)
        info["cv_ball_found"] = ball_x_cv is not None

        return obs, info

    def step(self, action):
        """
        Args:
            action_dict (dict): A dictionary expected to contain:
                'paddle_action': The action for the game paddle.
                'fovea_target_on_screen': A tuple (x, y) representing the fovea's
                                           target center coordinates on the screen
                                           for this step (position AFTER foveal move).
        """
        # Step the underlying game environment with the paddle action
        # `obs` here is the visual frame at t+1
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detect ball in the NEW visual frame `obs` using OpenCV
        ball_x_cv, ball_y_cv = self._find_ball_cv(obs)
        fov_loc = info["fov_loc"]

        foveal_tracking_reward = self._calculate_foveal_reward(
            fov_loc[0], fov_loc[1], ball_x_cv, ball_y_cv, self.fov_size[0]
        )

        # Add relevant info for the agent
        info["foveal_tracking_reward"] = foveal_tracking_reward
        info["original_game_reward"] = reward
        info["cv_ball_position"] = (ball_x_cv, ball_y_cv)
        info["cv_ball_found"] = ball_x_cv is not None

        current_reward_to_return = foveal_tracking_reward

        return (
            obs,
            current_reward_to_return,
            terminated,
            truncated,
            info,
        )  # obs is the visual frame

    def _find_ball_cv(self, frame_gray: np.ndarray) -> tuple[int | None, int | None]:
        """
        Finds the ball in a Pong frame (RGB) using basic CV.
        Returns: (x, y) coordinates of the ball centroid, or (None, None) if not found.
        """

        # print(f"frame_rgb shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
        # print(f"lower_rgb shape: {self.ball_color_lower_rgb.shape}, dtype: {self.ball_color_lower_rgb.dtype}")
        # print(f"upper_rgb shape: {self.ball_color_upper_rgb.shape}, dtype: {self.ball_color_upper_rgb.dtype}")
        # if frame_rgb is None:
        #     return None, None

        # --- Step 1: Normalize and convert grayscale to uint8 ---

        if frame_gray.ndim == 3:
            single_frame = frame_gray[-1, :, :]
        else:
            single_frame = frame_gray

        if single_frame.dtype != np.uint8:
            if np.max(single_frame) <= 1.001 and np.min(single_frame) >= -0.001:
                single_frame = (single_frame * 255).astype(np.uint8)
            else:
                single_frame = np.clip(single_frame, 0, 255).astype(np.uint8)

        # --- Step 2: Threshold to find bright spots (white ball) ---
        lower_intensity = 200
        upper_intensity = 255
        mask = cv2.inRange(single_frame, lower_intensity, upper_intensity)

        # --- Step 3: Find contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_x, ball_y = None, None
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                # Compute circularity only if perimeter is non-zero
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                else:
                    circularity = 0

                # Check area and circularity bounds
                if self.min_ball_area < area < self.max_ball_area and circularity > 0.7:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        ball_x = int(M["m10"] / M["m00"])
                        ball_y = int(M["m01"] / M["m00"])
                        break  # Assume single ball
        return ball_x, ball_y

    # NOTE: Ball can't be exactly at the center of the vision because our action space is discrete

    # def _calculate_foveal_reward(
    #     self, fovea_center_x, fovea_center_y, ball_screen_x, ball_screen_y, fovea_size
    # ):
    #     if ball_screen_x is None or ball_screen_y is None:  # Ball not found by CV
    #         return -0.1

    #     dist_x = abs(fovea_center_x - ball_screen_x)
    #     dist_y = abs(fovea_center_y - ball_screen_y)

    #     central_threshold = fovea_size / 8.0
    #     half_fov = fovea_size / 2.0

    #     if dist_x < central_threshold and dist_y < central_threshold:
    #         return 0.01
    #     elif dist_x < half_fov and dist_y < half_fov:
    #         return -0.05
    #     else:
    #         return -0.1
        
    def _calculate_foveal_reward(
        self,
        fovea_center_x,
        fovea_center_y,
        ball_screen_x,
        ball_screen_y,
        fovea_size,
        fov_step_size: tuple[int, int]=(10, 10),  # EXAMPLE: Approximate spacing of your discrete fovea grid
    ):
        if ball_screen_x is None or ball_screen_y is None:
            return -0.1
        
        # print(ball_screen_x, ball_screen_y, fovea_center_x, fovea_center_y)

        dist_x = abs(fovea_center_x - ball_screen_x)
        dist_y = abs(fovea_center_y - ball_screen_y)

        # Central threshold: If the ball is within roughly half a step of the chosen fovea center,
        # it's the best the discrete system could do.
        # Make this slightly larger than half the step size to be more lenient.
        leniency_pixels = 4
        central_threshold_x = (fov_step_size[0] / 2.0) + leniency_pixels  # e.g., 10/2 + 2 = 7 pixels
        central_threshold_y = (fov_step_size[1] / 2.0) + leniency_pixels

        # Within fovea threshold
        half_fov = fovea_size / 2.0

        if dist_x < central_threshold_x and dist_y < central_threshold_y:
            # This means the agent picked the fovea location that centers the ball
            # as best as possible given the discrete grid.
            return 0.01
        elif dist_x < half_fov and dist_y < half_fov:
            # Ball is within the fovea, but not optimally centered by the discrete choice
            return -0.05
        else:
            # Ball is outside the fovea
            return -0.1