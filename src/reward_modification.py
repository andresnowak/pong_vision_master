import gymnasium as gym
import numpy as np


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
