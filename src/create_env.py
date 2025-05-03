from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
import gymnasium as gym

from src.reward_modification import PongHitRewardWrapper

def make_no_vision_env(env_id: str, n_envs: int, seed: int, n_stack: int = 4):
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    # env = Monitor(env)
    env = VecFrameStack(env, n_stack=n_stack)

    return env


# make_atari_env already gives us the grayscale of the image (color is not necessary for this environments and also crops the image to only the necessary part of the game that is needed to play)
def make_full_vision_env(env_id: str, n_envs: int, seed: int):
    env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
    )

    env = VecTransposeImage(env)

    return env
