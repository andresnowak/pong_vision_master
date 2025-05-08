from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from active_gym import AtariBaseEnv, AtariEnvArgs, AtariFixedFovealEnv


from src.reward_modification import PongRamHitRewardWrapper


def make_no_vision_env(env_id: str, n_envs: int, seed: int, n_stack: int = 4):
    # env = make_vec_env(env_id, n_envs=n_envs, seed=seed, wrapper_class=PongRamHitRewardWrapper)
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    # env = Monitor(env)
    # vec_env_cls=SubprocVecEnv)
    return env



# make_atari_env already gives us the grayscale of the image (color is not necessary for this environments and also crops the image to only the necessary part of the game that is needed to play)
def make_full_vision_env(env_id: str, n_envs: int, seed: int):
    # oopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4) is already done by make_atari_env

    env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
    )

    # arranges the dimensions of the observations coming from the vectorized environment, changing them from the channels-last (N, H, W, C) format to the channels-first (N, C, H, W) format required by PyTorch CNNs.
    env = VecTransposeImage(env)

    return env


def make_fovea_env(env_name, seed, **kwargs):
    def thunk():
        env_args = AtariEnvArgs(game=env_name, seed=seed, obs_size=(84, 84), **kwargs)
        env = AtariFixedFovealEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
