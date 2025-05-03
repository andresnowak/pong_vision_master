from active_gym import AtariBaseEnv, AtariEnvArgs, AtariFixedFovealEnv

print('sodijfsiof')


env_args = AtariEnvArgs(
    game="pong",
    seed=42,
    obs_size=(84, 84),
    fov_size=(30, 30),  # the partial observation size
    fov_init_loc=(0, 0),  # the initial partial observation location
    sensory_action_mode="absolute",  # change the observation location by abs coordinates
    record=False,  # it integrates recording, if needed
    resize_to_full=True
)
env = AtariFixedFovealEnv(env_args)
obs, info = env.reset()
for i in range(5):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    # action is in the form of gymnasium.spaces.Dict
    # {"motor_action": np.ndarray, "sensory_action": np.ndarray}

    print(obs)


# this is needed python -m atari_py.import_roms /opt/homebrew/Caskroom/miniforge/base/envs/visual_intelligence_project_2/lib/python3.11/site-packages/AutoROM/roms

# and to install autorom and accept license to install the roms
# pip install autorom
# AutoROM --accept-license

# we also need cmake with conda so as to be able to install active-gym