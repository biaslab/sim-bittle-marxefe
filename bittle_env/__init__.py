from gymnasium.envs.registration import register
register(
    id="BittleBulletEnv-v0",
    entry_point="bittle_env.bittle_gym_env:BittleBulletEnv",
    max_episode_steps=1000,
    reward_threshold=15.0,
)
