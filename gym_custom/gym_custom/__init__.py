from gym.envs.registration import register
register(id='QPendulum-v0',
        entry_point='gym_custom.envs:QPendulumEnv',
        max_episode_steps=1000,
)
register(id='LinearPendulum-v0',
        entry_point='gym_custom.envs:LinearPendulumEnv',
        max_episode_steps=1000,
)

