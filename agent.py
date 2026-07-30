import gymnasium as gym
import ale_py

import itertools

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5")

for episode in itertools.count():
    state = env.reset()
    terminated = False
    episode_reward = 0

    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
