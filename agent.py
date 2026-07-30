import random

import gymnasium as gym
import ale_py

import itertools

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

class Agent:
    def __init__(self):
        self.epsilon = 1
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.05

    def run(self):
        for episode in itertools.count():
            state = env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                if random.random() < self.epsilon:
                    # random action
                    action = env.action_space.sample()
                else:
                    # choose action
                    pass

                state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                # decrease epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


if __name__ == "__main__":
    agent = Agent()
    agent.run()
