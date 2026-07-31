import random

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

import itertools

from dqn import DQN
from memory import Memory

import torch

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)

env = AtariPreprocessing(
    env,
    screen_size=84,
    grayscale_obs=True,
    frame_skip=4,
    scale_obs=True
)

env = FrameStackObservation(env, 4)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self):
        # later I'll move these into a yaml file
        self.epsilon = 1
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.05
        self.maxlen = 100000

        self.memory = Memory(maxlen=self.maxlen)

    def run(self):
        dqn = DQN(6).to(device)

        for episode in itertools.count():
            state, info = env.reset()
            state = torch.from_numpy(state).to(device)

            terminated = False
            episode_reward = 0

            while not terminated:

                if random.random() < self.epsilon:
                    # random action
                    action = env.action_space.sample()
                else:
                    action = dqn(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                action = torch.tensor([action], dtype=torch.float32).to(device)
                new_state = torch.from_numpy(new_state).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                terminated = torch.tensor(terminated, dtype=torch.float32).to(device)

                self.memory.append((state, action, new_state, reward, terminated))

                state = new_state

                episode_reward += reward

                # decrease epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


if __name__ == "__main__":
    agent = Agent()
    agent.run()
