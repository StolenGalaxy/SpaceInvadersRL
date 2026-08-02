import os
from datetime import datetime
import random

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

import itertools

from dqn import DQN
from memory import Memory

import torch
from torch import nn

import argparse

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, train):
        self.training = train

        # later I'll move these into a yaml file
        self.epsilon = 1
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.05
        self.maxlen = 100000
        self.batch_size = 32
        self.learning_rate_alpha = 0.0001
        self.network_sync_rate = 1000
        self.discount_factor_gamma = 0.99
        self.model_save_rate = 10

        self.memory = Memory(maxlen=self.maxlen)

    def run(self):
        # initialise game
        gym.register_envs(ale_py)
        if self.training:
            env = gym.make("ALE/SpaceInvaders-v5",
                           frameskip=1)  # frameskip=1 disables frameskips as the wrapper performs them
        else:
            env = gym.make("ALE/SpaceInvaders-v5",
                           frameskip=1, render_mode="human")

        env = AtariPreprocessing(
            env,
            screen_size=84,
            grayscale_obs=True,
            frame_skip=4,
            scale_obs=True
        )

        env = FrameStackObservation(env, 4)


        self.policy_dqn = DQN(output_size=6).to(device)

        if not self.training:
            self.policy_dqn.load_state_dict(torch.load(f"{MODEL_DIR}/best.pt"))
            self.policy_dqn.eval()


        self.target_dqn = DQN(output_size=6).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_alpha)

        step = 0

        highest_reward = -9999999

        for episode in itertools.count():
            state, info = env.reset()
            state = torch.from_numpy(state).to(device)

            terminated = False
            episode_reward = 0

            while not terminated:
                step += 1

                if random.random() < self.epsilon and self.training:
                    # random action
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_dqn(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                action = torch.tensor([action], dtype=torch.float32).to(device)
                new_state = torch.from_numpy(new_state).to(device)
                reward = torch.tensor([reward], dtype=torch.float32).to(device)
                terminated = torch.tensor([terminated], dtype=torch.float32).to(device)

                self.memory.append((state, action, new_state, reward, terminated))

                state = new_state

                episode_reward += reward.item()

                # decrease epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


                # optimise
                if self.training:
                    if len(self.memory) > self.batch_size:
                        batch = self.memory.sample(self.batch_size)
                        self.optimise(batch)
                    if step >= self.network_sync_rate:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            if episode_reward > highest_reward:
                torch.save(self.policy_dqn.state_dict(), f"{MODEL_DIR}/best.pt")
                print(f"{datetime.now()} | Episode {episode} | New highest reward: {episode_reward} | Epsilon: {self.epsilon}")
                highest_reward = episode_reward
            if not episode % self.model_save_rate:
                torch.save(self.policy_dqn.state_dict(), f"{MODEL_DIR}/recent.pt")


    def optimise(self, mini_batch):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # .stack() combines tensors along a new dimension, .cat() does it along an existing dimension.
        # we need to create a new dimension because the first dimension is currently being used to group frames.
        states = torch.stack(states)

        actions = torch.cat(actions).long()
        new_states = torch.stack(new_states)
        rewards = torch.cat(rewards)
        terminations = torch.cat(terminations)


        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # ddqn
        best_actions = self.policy_dqn(new_states).argmax(dim=1)
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_gamma * self.target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()

        self.optimiser.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()

        # update weights/biases
        self.optimiser.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SpaceInvadersRL")
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    agent = Agent(args.train)
    agent.run()
