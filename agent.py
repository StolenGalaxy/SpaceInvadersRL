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
    def __init__(self, arguments):
        self.training = arguments.train
        self.saved_model_file = arguments.r

        # later I'll move these into a yaml file
        self.epsilon = 1
        self.epsilon_decay = 0.999999693
        self.epsilon_min = 0.05
        self.maxlen = 100000
        self.batch_size = 32
        self.learning_rate_alpha = 0.0001
        self.network_sync_rate = 1000
        self.discount_factor_gamma = 0.99
        self.model_save_rate = 400
        self.optimise_frequency = 4 # after every x steps the model will optimise. why not optimise after every step?
                                    # in pong, we could optimise every step as we were only passing data through
                                    # a few linear layers. here we are using complex convolutional layers that are
                                    # much more computationally expensive.
                                    # in addition, the difference between pixels after a single step is a lot less
                                    # noticable than the difference between the coordinates we gave in pong.
        self.warmup_steps = 20000

        self.memory = Memory(maxlen=self.maxlen)
        self.store_memory = True

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
            scale_obs=False
        )

        env = FrameStackObservation(env, 4)


        self.policy_dqn = DQN(output_size=6).to(device)

        if not self.training:
            if self.saved_model_file:
                print(f"Loading {MODEL_DIR}/{self.saved_model_file}")
                self.policy_dqn.load_state_dict(torch.load(f"{MODEL_DIR}/{self.saved_model_file}"))
            self.policy_dqn.eval()
        else:
            self.policy_dqn.train()

        self.target_dqn = DQN(output_size=6).to(device)

        self.loss_fn = nn.HuberLoss(delta=1) # huber should help reduce the influence of outliers such as the mothership
        self.optimiser = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_alpha)

        step = 0

        highest_reward = -9999999

        episode_number = 0

        if self.training and self.saved_model_file:
            print(f"Loading {MODEL_DIR}/{self.saved_model_file}")
            checkpoint = torch.load(f"{MODEL_DIR}/{self.saved_model_file}", weights_only=False)
            self.policy_dqn.load_state_dict(checkpoint["policy_state"])
            self.optimiser.load_state_dict(checkpoint["optimiser_state"])

            self.epsilon = checkpoint["epsilon"]
            episode_number = checkpoint["episode"]
            step = checkpoint["step"]
            highest_reward = checkpoint["highest_reward"]

            if "memory" in checkpoint:
                self.memory = checkpoint["memory"]



        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())


        for episode in itertools.count(episode_number):
            state, info = env.reset()
            state = torch.from_numpy(state)

            lives = info["lives"]
            terminated = False
            truncated = False
            episode_reward = 0

            while not terminated and not truncated:
                step += 1

                if random.random() < self.epsilon and self.training:
                    # random action
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_dqn((state / 255).unsqueeze(0).to(device)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                action = torch.tensor([action], dtype=torch.float32)
                new_state = torch.from_numpy(new_state)


                # ensure dying causes a negative reward and a "terminated" flag (even though the game will continue
                # running)
                new_lives = info["lives"]
                life_lost = new_lives < lives
                if life_lost:
                    reward = -100
                    lives = new_lives

                reward = torch.tensor([reward], dtype=torch.float32)

                fixed_negative_reward = terminated or life_lost # if the agent dies or loses a life, we will force the
                # bellman equation to give it a negative reward no matter what the future value of rewards could be
                # to ensure it sees losing a life as negative

                fixed_negative_reward = torch.tensor([fixed_negative_reward], dtype=torch.float32)

                if self.training:
                    self.memory.append((state, action, new_state, reward, fixed_negative_reward))

                state = new_state

                episode_reward += reward.item()

                # decrease epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


                # optimise
                if self.training:
                    if len(self.memory) > self.warmup_steps and not step % self.optimise_frequency:
                        batch = self.memory.sample(self.batch_size)
                        self.optimise(batch)
                    if not step % self.network_sync_rate:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            if episode_reward > highest_reward:
                print(f"{datetime.now()} | Episode {episode} | New highest reward: {episode_reward} | Epsilon: {self.epsilon}")
                highest_reward = episode_reward

                if self.training:
                    torch.save(self.policy_dqn.state_dict(), f"{MODEL_DIR}/best.pt")

            elif not episode % 50:
                print(f"{datetime.now()} | Episode {episode} | Epsilon: {self.epsilon}")

            if not episode % self.model_save_rate and self.training:
                # save data for resuming training later
                checkpoint = {
                    "episode": episode,
                    "step": step,
                    "highest_reward": highest_reward,
                    "epsilon": self.epsilon,
                    "policy_state": self.policy_dqn.state_dict(),
                    "optimiser_state": self.optimiser.state_dict()
                }

                if self.store_memory:
                    checkpoint["memory"] = self.memory

                print(f"{datetime.now()} | Saving")
                torch.save(checkpoint, f"{MODEL_DIR}/training.pt.tmp") # keep the original while writing in case of
                # interruption
                os.replace(f"{MODEL_DIR}/training.pt.tmp", f"{MODEL_DIR}/training.pt") # then rename
                print(f"{datetime.now()} | Finished saving")


    def optimise(self, mini_batch):
        states, actions, new_states, rewards, fixed_negative_reward = zip(*mini_batch)

        # .stack() combines tensors along a new dimension, .cat() does it along an existing dimension.
        # we need to create a new dimension because the first dimension is currently being used to group frames.
        states = torch.stack(states).to(device).div(255)
        new_states = torch.stack(new_states).to(device).div(255)

        actions = torch.cat(actions).long().to(device)
        rewards = torch.cat(rewards).to(device).div(100) # scale rewards to keep gradients stable
        fixed_negative_reward = torch.cat(fixed_negative_reward).to(device)


        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # ddqn
        with torch.no_grad():
            best_actions = self.policy_dqn(new_states).argmax(dim=1)
            target_q = rewards + (1 - fixed_negative_reward) * self.discount_factor_gamma * self.target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()

        self.optimiser.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()

        # update weights/biases
        self.optimiser.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SpaceInvadersRL")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("-r")

    args = parser.parse_args()

    agent = Agent(args)
    agent.run()
