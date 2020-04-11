import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argh
from tqdm import tqdm
from typing import Any

# from collections import deque
from dataclasses import dataclass
from random import sample, random

import wandb


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, obs):
        # obs shape is (N, 4)
        q_vals = self.model(obs)

        # q_vals shape (N, 2)
        return q_vals.max(-1)[1]


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.001):
        super(Model, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_shape[0], 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, num_actions),
                )
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)


# Refactor the ReplayBuffer to make it more efficient
# by replacing the `deque` with an array keeping the index of elements.
# Instead an append we use a rolling index mechanism.
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        # self.buffer = deque(maxlen=buffer_size)
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, sars):
        # self.buffer.append(sars)
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_saples):
        assert num_saples < min(self.idx, self.buffer_size)
        # if num_saples > min(self.idx, self.buffer_size):
        #     import ipdb; ipdb.set_trace()
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_saples)
        return sample(self.buffer, num_saples)


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    cur_states = torch.stack(([torch.Tensor(s.state)
                             for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]).to(device)
                            for s in state_transitions]))
    mask = torch.stack(([torch.Tensor([0])
                         if s.done else torch.Tensor([1])
                         for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state)
                              for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(torch.LongTensor(actions),
                                num_actions).to(device)

    loss = ((rewards + mask[:, 0] * qvals_next * gamma - torch.sum(
            qvals * one_hot_actions, -1))**2).mean()
    loss.backward()
    model.opt.step()
    return loss


def main(name, test=False, chkp=None, device='cuda'):
    if not test:
        wandb.init(project="dqn-pytorch", name=name)
    # memory_size = 500000
    min_rb_size = 20000
    sample_size = 750
    lr = 0.001

    # eps_min = 0.01
    eps_decay = 0.999999

    env_steps_before_train = 100
    tgt_model_update = 500

    env = gym.make('CartPole-v1')
    last_obs = env.reset()

    m = Model(env.observation_space.shape,
              env.action_space.n, lr=lr).to(device)
    if chkp is not None:
        m.load_state_dict(torch.load(chkp))
    tgt = Model(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    steps_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)

            tq.update(1)

            eps = eps_decay**(step_num)
            if test:
                eps = 0

            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_obs).to(
                    device)).max(-1)[-1].item()

            obs, reward, done, info = env.step(action)
            rolling_reward += reward

            reward = reward * 0.1

            rb.insert(Sarsd(last_obs, action, reward, obs, done))

            last_obs = obs

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                obs = env.reset()

            steps_since_train += 1
            step_num += 1

            if (
                    (not test)
                    and rb.idx > min_rb_size
                    and steps_since_train > env_steps_before_train
                    ):
                loss = train_step(m, rb.sample(sample_size),
                                  tgt, env.action_space.n, device)
                wandb.log(
                        {
                            'loss': loss.detach().cpu().item(),
                            'eps': eps,
                            'avg_reward': np.mean(episode_rewards)
                            },
                        step=step_num)

                episode_rewards = []
                steps_since_tgt += 1
                if steps_since_tgt > tgt_model_update:
                    print("Updating target model")
                    update_tgt_model(m, tgt)
                    steps_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")
                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == '__main__':
    argh.dispatch_command(main)
    # main(True, "models/8298464.pth")
    # main(True, "models/6518513.pth")
    # main(test=False)
