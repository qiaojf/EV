import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:#保存数据，以备后续训练使用
    def __init__(self, obs_dim: int, action_dim:int,  size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size,action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):              #存储数据
        self.obs_buf[self.ptr] = obs.flatten()
        self.next_obs_buf[self.ptr] = next_obs.flatten()
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
    def __len__(self) -> int:
        return self.size

class Network(nn.Module):#DQN网络
    def __init__(self, in_dim: int, out_dim):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, out_dim),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out= self.layers(x)
        # action = self.layers(x).tanh()
        out = self.layers(x)

        return out.reshape(-1,60,201)

class DQNAgent:
    def __init__(
        self, 
        env,
        memory_size: int,
        batch_size: int,
        target_update: int=100,
        epsilon_decay: float=0.001,
        # seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        obs_dim = env.state.size
        action_dim = env.actions_dim
        self.env = env
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon#epsilon初始值
        self.epsilon_decay = epsilon_decay#每次动作衰减
        # self.seed = seed
        self.max_epsilon = max_epsilon  #最大值
        self.min_epsilon = min_epsilon #最小值
        self.target_update = target_update
        self.gamma = gamma
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # print(self.device)

        self.dqn = Network(obs_dim, action_dim*201).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim*201).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()
        self.total_step = 0
        self.is_test = False
    def select_action(self, state: np.ndarray):
        self.env.update_ev(self.total_step)
        act = [i/100 for i in range(-100,101,1)]
        if self.epsilon > np.random.random():
            selected_action = self.env.action_sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state.flatten()).to(self.device)
            )
            selected_action = selected_action.argmax(dim=-1)
            selected_action = selected_action.reshape(self.env.actions_dim).detach().cpu().numpy()
            selected_action = [act[x] for x in selected_action]

        res = self.env.check_action(selected_action)
        if not res[0]:
            while True:
                selected_action = self.env.action_sample()
                res = self.env.check_action(selected_action)
                if res[0]:
                    break
        print('action',selected_action)
        action = [act.index(x) for x in selected_action]
        # print('action:',action)
        self.transition = [state, action]
        return selected_action

    def step(self, action,state):
        next_state, reward, done = self.env.step(action,state,self.total_step)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state = self.env.init_state()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for self.total_step in range(1, num_frames + 1):
            print('step:',self.total_step)
            action = self.select_action(state,)
            next_state, reward, done = self.step(action,state,)
            state = next_state
            score += reward

            if done:
                state = self.env.init_state()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)

            if update_cnt % self.target_update == 0:
                self._target_hard_update()
        self.env.wb.save('程序数据集/my_data.xlsx')
        print('total_reward:',score)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]):
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # curr_q_value = self.dqn(state).gather(-1, action)
        curr_q_value = self.dqn(state).gather(-1, action.reshape(self.batch_size,self.env.actions_dim, -1))
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=-1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value.reshape(self.batch_size,self.env.actions_dim) * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value.reshape(self.batch_size,self.env.actions_dim), target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())









