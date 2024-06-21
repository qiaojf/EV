import copy
import random,time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim:int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size,action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):#存储trainsition=【state，action,reward,next_state,done】
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

class Actor(nn.Module):#策略网络
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        init_w: float = 3e-3,
    ):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()
        # action = 2*(out - 0.5)
        return action
    
class Critic(nn.Module):#价值网络
    def __init__(
        self, 
        in_dim: int, 
        init_w: float = 3e-3,
    ):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value

class DDPGAgent:
    def __init__(
        self,
        env,
        memory_size: int,
        batch_size: int,

        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 5000,
    ):
        obs_dim = env.state.size
        action_dim = env.actions_dim
        self.env = env
        self.memory = ReplayBuffer(obs_dim,action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
                
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # print(self.device)

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.transition = list()
        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:#生成动作
        self.env.update_ev(self.total_step)
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_sample()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state.flatten()).to(self.device)
            ).detach().cpu().numpy()
        res = self.env.check_action(selected_action)
        if not res[0]:
            while True:
                selected_action = self.env.action_sample()
                res = self.env.check_action(selected_action)
                if res[0]:
                    break
        print('action:',selected_action)
        self.transition = [state, selected_action]
        return selected_action
    
    def step(self, action,state):#更新执行动作后的状态
        next_state, reward, done = self.env.step(action,state,self.total_step)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        device = self.device  # for shortening the following lines
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks

        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._target_soft_update()
        return actor_loss.data, critic_loss.data

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state = self.env.init_state()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0
        for self.total_step in range(1, num_frames + 1):
            print('step:',self.total_step)
            # for ev in self.env.ev_list:
            #     if ev.id <= 40:
            #         pass
            #     else:
            #         pass
            action = self.select_action(state,)
            next_state, reward, done = self.step(action,state,)
            state = next_state
            score += reward
            if done:         
                state = self.env.init_state()
                scores.append(score)
                score = 0
            if (
                len(self.memory) >= self.batch_size 
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
        print('total_reward:',score)
        self.env.wb.save('程序数据集/my_data.xlsx')
        time.sleep(5)
    
    def _target_soft_update(self):
        tau = self.tau
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)



