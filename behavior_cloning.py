import sys
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

def collect_trajs_iid(env, policy, N=100, nu=0.0):
    n = len(env.observation_space.low)
    m = len(env.action_space.low)
    X = []
    U = []
    k = 0
    for i in tqdm(range(N)):
        x1 = np.random.uniform(-2.5, 2.5)
        x2 = np.random.uniform(-6, 6)
        xt = np.array([x1, x2])
        ut = policy(xt)
        if not np.isnan(ut):
            X.append(xt)
            U.append(ut + np.random.normal(0, nu, size=(m,)))
            k += 1
    return np.asarray(X).reshape(k, n), np.asarray(U).reshape(k, m)

def collect_trajs(env, policy, n_traj=5, T=20, nu=0.0):
    n = len(env.observation_space.low)
    m = len(env.action_space.low)
    N = int(n_traj*T)
    X = np.zeros((N, n))
    U = np.zeros((N, m))
    k = 0
    for i in tqdm(range(n_traj)):
        xt = env.reset()
        for j in range(T):
            ut = policy(xt)
            ut = ut + np.random.normal(0,nu,size=(m,))
            X[k] = xt
            U[k] = ut
            xt,r,done,_ = env.step(ut)
            k += 1
    return X, U


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        bias=True
        self.linear1 = nn.Linear(num_inputs, hidden_size, bias=bias)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear3 = nn.Linear(hidden_size, num_outputs, bias=bias)
    def forward(self, inputs):
        x = inputs
        zero = torch.zeros_like(inputs)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        u = self.linear3(x)
        offset = self.linear3(torch.tanh(self.linear2(torch.tanh(self.linear1(zero)))))
        return u - offset


class BehaviorCloning:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.model.train()

    def set_expert_data(self, X, U):
        self.x_data = torch.from_numpy(X).cuda().float()
        self.u_data = torch.from_numpy(U).cuda().float()

    def select_action(self, state):
        action = self.model(Variable(state).cuda())        
        return action

    def update_parameters(self):
        u_hat = self.model(self.x_data)
        IL_loss = torch.nn.MSELoss()
        loss = 0
        IL_e = IL_loss(u_hat, self.u_data)
        loss += IL_e
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
        return IL_e.item()