import argparse, math, os
import numpy as np
import gym
import gym_custom

from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from behavior_cloning import Policy, BehaviorCloning, collect_trajs, collect_trajs_iid
import matlab.engine


eng = matlab.engine.start_matlab()

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_epochs', type=int, default=2000, metavar='N',
                    help='number of epochs (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--ckpt_freq', type=int, default=100, 
            help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
args = parser.parse_args()

env_name = args.env_name
env = gym.make(env_name)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = BehaviorCloning(args.hidden_size, env.observation_space.shape[0], env.action_space)
#params = torch.load('ckpt_QPendulum-v0/reinforce-1500.pkl')
expert_mpc = eng.get_pendulum_mpc()

def expert(x):
    x = matlab.double(x.reshape(2,1).tolist())
    return np.array([eng.evaluate_mpc(expert_mpc, x)])

#X, U = collect_trajs(env, expert, nu=0.0, n_traj=100, T=20)
print('collecting MPC expert data')
X, U = collect_trajs_iid(env, expert, N=1000, nu=0.0)
agent.set_expert_data(X, U)

dir = 'models/' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_epoch in range(args.num_epochs):
    IL_loss = agent.update_parameters()
    if i_epoch%args.ckpt_freq == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'IL_mpc_'+str(i_epoch)+'.pkl'))
    print("Epoch: {}, IL loss: {}".format(i_epoch, IL_loss))
env.close()