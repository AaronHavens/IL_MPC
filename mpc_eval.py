import torch
from behavior_cloning import Policy
import gym
import gym_custom
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
from tqdm import tqdm
eng = matlab.engine.start_matlab()

env = gym.make('LinearPendulum-v0')
env_expert = gym.make('LinearPendulum-v0')
params = torch.load('models/LinearPendulum-v0/IL_mpc_19900.pkl')
model_expert, S, AG, BG = eng.get_pendulum_mpc(nargout=4)

model = Policy(128, 2, env.action_space)
model.load_state_dict(params)
model.eval()

def MPC_eval(u_action, model, N):
	x = np.linspace(-2.5, 2.5, N)
	y = np.linspace(-6, 6, N)
	xv, yv = np.meshgrid(x, y)
	U = np.zeros((N,N))
	pbar = tqdm(total=int(N*N))
	for i in range(N):
		for j in range(N):
			x_t = np.array([xv[i,j], yv[i,j]])
			U[i, j] = u_action(x_t, model)[0]
			pbar.update(1)

	return xv, yv, U

def get_action(x, pi):
	x = torch.from_numpy(x).float()
	with torch.no_grad():
		u = pi(x)
	return u.detach().numpy()

def get_action_projection(x, pi, S, AG, BG):
	x_torch = torch.from_numpy(x).float()
	with torch.no_grad():
		u = pi(x_torch)
	u = u.detach().numpy()
	u_ = matlab.double(u.reshape(1,1).tolist())
	x_ = matlab.double(x.reshape(2,1).tolist())
	u_proj = np.array([eng.Project(S, AG, BG, u_, x_)], dtype=np.float32)
	return u_proj, u

def get_mpc_action(x,pi):
	x = matlab.double(x.reshape(2,1).tolist())
	return np.array([eng.evaluate_mpc(pi, x)])



#xv_e, yv_e, U_e = MPC_eval(get_mpc_action, model_expert, 40)
#xv, yv, U = MPC_eval(get_action, model, 100)

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(xv, yv, U, label='IL')
#ax.plot_surface(xv_e, yv_e, U_e, label='expert')
#ax.scatter3D(xv, yv, U, label='IL')
#ax.scatter3D(xv_e, yv_e, U_e, label='expert')
#plt.legend()
plt.show()
T = 200
x_hist = np.zeros((T, 2))
x_hist_expert = np.zeros((T, 2))
u_hist = np.zeros((T,1))
u_hist_expert = np.zeros((T, 1))
x = env.reset()
_ = env_expert.reset()
env_expert.env.state = np.copy(x)
x_expert = np.copy(x)

for i in range(T):
	x_hist[i,:] = x
	x_hist_expert[i,:] = x_expert
	u, u_ = get_action_projection(x, model, S, AG, BG)
	u_expert = get_mpc_action(x_expert, model_expert)
	u_hist[i,:] = u
	u_hist_expert[i,:] = u_expert
	print(u_expert, u, u_)
	x,r,_,done = env.step(u)
	x_expert,r,_,done_expert = env_expert.step(u_expert)
	if done or done_expert:
		print('done')
		break

# plt.plot(x_hist[:,0],label='position',c='r')
# plt.plot(x_hist[:,1],label='velocity',c='b')
# plt.plot(x_hist_expert[:,0],label='position (expert)',c='r', linestyle='--')
# plt.plot(x_hist_expert[:,1],label='velocity (expert)',c='b', linestyle='--')
plt.plot(u_hist[:,0],label='learned',c='r')
plt.plot(u_hist_expert[:,0],label='expert',c='b')
plt.legend()
plt.show()
