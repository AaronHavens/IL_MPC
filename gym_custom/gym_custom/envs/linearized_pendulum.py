import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate as integrate

class LinearPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_torque=2.
        self.max_speed=100.
        self.dt=.02
        self.viewer = None

        high = np.array([np.inf, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self,u):
        done = False

        th, thdot = self.state # th := theta

        g = 10.
        m = 0.15
        l = 0.5
        mu = 0.05
        dt = self.dt
    

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        # def f(t, y):
        #     theta, theta_dot = y
        newth = th + dt*thdot
        newthdot = g/l*dt*th + (1-mu/(m*l**2)*dt)*thdot + 1/(m*l**2)*dt*u
        costs = 10*th**2 + 0.01*thdot**2 + 0.01*u**2
        #costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        if abs(th) > np.pi-0.001: done = True 
        return np.asarray(self.state), -costs, done, {}

    def reset(self):
        #sign = [-1, 1]
        #theta = np.pi - self.np_random.uniform(low=0.1, high=1./2.*np.pi)
        #theta = np.random.uniform(low=np.pi/3, high=2.*np.pi-np.pi/3)
        #theta = self.np_random.choice(sign)*theta
        #theta_dot = self.np_random.uniform(-0.5, 0.5)
        #self.state = np.array([theta, theta_dot])
        high = np.array([np.pi/3, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return np.asarray(self.state)

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
