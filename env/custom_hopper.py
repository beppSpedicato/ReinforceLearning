"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass)    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7

    def udr_sample_parameters(self, delta, log: False):
        """Sample masses according to a domain randomization distribution"""
        old_masses = np.copy(self.sim.model.body_mass)
        body_masses = self.original_masses
        parameters = body_masses[2:]
        
        # get upper and lower using a delta parameter for masses
        upper_bounds = (1 + delta)*parameters
        lower_bounds = (1 - delta)*parameters

        if log:
            print(upper_bounds, lower_bounds)

        # calculate uniform parameters
        new_parameters = np.random.uniform(lower_bounds, upper_bounds)
        self.sim.model.body_mass[2:] = new_parameters
        
        if log:
            print("Old parameters: ", old_masses)
            print("New parameters: ", self.sim.model.body_mass)
            
        return

    def get_masses_ranges(self, delta):
        body_masses = self.original_masses
        parameters = body_masses[2:]
        upper_bounds = (1 + delta)*parameters
        lower_bounds = (1 - delta)*parameters

        return list(zip(lower_bounds, upper_bounds))

        

    def beta_sample_parameters(self, a, b, delta, log: False):
        """Sample masses according to a domain randomization distribution"""
        old_masses = np.copy(self.sim.model.body_mass)
        body_masses = self.original_masses
        parameters = body_masses[2:]
        
        # get upper and lower using a delta parameter for masses
        upper_bounds = (1 + delta)*parameters
        lower_bounds = (1 - delta)*parameters

        if log:
            print(upper_bounds, lower_bounds)
            
        samples = np.random.beta(a, b)
        new_parameters = samples * (upper_bounds - lower_bounds) + lower_bounds
            
        self.sim.model.body_mass[2:] = new_parameters
        
        if log:
            print("Old parameters: ", old_masses)
            print("New parameters: ", self.sim.model.body_mass)
            
        return self.sim.model.body_mass[2:]

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

