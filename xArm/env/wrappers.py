import numpy as np
from numpy.random import randint
import os
import gym
from gym.wrappers import TimeLimit
from xArm.env.robot.registration import register_robot_envs
import xArm.utils as utils
from collections import deque
from mujoco_py import modder
import copy

ACTION_SPACE = {
    "lift":"xyzw",
    "peginsert":"xyzw",
    "shelfplacing":"xyzw",
    "push":"xyz",
    "reach":"xyz",
    "pegbox":"xyz",
}
def make_env(
        task_name,
        seed=0,
        episode_length=50,
        frame_stack=1,
        image_size=84,
        render=False,
        observation_type='state',
        domain_randomization=0,
):
    domain_name = "robot"
    """Make environment for experiments"""
    action_space = ACTION_SPACE[task_name]

    print("[make_env] type: ", observation_type)
    register_robot_envs(
        n_substeps=20,
        observation_type=observation_type,
        image_size=image_size,
        use_xyz=action_space.replace('w', '') == 'xyz')

    
    env_id = 'Robot' + task_name.capitalize() + '-v0'

    camera_list = ["camera_static"]
    env = gym.make(env_id, cameras=camera_list, render=render, observation_type=observation_type, use_xyz='xyz' in action_space)
    
    env.seed(seed)
    env.task_name = task_name
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = SuccessWrapper(env, any_success=True)
    env = ObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size, num_cameras=len(camera_list))
    env = ActionSpaceWrapper(env, action_space=action_space)
    env = FrameStack(env, frame_stack)
    if domain_randomization:
        env = DomainRandomizationWrapper(env, seed=seed)

    print(f'[make_env] {env_id} is created.')
    return env



class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        if len(shp) == 3:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(shp[0] * k,),
                dtype=env.observation_space.dtype
            )
        self._max_episode_steps = 1000 # hardcode

    def reset(self):
        obs, state_obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(),  reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env, any_success=True):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.any_success = any_success
        self.success = False

    def reset(self):
        self.success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.any_success:
            self.success = self.success or bool(info['is_success'])
        else:
            self.success = bool(info['is_success'])
        info['is_success'] = self.success
        return obs, reward, done, info


class MetaworldSuccessWrapper(gym.Wrapper):
    def __init__(self, env, any_success=True):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.any_success = any_success
        self.success = False

    def reset(self):
        self.success = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.any_success:
            self.success = self.success or bool(info['success'])
        else:
            self.success = bool(info['success'])
        info['is_success'] = info['success'] = self.success
        return obs, reward, done, info

class ObservationSpaceWrapper(gym.Wrapper):
    def __init__(self, env, observation_type, image_size, num_cameras):
        # assert observation_type in {'state', 'image'}, 'observation type must be one of \{state, image\}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size
        self.num_cameras = num_cameras


        if self.observation_type in ['image', 'state+image']:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * self.num_cameras, image_size, image_size),
                                                    dtype=np.uint8)

        elif self.observation_type in ['state', 'state+state']:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim,
                                                    dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs), obs['state'] if 'state' in obs else None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), obs['state'] if 'state' in obs else None, reward, done, info

    def _get_obs(self, obs_dict):
        obs = obs_dict['observation']
        if self.observation_type in ['image', "state+image"]:
            output = np.empty((3 * obs.shape[0], self.image_size, self.image_size), dtype=obs.dtype)
            for i in range(obs.shape[0]):
                output[3 * i: 3 * (i + 1)] = obs[i].transpose(2, 0, 1)
        elif self.observation_type in ['state', 'state+state']:
            output = obs_dict['observation']
        return output

class MetaworldObservationSpaceWrapper(gym.Wrapper):
    """
    Provide more modalities for metaworld env.
    
    Originally, metaworld gives state by default. Our wrapper gives robot state (gripper distance and position) and image obs as well.
    """

    def __init__(self, env, observation_type, image_size):
        assert observation_type in {'state', 'image', 'state+image', 'state+state'}, 'observation type must be one of \{state, image, state+image\}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size

        self.state_space_shape = (4,) # the state refers to robot state
        if self.observation_type in {'state', 'image'}:
            self.state_space_shape = None

        if self.observation_type in ['image', 'state+image']:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * 2, image_size, image_size),
                                                    dtype=np.uint8)
        elif self.observation_type in ['state', 'state+state']:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim,
                                                    dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._get_obs(obs)
        return obs_dict['observation'], obs_dict['robot_state']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # obs is `state` now. and we fetch other modalities by `_get_obs`
        obs_dict = self._get_obs(obs)
        return obs_dict['observation'], obs_dict['robot_state'], reward, done, info


    def _get_obs(self, obs):
        obs_dict = dict()
        # fetch img obs or state obs
        if self.observation_type in ['image', "state+image"]:
            img_obs_origin = self.render_obs(width=self.image_size, height=self.image_size)
            img_obs = np.empty((3 * 2, self.image_size, self.image_size), dtype=obs.dtype)
            img_obs[:3] = img_obs_origin[0].transpose(2, 0, 1)
            img_obs[3:] = img_obs_origin[1].transpose(2, 0, 1)
            obs_dict['observation'] = img_obs
        elif self.observation_type in ['state', 'state+state']:
            obs_dict['observation'] = obs

        # fetch robot state
        if self.observation_type in ['state+image', 'state+state']:
            obs_dict['robot_state'] = self.get_robot_state_obs()
        else:
            obs_dict['robot_state'] = None

        return obs_dict


class ActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, action_space):
        assert action_space in {'xy', 'xyz', 'xyzw'}, 'task must be one of {xy, xyz, xyzw}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.action_space_dims = action_space
        self.use_xyz = 'xyz' in action_space
        self.use_gripper = 'w' in action_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 + self.use_xyz + self.use_gripper,),
                                           dtype=np.float32)

    def step(self, action):
        assert action.shape == self.action_space.shape, 'action shape must match action space'
        action = np.array(
            [action[0], action[1], action[2] if self.use_xyz else 0, action[3] if self.use_gripper else 1],
            dtype=np.float32)
        return self.env.step(action)




class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, seed=None):

        # assert domain_name in {'reach', 'push', 'cloth'}, \
        #	'domain randomization only implemented for reach, push, cloth domains'
        gym.Wrapper.__init__(self, env)

        self.randomizations = {'camera', 'material', 'skybox', 'brightness'}
        # self.randomizations = {'camera', 'light', 'material', 'brightness'}
        # self.randomizations = {'camera',  'skybox', 'material', 'brightness'}
        self.sim = self.env.unwrapped.sim

        self.random_state = np.random.RandomState(seed)
        self.camera_name = "camera_static"
        self.cam_modder = modder.CameraModder(self.sim, random_state=self.random_state)
        self.light_name = 'light0'
        self.light_modder = modder.LightModder(self.sim, random_state=self.random_state)
        self.geom_names = ['tablegeom0', 'floorgeom0']
        self.material_modder = modder.MaterialModder(self.sim, random_state=self.random_state)
        self.texture_modder = modder.TextureModder(self.sim, random_state=self.random_state)
        self.brightness_std = 0.05


        self.record_inital_camera() # record camera pos


    def reset(self):
        self.traj_idx = 0
        if 'texture' in self.randomizations:
            self._randomize_texture()
        if 'camera' in self.randomizations:
            self._randomize_camera()
        if 'light' in self.randomizations:
            self._randomize_light()
        if 'material' in self.randomizations:
            self._randomize_material()
        if 'skybox' in self.randomizations and not 'texture' in self.randomizations:
            self._randomize_skybox()
        if 'brightness' in self.randomizations:
            self._randomize_brightness()
        obs, state= self.env.reset()
        return self._modify_obs(obs), state
    
    def change_traj_idx(self, idx):
        self.traj_idx = idx % self.traj_len
        
    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        return self._modify_obs(obs), state, reward, done, info

    def _modify_obs(self, obs):
        if len(obs[:].shape) > 1:
            return (np.clip(obs[:]/255. + np.ones_like(obs) * self.brightness_std, 0, 1)*255).astype(np.uint8)
        return obs

    def _randomize_texture(self):
        for name in self.geom_names:
            self.texture_modder.whiten_materials()
            self.texture_modder.set_checker(name, (255, 0, 0), (0, 0, 0))
            self.texture_modder.rand_all(name)
        self.texture_modder.rand_gradient('skybox')


    def get_new_position(self, camera_name):

        if camera_name not in ['camera_static', 'camera_dynamic']:
            raise Exception('Wrong Camera Name in Randomness Wrapper.')
        init_position = copy.deepcopy(self.init_camera_positions[camera_name])
        new_x = init_position [0]
        new_x += 0.2*(np.random.rand()-0.5)

        new_y = init_position [1]
        new_y += 0.2*(np.random.rand()-0.5)

        new_z = init_position [2] 
        new_z += 0.2*(np.random.rand()-0.5)
  
        return np.array([new_x,new_y,new_z])


    def get_new_quaternion(self, camera_name):            
        if camera_name not in ['camera_static', 'camera_dynamic']:
            raise Exception('Wrong Camera Name in Randomness Wrapper.')
        init_quaternion = copy.deepcopy(self.init_camera_quaternions[camera_name])
        new_quat = copy.deepcopy(init_quaternion)
        new_quat[0] += 0.1*(np.random.rand()-0.5)
        new_quat[1] += 0.1*(np.random.rand()-0.5)
        new_quat[2] += 0.1*(np.random.rand()-0.5)
        new_quat[3] += 0.1*(np.random.rand()-0.5)
        return new_quat

    def _randomize_camera(self):

        # get current camera position
        pos = self.cam_modder.get_pos(self.camera_name)
        new_pos = self.get_new_position(self.camera_name)
        self.cam_modder.set_pos(self.camera_name, new_pos)
            
        # get current camera rotation
        quat = self.cam_modder.get_quat(self.camera_name)
        new_quat = self.get_new_quaternion(self.camera_name)
        self.cam_modder.set_quat(self.camera_name, new_quat)


  
    def record_inital_camera(self):
        """
        Record new initialized camera.
        """
        self.init_camera_positions = {}
        self.init_camera_positions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_pos("camera_dynamic"))
        self.init_camera_positions["camera_static"] = copy.deepcopy(self.cam_modder.get_pos("camera_static"))

        self.init_camera_quaternions = {}
        self.init_camera_quaternions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_quat("camera_dynamic"))
        self.init_camera_quaternions["camera_static"] = copy.deepcopy(self.cam_modder.get_quat("camera_static"))
        
   
    def _randomize_brightness(self):
        self.brightness_std = self._add_noise(0, 0.05)

    def _randomize_light(self):
        self.light_modder.set_ambient(self.light_name, self._uniform([0.2, 0.2, 0.2]))
        self.light_modder.set_diffuse(self.light_name, self._uniform([0.8, 0.8, 0.8]))
        self.light_modder.set_specular(self.light_name, self._uniform([0.3, 0.3, 0.3]))
        self.light_modder.set_pos(self.light_name, self._add_noise([0, 0, 4], 0.5))
        self.light_modder.set_dir(self.light_name, self._add_noise([0, 0, -1], 0.25))
        self.light_modder.set_castshadow(self.light_name, self.random_state.randint(0, 2))

    def _randomize_material(self):
        for name in self.geom_names:
            self.material_modder.rand_all(name)

    def _randomize_skybox(self):
        self.texture_modder.rand_gradient('skybox')
        geom_id = self.sim.model.geom_name2id('floorgeom0')
        mat_id = self.sim.model.geom_matid[geom_id]
        self.sim.model.mat_rgba[mat_id] = np.clip(self._add_noise([0.2, 0.15, 0.1, 1], [0.1, 0.2, 0.2, 0]), 0, 1)

    def _uniform(self, default, low=0.0, high=1.0):
        if isinstance(default, list):
            default = np.array(default)
        if isinstance(low, list):
            assert len(low) == len(default), 'low and default must be same length'
            low = np.array(low)
        if isinstance(high, list):
            assert len(high) == len(default), 'high and default must be same length'
            high = np.array(high)
        return np.random.uniform(low=low, high=high, size=len(default))

    def _add_noise(self, default, std):
        if isinstance(default, list):
            default = np.array(default)
        elif isinstance(default, (float, int)):
            default = np.array([default], dtype=np.float32)
        if isinstance(std, list):
            assert len(std) == len(default), 'std and default must be same length'
            std = np.array(std)
        return default + std * self.random_state.randn(len(default))
