import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gym.wrappers import TimeLimit
from xArm.env.robot.registration import register_robot_envs
import xArm.utils as utils
from collections import deque
from mujoco_py import modder
import copy
import math
import xArm.rot_utils as rot_utils
from termcolor import colored
from gym.spaces.box import Box

def make_env(
        task_name,
        seed=0,
        episode_length=50,
        n_substeps=20,
        frame_stack=1,
        image_size=84,
        cameras="static",
        render=False,
        observation_type='image',
        action_space='xyzw',
        camera_move_range=30,
        domain_randomization=0,
        num_static_cameras=1,
        num_dynamic_cameras=1,
        embedding_model=None,
        cfg=None,
):
    domain_name = "robot"
    """Make environment for experiments"""
    assert action_space in {'xy', 'xyz', 'xyzw'}, f'unexpected action space "{action_space}"'

    print("[make_env] type: ", observation_type)
    register_robot_envs(
        n_substeps=n_substeps,
        observation_type=observation_type,
        image_size=image_size,
        use_xyz=action_space.replace('w', '') == 'xyz')
    


    assert cameras in ['static', 'dynamic', 'static+dynamic'], "Please specify cameras as static or dynamic or static+dynamic."

    
    env_id = 'Robot' + task_name.capitalize() + '-v0'

    camera_list = []
    if 'static' in cameras:
        camera_list += ["camera_static"]
        camera_list += ["camera_static{}".format(i+1) for i in range(num_static_cameras-1)]
    if 'dynamic' in cameras:
        camera_list += ["camera_dynamic"]
        camera_list += ["camera_dynamic{}".format(i+1) for i in range(num_dynamic_cameras-1)]
    env = gym.make(env_id, cameras=camera_list, render=render, observation_type=observation_type, use_xyz='xyz' in action_space)
    
    env.seed(seed)
    env.task_name = task_name
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = SuccessWrapper(env, any_success=True)
    env = ObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size, num_cameras=len(camera_list))
    env = ActionSpaceWrapper(env, action_space=action_space)
    if embedding_model is not None:
        env = EmbeddingWrapper(env, embedding_model)
    env = FrameStack(env, frame_stack)
    env = DynamicCameraWrapper(env, domain_name=domain_name, camera_move_range=camera_move_range, \
                    num_static_cameras=num_static_cameras, \
                    num_dynamic_cameras=num_dynamic_cameras, \
                    seed=seed)
    if domain_randomization:
        env = DomainRandomizationWrapper(env, seed=seed)
    env = CameraPosWrapper(env)
    env = GripperWrapper(env)

    print(f'[make_env] {env_id} is created.')
    return env


class FormatAlignWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, state = self.env.reset()
        return obs, state, None

# ==============================================================================
# EMBEDDING WRAPPER
# ==============================================================================

class EmbeddingWrapper(gym.Wrapper):
    """
    This wrapper places a convolution model over the observation.
    The original observation shape must be (H, W, n * 3), where n is the number
    of frames per observation.
    If n > 1, each frame will pass through the convolution separately.
    The outputs will then be stacked.

    Args:
        env (gym.Env): the environment,
        embedding (torch.nn.Module): neural network defining the observation
            embedding.
    """
    def __init__(self, env, embedding):
        super(EmbeddingWrapper, self).__init__(env)
        in_channels = env.observation_space.shape[0]
        assert in_channels % 3 == 0,  \
                """ Only RGB images are supported.
                    Be sure that observation shape is (H, W, n * 3),
                    where n is the number of frames per observation. """

        self.in_channels = 3
        self.n_frames = in_channels // 3

        self.embedding = embedding
        self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.embedding.output_dim * self.n_frames,)
        )

    def reset(self):
        obs, state = self.env.reset()
        return self.observation(obs), state

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        return self.observation(obs), state, reward, done, info

    def observation(self, observation):
        # if self.n_frames > 1, each passes through the embedding separately
        # observation = np.stack(np.split(observation, self.n_frames, axis=-1)) # (H, W, self.n_frames * 3) -> (self.n_frames, H, W, 3)
        observation = torch.from_numpy(observation.copy()).div(255.0).unsqueeze(0).cuda()
        with torch.no_grad():
            observation = self.embedding(observation)
        if isinstance(observation, torch.Tensor):
            observation = observation.cpu().data.numpy()[0]
        return observation
    
    def update_embedding(self, embedding):
        self.embedding = embedding




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
        return self._get_obs(), state_obs

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), state,  reward, done, info

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



class CameraPosWrapper(gym.Wrapper):
    """
    Record camera pos and return
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        obs, state = self.env.reset()
        
        info = {}
        info["camera_RT"] = self.get_camera_RT()
        info["camera_intrinsic"] = self.get_camera_intrinsic()
        info["camera_extrinsic"] = self.get_camera_extrinsic()
        info["focal_length"] = self.get_focal_length()

        return obs, state, info

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
    
        info["camera_RT"] = self.get_camera_RT()
        info["camera_intrinsic"] = self.get_camera_intrinsic()
        info["camera_extrinsic"] = self.get_camera_extrinsic()
        info["focal_length"] = self.get_focal_length()

        return obs, state, reward, done, info


    def get_camera_RT(self):
        """
        get camera front/dynamic 's [eluer angle, translation]

        eluer angle order: roll, pitch, yaw
        """
        camera_param = []
        for cam_name in self.cameras:
            camera_pos = self.cam_modder.get_pos(cam_name)
            camera_quat = self.cam_modder.get_quat(cam_name)
            camera_eluer = self.euler_from_quaternion(camera_quat)
            camera_param_i = np.hstack([camera_eluer, camera_pos])
            camera_param.append(camera_param_i)

        camera_param = np.hstack(camera_param)
        return camera_param

    def get_camera_extrinsic(self):
        """
        get camera extrinsic, 3x4 matrix
        """
        camera_extrinsics = []

        for cam_name in self.cameras:
            camera_pos = self.cam_modder.get_pos(cam_name)
            camera_quat = self.cam_modder.get_quat(cam_name)
            
            q0, q1, q2, q3 = camera_quat
            r00 = 2 * (q0 **2 + q1**2) - 1
            r01 = 2 * (q1 * q2 - q0 * q3)
            r02 = 2 * (q1 * q3 + q0 * q2)
            r10 = 2 * (q1 * q2 + q0 * q3)
            r11 = 2 * (q0 **2 + q2**2) - 1
            r12 = 2 * (q2 * q3 - q0 * q1)
            r20 = 2 * (q1 * q3 - q0 * q2)
            r21 = 2 * (q2 * q3 + q0 * q1)
            r22 = 2 * (q0 **2 + q3**2) - 1

            rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

            # way 1
            # eluer to rotation matrix
            # camera_eluer = self.euler_from_quaternion(camera_quat)
            # camera_eluer = torch.tensor(camera_eluer).unsqueeze(0)
            # rotation_matrix = self.euler2rotmat(camera_eluer)
            
            # way 2
            # rotation_matrix = self.quat2mat(camera_quat)

            # way 3
            # rotation_matrix = self.quat2mat_new(torch.from_numpy(camera_quat).unsqueeze(0).cuda()).squeeze(0).cpu().numpy()

        
            # get extrinsic matrix
            camera_extrinsic = np.eye(4)
            camera_extrinsic[:3, :3] = rotation_matrix
            camera_extrinsic[:3, 3] = camera_pos
            
            camera_extrinsics.append(camera_extrinsic)

        camera_extrinsics = np.stack(camera_extrinsics, axis=0)
        return camera_extrinsics
      
    
    @staticmethod
    def quat2mat_new(quat, scaling=False, translation=False):
            """Convert quaternion coefficients to rotation matrix.
            Args:
                quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
            Returns:
                Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
            """
            norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
            norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
            w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

            B = quat.size(0)

            w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
            wx, wy, wz = w*x, w*y, w*z
            xy, xz, yz = x*y, x*z, y*z

            rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                                2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                                2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
            return rotMat

    def get_camera_intrinsic(self):

        img_height = img_width = self.image_size

        fovys = self.sim.model.cam_fovy
        assert fovys[0]==fovys[1], "two cameras should use same fovy"
        fovy = fovys[0]

        f = 0.5 * img_height / math.tan(fovy * math.pi / 360)

        intrinsic = np.array(((f, 0, img_width / 2), (0, f, img_height / 2), (0, 0, 1)))

        return intrinsic

    def get_focal_length(self, image_size=None):
        """
        Focal length (f) and field of view (FOV) of a lens are inversely proportional. 
        For a standard rectilinear lens, FOV = 2 arctan x/2f
        """
        if image_size is None:
            img_height = img_width = self.image_size
        else:
            img_height = img_width = image_size
        fovys = self.sim.model.cam_fovy
        assert fovys[0]==fovys[1], "two cameras should use same fovy"
        fovy = fovys[0]
        focal_length = 0.5 * img_height / math.tan(fovy * math.pi / 360)
        return focal_length

    def euler_from_quaternion(self, quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def quat2mat(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        
        return rot_matrix

    @staticmethod                    
    def euler2rotmat(angle, scaling=False):
        """Convert euler angles to rotation matrix.
        Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        Args:
            angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        B = angle.size(0)
        euler_angle = 'aby'

        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach() * 0
        ones = zeros.detach() + 1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz, cosz, zeros,
                            zeros, zeros, ones], dim=1).reshape(B, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack([cosy, zeros, siny,
                            zeros, ones, zeros,
                            -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack([ones, zeros, zeros,
                            zeros, cosx, -sinx,
                            zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

        rotMat = xmat @ ymat @ zmat
        # rotMat = zmat
        # rotMat = ymat @ zmat
        # rotMat = xmat @ ymat
        # rotMat = xmat @ zmat

        if scaling:
            v_scale = angle[:,3]
            v_trans = angle[:,4:]
        else:
            v_trans = angle[:,3:]

        if scaling:
            # one = torch.ones_like(v_scale).detach()
            # t_scale = torch.stack([v_scale, one, one,
            #                        one, v_scale, one,
            #                        one, one, v_scale], dim=1).view(B, 3, 3)
            rotMat = rotMat * v_scale.unsqueeze(1).unsqueeze(1)

        return rotMat


class GripperWrapper(gym.Wrapper):
    """
    Wrapper for gripper
    """
    def __init__(self, env, angle_threshold=0.18):
        gym.Wrapper.__init__(self, env)
        # 0.18 is good for lift. 0.4 is good for peg insert.
        self.angle_threshold = angle_threshold 
        print("[GripperWrapper] angle_threshold = {}".format(self.angle_threshold))
    

    def is_gripper_close(self):
        # gripper angle min: 0, max: 0.8
        # min: close, max: open
        gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') 
        is_close = gripper_angle > self.angle_threshold
        return is_close, gripper_angle


    def reset(self):
        obs, state, info = self.env.reset()
        info['is_gripper_close'], info['gripper_angle'] = self.is_gripper_close()
        return obs, state, info


    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        info['is_gripper_close'], info['gripper_angle'] = self.is_gripper_close()
        return obs, state, reward, done, info


class DynamicCameraWrapper(gym.Wrapper):
    """
    wrapper for randomizing camera
    """
    def __init__(self, env, domain_name, 
            camera_move_range=30, 
            num_static_cameras=1,
            num_dynamic_cameras=1,
            seed=None):
        gym.Wrapper.__init__(self, env)
        self.random_state = np.random.RandomState(seed)
        self.cam_modder = modder.CameraModder(self.sim, random_state=self.random_state)

        self.camera_move_range = np.deg2rad(camera_move_range)
        self.num_static_cameras = num_static_cameras
        self.num_dynamic_cameras = num_dynamic_cameras


        if domain_name == "robot":
            # for fixed mode
            self.start_angle = np.deg2rad(-30) # depends on env
            # self.interpolation_step = 0.05 # depends on episode length, 1/50 = 0.02
            self.interpolation_step = 0.02 # depends on episode length, 1/50 = 0.02
            self.camera_rotation_radius = 0.85 # depends on env
            
        elif domain_name == "metaworld":
            # for fixed mode
            # self.interpolation_step = 0.005
            # self.start_angle = np.deg2rad(120)
            # self.camera_rotation_radius = 1.3 # for fixed mode

            # for target mode
            self.interpolation_step = 0.005 # 1/200 = 0.005
            self.start_angle = np.deg2rad(-90) 
            self.camera_rotation_radius = 0.55 # for target mode

        else:
            raise NotImplementedError("domain {} is not implemented".format(domain_name))


        self.record_inital_camera_pos()
        self.compute_camera_rotation_base()
        self.compute_interpolate_trajectory()

        # set other static cameras
        if self.num_static_cameras > 1:
            self.set_all_static_cameras()
        
        if self.num_dynamic_cameras > 1:
            self.set_all_dynamic_cameras()
        
        
    
    def set_all_static_cameras(self):
        step_size = (self.traj_len-1) // (self.num_static_cameras-1)
        idx = [step_size*(i+1) for i in range(self.num_static_cameras-1)]

        origin_idx = self.traj_idx
        # set static cameras with the help of dynamic camera
        for i in range(self.num_static_cameras-1):
            traj_idx = idx[i]
            camera_name = "camera_static{}".format(i+1)
            self.change_traj_idx(traj_idx)
            self.randomize_camera()
            pos, quat = self.get_camera_pos_and_quat("camera_dynamic")
            self.set_camera_pos_and_quat(camera_name, pos, quat)
        # recover the dynamic camera back to the original position
        self.change_traj_idx(origin_idx)
        self.randomize_camera()

    def set_all_dynamic_cameras(self):
        step_size = (self.traj_len-1) // (self.num_dynamic_cameras-1)
        origin_idx = self.traj_idx
        idx = [origin_idx + step_size*(i+1) for i in range(self.num_dynamic_cameras-1)]
        
        # set static cameras with the help of dynamic camera
        for i in range(self.num_dynamic_cameras-1):
            traj_idx = idx[i]
            camera_name = "camera_dynamic{}".format(i+1)
            self.change_traj_idx(traj_idx)
            self.randomize_camera()
            pos, quat = self.get_camera_pos_and_quat("camera_dynamic")
            self.set_camera_pos_and_quat(camera_name, pos, quat)
        # recover the dynamic camera back to the original position
        self.change_traj_idx(origin_idx)
        self.randomize_camera()

    def step(self, action):
        self._randomize_camera()
        if self.num_dynamic_cameras > 1:
            self.set_all_dynamic_cameras()
        obs, state, reward, done, info = self.env.step(action)
        return obs, state, reward, done, info
    
    def reset(self):
        return self.env.reset()

    def record_inital_camera_pos(self):
        """
        Record new initialized camera.
        """
        self.init_camera_positions = {}
        self.init_camera_positions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_pos("camera_dynamic"))
        self.init_camera_positions["camera_static"] = copy.deepcopy(self.cam_modder.get_pos("camera_static"))
        
        self.init_camera_quaternions = {}
        self.init_camera_quaternions["camera_dynamic"] = copy.deepcopy(self.cam_modder.get_quat("camera_dynamic"))
        self.init_camera_quaternions["camera_static"] = copy.deepcopy(self.cam_modder.get_quat("camera_static"))


    def compute_interpolate_trajectory(self):
        """
        Use front and dynamic view to generate interpolated traj
        """
        
        self.first_pos = copy.deepcopy(self.init_camera_positions["camera_static"])
        self.second_pos = copy.deepcopy(self.init_camera_positions["camera_dynamic"])
        self.traj_idx = 0 
        
        self.camera_traj = []
        self.roll_traj = []

       
        interpolation_sequence = np.arange(0.0, 0.99, self.interpolation_step)
        self.traj_len = interpolation_sequence.shape[0]
        print(colored("[DynamicCameraWrapper] camera traj len: %u"%self.traj_len, color="cyan"))

        for a in interpolation_sequence:
            self.camera_traj.append( self.start_angle + a*self.camera_move_range ) 
            self.roll_traj.append( self.base_roll + a*self.camera_move_range )


    def compute_camera_rotation_base(self):
        # use the static camera to compute the centre, and apply on dynamic camera
        pos = self.cam_modder.get_pos("camera_dynamic")
        self.base_x = pos[0] -  self.camera_rotation_radius * np.sin(self.start_angle)
        self.base_y = pos[1] -  self.camera_rotation_radius * np.cos(self.start_angle)
        self.base_z = pos[2]

        # and get base euler
        quat = self.cam_modder.get_quat("camera_dynamic")
        self.base_roll, self.base_pitch, self.base_yaw = rot_utils.quat2euler(*quat)


    def get_camera_pos_and_euler(self, camera_name):
        pos = self.cam_modder.get_pos(camera_name)
        quat = self.cam_modder.get_quat(camera_name)
        roll, pitch, yaw = rot_utils.quat2euler(*quat)
        return pos, roll, pitch, yaw
    

    def get_camera_pos_and_quat(self, camera_name):
        pos = self.cam_modder.get_pos(camera_name)
        quat = self.cam_modder.get_quat(camera_name)
        return pos, quat
    

    def set_camera_pos_and_quat(self, camera_name, pos, quat):
        self.cam_modder.set_pos(camera_name, pos)
        self.cam_modder.set_quat(camera_name, quat)


    def randomize_camera(self, rand_first=False):
        """
        The core of the dynamic camera moving.
        """

        # set dynamic camera, which is moving
        theta = self.camera_traj[self.traj_idx]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # change translation
        pos = self.cam_modder.get_pos("camera_dynamic")

        pos[0] = self.base_x + self.camera_rotation_radius * sin_theta
        pos[1] = self.base_y + self.camera_rotation_radius * cos_theta
        pos[2] = self.base_z
        
        self.cam_modder.set_pos("camera_dynamic", pos)


        # change rotation
        quat = self.cam_modder.get_quat("camera_dynamic")
        roll, pitch, yaw = rot_utils.quat2euler(*quat)
        roll = self.roll_traj[self.traj_idx]
        pitch = self.base_pitch
        yaw = self.base_yaw
        quat = rot_utils.euler2quat(roll, pitch, yaw)
        self.cam_modder.set_quat("camera_dynamic", quat)

        # increment traj idx
        self.traj_idx = (self.traj_idx + 1)%self.traj_len

    

    def _randomize_camera(self, rand_first=False):
        """
        The core of the dynamic camera moving.
        """

        # set dynamic camera, which is moving
        theta = self.camera_traj[self.traj_idx]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)


        # change translation
        pos = self.cam_modder.get_pos("camera_dynamic")

        pos[0] = self.base_x + self.camera_rotation_radius * sin_theta
        pos[1] = self.base_y + self.camera_rotation_radius * cos_theta
        pos[2] = self.base_z
        
        self.cam_modder.set_pos("camera_dynamic", pos)


        # change rotation
        quat = self.cam_modder.get_quat("camera_dynamic")
        roll, pitch, yaw = rot_utils.quat2euler(*quat)
        roll = self.roll_traj[self.traj_idx]
        pitch = self.base_pitch
        yaw = self.base_yaw
        quat = rot_utils.euler2quat(roll, pitch, yaw)
        self.cam_modder.set_quat("camera_dynamic", quat)

        # increment traj idx
        self.traj_idx = (self.traj_idx + 1)%self.traj_len

    
    def reset(self):
        self.traj_idx = 0 # reset dynamic camera
        obs, state= self.env.reset()
        return obs, state
    
    def change_traj_idx(self, idx):
        self.traj_idx = idx % self.traj_len
        


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
