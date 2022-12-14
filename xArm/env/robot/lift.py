import numpy as np
import os
from gym import utils
from xArm.env.robot.base import BaseEnv, get_full_asset_path
import math

def euler2quat(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]

def quat2euler(x, y, z, w):
	"""
	Convert a quaternion into euler angles (roll, pitch, yaw)
	roll is rotation around x in radians (counterclockwise)
	pitch is rotation around y in radians (counterclockwise)
	yaw is rotation around z in radians (counterclockwise)
	"""
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
	
	return [roll_x, pitch_y, yaw_z] # in radians
 

class LiftEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, \
				 use_xyz=False, render=False):
		self.sample_large = 1
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
		)
		self.state_dim = (26,) if self.use_xyz else (20,)
		self.flipbit = 1
		utils.EzPickle.__init__(self)


	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action

		objPos = self.sim.data.get_site_xpos('object_site').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		heightTarget = self.lift_height + self.objHeight
		reachDist = np.linalg.norm(objPos - fingerCOM)

		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(np.linalg.norm(objPos[-1] - fingerCOM[-1]))


			if reachDistxy < 0.05:
				reachRew = -reachDist
			else:
				reachRew =  -reachDistxy - 2*zRew

			# incentive to close fingers when reachDist is small
			if reachDist < 0.05:
				reachRew = -reachDist + max(actions[-1],0)/50
			return reachRew , reachDist

		def pickCompletionCriteria():
			tolerance = 0.01
			return objPos[2] >= (heightTarget- tolerance)

		self.pickCompleted = pickCompletionCriteria()


		def objDropped():
			return (objPos[2] < (self.objHeight + 0.005)) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper

		def orig_pickReward():
			hScale = 100
			if self.pickCompleted and not(objDropped()):
				return hScale*heightTarget
			elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)):
				return hScale* min(heightTarget, objPos[2])
			else:
				return 0


		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()

		reward = reachRew + pickRew

		return reward	

	def _reset_sim(self):
		self.lifted = False # reset stage flag
		self.over_obj = False
		self.over_goal = False

		return BaseEnv._reset_sim(self)
	
	def _is_success(self, achieved_goal, desired_goal):
		''' The block is lifted above a certain threshold in z'''
		object_pos = self.sim.data.get_site_xpos('object_site').copy()
		return (object_pos[2] - self.center_of_table.copy()[2]) > self.lift_height


	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8


		obj_pos = self.sim.data.get_site_xpos('object_site')
		obj_rot = self.sim.data.get_joint_qpos('object_joint0')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object_site') * dt
		obj_velr = self.sim.data.get_site_xvelr('object_site') * dt

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			obj_pos = obj_pos[:2]
			obj_velp = obj_velp[:2]
			obj_velr = obj_velr[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			self.goal_distance(obj_pos, goal_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)


	def _set_action(self, action):
		assert action.shape == (4,)
		if self.flipbit:
			action[3] = 0
			self.flipbit = 0
		else:
			action[:3] = np.zeros(3)
			self.flipbit = 1

		BaseEnv._set_action(self, action)
		self.current_action = action # store current_action


	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('object_site').copy())

	
 
	def _sample_object_pos(self):
		# scale = 0.1
		scale = 0.01
		object_xpos = self.center_of_table.copy() - np.array([0.15, 0.1, 0.05])

		object_xpos[0] += self.np_random.uniform(-scale, scale, size=1)
		object_xpos[1] += self.np_random.uniform(-scale, scale, size=1)

		object_qpos = self.sim.data.get_joint_qpos('object_joint0')
		object_quat = object_qpos[-4:]

		rotation_randomize = False
		if rotation_randomize:
			rotation_random_scale = 3.1415926

			eluer_angle = quat2euler(*object_quat)
			eluer_angle[0] += self.np_random.uniform(-rotation_random_scale, rotation_random_scale)
			object_quat = euler2quat(*eluer_angle)

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object_joint0', object_qpos)

		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]


	def _sample_goal(self, new=True): # task has no goal
		goal = self.center_of_table.copy() - np.array([0.3, 0, 0])
		goal[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		goal[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		goal[2] += 0.08 + 0.05
		self.lift_height = 0.15
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.28, .295, 0.71])
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)