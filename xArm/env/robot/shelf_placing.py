import numpy as np
import os
import xArm.env.robot.reward_utils as reward_utils
from gym import utils
from xArm.env.robot.base import BaseEnv, get_full_asset_path


class ShelfPlacingEnv(BaseEnv, utils.EzPickle):
	"""
	Place the object on the shelf
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', 
				image_size=84, use_xyz=False, render=False, env_randomness_scale=.3):
		self.sample_large = 1
		self.env_randomness_scale = env_randomness_scale
		
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
		
		self.flipbit = 1
		self.state_dim = (26,)

		self.distance_threshold = 0.05
		
		utils.EzPickle.__init__(self)


	def _is_success(self, achieved_goal, desired_goal):
		success = float(self.obj_place_on_shelf)
		return success


	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action

		objPos = self.sim.data.get_site_xpos('object0_site').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		heightTarget = self.goal[2]
		target_pos = self.sim.data.get_site_xpos('target0').copy()
		"""
		height: 0.90
		target pos: [1.60129918 0.29330338 0.90096052]
		"""
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

		# stage 1: reach and lift
		reward = reachRew + pickRew


		# stage 2: get close to the target
		dist_to_target = np.linalg.norm(objPos - target_pos)
		max_dist = np.linalg.norm(self.sim.data.get_site_xpos('target0') - self.obj_init_pos)
		reward_dist_to_target = 10*(max_dist - dist_to_target)
		reward += reward_dist_to_target * (self.pickCompleted and not(objDropped()))
		if dist_to_target < self.distance_threshold:
			self.obj_place_on_shelf = True

		reward_to_place = 0.
		if self.obj_place_on_shelf:
			reward_to_place = 100*(self.current_action[-1] < 0)*(-self.current_action[-1])
			reward = 500 + reward_to_place
		
		return reward


	def _get_state_obs(self):
    	
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8


		obj_pos = self.sim.data.get_site_xpos('object0_site')
		obj_rot = self.sim.data.get_joint_qpos('object0_joint')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object0_site') * dt
		obj_velr = self.sim.data.get_site_xvelr('object0_site') * dt


		target_pos = self.sim.data.get_site_xpos('target0')
		target_dist = np.linalg.norm(obj_pos - target_pos)


		values = np.array([
			self.goal_distance(eef_pos, target_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			target_dist,
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, target_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)


	def _reset_sim(self):
		self.obj_place_on_shelf = False
		return BaseEnv._reset_sim(self)

	def _set_action(self, action):
		assert action.shape == (4,)

		if self.flipbit:
			action[3] = 0
			self.flipbit = 0
		else:
			action[:3] = np.zeros(3)
			self.flipbit = 1
			
		self.current_action = action # store current_action

		BaseEnv._set_action(self, action)


	def _get_achieved_goal(self):
		"""
		Get the position of the target pos.
		"""
		return np.squeeze(self.sim.data.get_site_xpos('object0_site').copy())

	def _sample_object_pos(self):
		"""
		Sample the initial position of the object
		"""
		scale = 0.05 # 1(not working)->0.05
		object_xpos = np.array([1.355, 0.3, 0.58625])
		object_xpos[0] += self.np_random.uniform(-scale/2, scale/2, size=1)
		object_xpos[1] += self.np_random.uniform(-scale, scale, size=1)
		self.obj_init_pos = object_xpos.copy()

		object_qpos = self.sim.data.get_joint_qpos('object0_joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object0_joint', object_qpos)
		
		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]
		
		self.maxPlacingDist = np.linalg.norm(self.obj_init_pos - self.goal)


	def _sample_goal(self, new=True):
		"""
		Sample the position of the shelf, and the goal is bound to the shelf.
		"""		
		self.lift_height = 0.15


		goal = self.sim.data.get_site_xpos('target0')
		
		self.heightTarget = goal[2]

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		"""
		Sample the initial position of arm
		"""
		gripper_target = np.array([1.28, .295, 0.71])
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)