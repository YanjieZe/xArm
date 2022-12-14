import numpy as np
import os
from gym import utils
from xArm.env.robot.base import BaseEnv, get_full_asset_path


class PeginsertEnv(BaseEnv, utils.EzPickle):
	"""
	Insert into the box
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense',
				 image_size=84, use_xyz=False, render=False):
		self.sample_large = 1
		
		self.box_init_pos = None

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
		
		self.state_dim = (31,)
		self.flipbit = 1
		# 2022.01.19: 0.05 -> 0.1
		self.distance_threshold = 0.1
		utils.EzPickle.__init__(self)


	
	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action
		
		objPos = self.sim.data.get_site_xpos('pegGrasp').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		# heightTarget = self.lift_height + self.objHeight
		heightTarget = self.sim.data.get_site_xpos('goal')[-1] - 0.05 # 2022/1/25
		reachDist = np.linalg.norm(objPos - fingerCOM)
		
		placingGoal = self.sim.data.get_site_xpos('goal')
		self.goal = placingGoal
		
		self.objHeight = objPos[2]

		assert (self.goal-placingGoal==0).any(), "goal does not match"

		placingDist = np.linalg.norm(objPos - placingGoal)

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
		

		def placeReward():
			c1 = 1000
			c2 = 0.01
			c3 = 0.001
			cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

			if cond:
				placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
				placeRew = max(placeRew,0)
				return [placeRew , placingDist]
			else:
				return [0 , placingDist]
		
		def directPlaceReward(): 
			# give a reward for the robot to make object close
			if(objPos[2] > heightTarget-0.1):
				return 1000*(self.maxPlacingDist - placingDist)
			else:
				return 0

		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()
		placeRew , placingDist = placeReward()
		direct_place_reward = directPlaceReward()# added in 2022/1/23
		assert ((placeRew >=0) and (pickRew>=0))
		reward = reachRew + pickRew + placeRew + direct_place_reward

		return reward

	def _get_state_obs(self):
		
		# 1
		grasp_pos = self.sim.data.get_site_xpos("grasp")

		# 2
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8
		
		# 3
		peg_grasp = self.sim.data.get_site_xpos('pegGrasp')
		peg_head = self.sim.data.get_site_xpos('pegHead')
		peg_end = self.sim.data.get_site_xpos('pegEnd')
		

		# 4
		# hole and goal are different
		hole_pos = self.sim.data.get_site_xpos('hole').copy()
		goal_pos = self.goal

		# 5
		box_bottom_right_corner_1 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_1')
		box_bottom_right_corner_2 = self.sim.data.get_site_xpos('bottom_right_corner_collision_box_2')
		box_top_left_corner_1 = self.sim.data.get_site_xpos('top_left_corner_collision_box_1')
		box_top_left_corner_2 = self.sim.data.get_site_xpos('top_left_corner_collision_box_2')

		# concat
		state = np.concatenate([grasp_pos,[gripper_angle],peg_head, peg_grasp,peg_end , hole_pos, goal_pos, 
						box_bottom_right_corner_1, box_bottom_right_corner_2, box_top_left_corner_1, box_top_left_corner_2])

		return state

	def _reset_sim(self):
		self.over_obj = False
		self.lifted = False # reset stage flag
		self.placed = False # reset stage flag
		self.over_goal = False

		return BaseEnv._reset_sim(self)

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
		"""
		Get the position of the target pos.
		"""
		return np.squeeze(self.sim.data.get_site_xpos('pegHead').copy())

	def _is_success(self, achieved_goal, desired_goal):
		dist = np.linalg.norm(achieved_goal - desired_goal)
		success = (dist < self.distance_threshold).astype(np.float32)
		return success

	def _sample_object_pos(self):
		"""
		Sample the initial position of the peg
		"""
		scale = 0.05 # 0.1 -> 0.05
		object_xpos = np.array([1.33, 0.23, 0.565]) # to align with real
		object_xpos[0] += self.np_random.uniform(-scale, scale, size=1)
		object_xpos[1] += self.np_random.uniform(-scale, scale, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('peg:joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('peg:joint', object_qpos)


		self.peg_init_pos = object_xpos.copy()
		self.peg_height = object_xpos[-1]

		
		
		self.initial_gripper_xpos = self.sim.data.get_site_xpos('grasp').copy()
		self.maxPlacingDist = np.linalg.norm(self.peg_init_pos - self.goal)


	def _sample_goal(self, new=True):

		# Randomly sample the position of the box
		box_pos = self.sim.data.get_body_xpos("box")
		
		if self.box_init_pos is None:
			self.box_init_pos = box_pos.copy()
		else:
			box_pos[0] = self.box_init_pos[0] + self.np_random.uniform(-0.05, 0.05, size=1)
			box_pos[1] = self.box_init_pos[1] + 0.1*self.np_random.uniform(-0.1, 0.1, size=1)
			box_pos[2] = self.box_init_pos[2]
		
			
		goal = self.sim.data.get_site_xpos('goal').copy()

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