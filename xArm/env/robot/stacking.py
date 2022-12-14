import numpy as np
import os
from gym import utils
from xArm.env.robot.base import BaseEnv, get_full_asset_path
import math

class StackingEnv(BaseEnv, utils.EzPickle):
	"""
	stacking object1 onto object0
	"""
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, \
				 use_xyz=False, render=False):
		self.sample_large = 1
		self.object_height = 0.04
		
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
		self.distance_threshold = 0.04 # distance threshold for success
		self.state_dim = (37,)
		self.flipbit = 1

		self.object0_qpos_init = self.sim.data.get_joint_qpos('object0_joint0').copy()
		self.object1_qpos_init = self.sim.data.get_joint_qpos('object1_joint0').copy()
		utils.EzPickle.__init__(self)


	def check_grasp(self, object_name):
		"""
		check if gripper is grasping object
		"""
		gripper_pos = self.sim.data.get_site_xpos('grasp').copy() # gripper position
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		object_pos = self.sim.data.get_site_xpos(f'{object_name}_site').copy() # object position
		distance = np.linalg.norm(gripper_pos - object_pos) # distance between gripper and object
		is_grasp = distance < 0.05 and gripper_angle > 0.22 # gripper is grasping object
		return is_grasp

    		
	def compute_reward(self, achieved_goal, goal, info):
		reward = 0.

		#### current state ####
		object0_pos = self.sim.data.get_site_xpos(f'object0_site').copy()
		object1_pos = self.sim.data.get_site_xpos(f'object1_site').copy()
		object1_target_pos = object0_pos.copy()
		object1_target_pos[2] += self.object_height
		gripper_pos = self.sim.data.get_site_xpos('grasp').copy() # gripper position

		# compute distance
		object1_distance = np.linalg.norm(object1_pos - object1_target_pos) # distance to target

		# task progress
		success = object1_distance <= self.distance_threshold # object1 is on target

		if success:
			reward = 15.0
		else:
			# reaching object reward
			tcp_pose = gripper_pos
			cubeA_pos = object1_pos
			cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
			reaching_reward = 1 - np.tanh(3.0 * cubeA_to_tcp_dist)
			reward += reaching_reward

			# check if cubeA is on cubeB
			cubeA_pos = object1_pos
			cubeB_pos = object0_pos
			goal_xyz = np.hstack(
				[cubeB_pos[0:2], cubeB_pos[2] + self.object_height]
			)
			cubeA_on_cubeB = (
				np.linalg.norm(goal_xyz[:2] - cubeA_pos[:2])
				< self.object_height * 0.8
			)
			cubeA_on_cubeB = cubeA_on_cubeB and (
				np.abs(goal_xyz[2] - cubeA_pos[2]) <= 0.005
			)
			if cubeA_on_cubeB:
				reward = 10.0
				# ungrasp reward
				is_cubeA_grasped = self.check_grasp("object1")
				if not is_cubeA_grasped:
					reward += 2.0
				else:
					reward = (
						reward
						+ 2.0 * np.sum(self.agent.robot.get_qpos()[-2:]) / 1.0
					)
			else:
				# grasping reward
				is_cubeA_grasped = self.check_grasp("object1")
				if is_cubeA_grasped:
					reward += 1.0

				# reaching goal reward, ensuring that cubeA has appropriate height during this process
				if is_cubeA_grasped:
					cubeA_to_goal = goal_xyz - cubeA_pos
					# cubeA_to_goal_xy_dist = np.linalg.norm(cubeA_to_goal[:2])
					cubeA_to_goal_dist = np.linalg.norm(cubeA_to_goal)
					appropriate_height_penalty = np.maximum(
						np.maximum(2 * cubeA_to_goal[2], 0.0),
						np.maximum(2 * (-0.02 - cubeA_to_goal[2]), 0.0),
					)
					reaching_reward2 = 2 * (
						1 - np.tanh(5.0 * appropriate_height_penalty)
					)
					# qvel_penalty = np.sum(np.abs(self.agent.robot.get_qvel())) # prevent the robot arm from moving too fast
					# reaching_reward2 -= 0.0003 * qvel_penalty
					# if appropriate_height_penalty < 0.01:
					reaching_reward2 += 4 * (1 - np.tanh(5.0 * cubeA_to_goal_dist))
					reward += np.maximum(reaching_reward2, 0.0)
		return reward
	

	def _reset_sim(self):
		self.lifted = False # reset stage flag
		self.over_obj = False
		self.over_goal = False

		return BaseEnv._reset_sim(self)
	

	def _is_success(self, achieved_goal, desired_goal):
		# current state
		object0_pos = self.sim.data.get_site_xpos(f'object0_site').copy()
		object1_pos = self.sim.data.get_site_xpos(f'object1_site').copy()
		object1_target_pos = object0_pos.copy()
		object1_target_pos[2] += self.object_height

		# compute distance
		object1_distance = np.linalg.norm(object1_pos - object1_target_pos) # distance to target

		return  object1_distance <= self.distance_threshold # object1 is on target


	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8


		obj0_pos = self.sim.data.get_site_xpos('object0_site')
		obj0_rot = self.sim.data.get_joint_qpos('object0_joint0')[-4:]
		obj0_velp = self.sim.data.get_site_xvelp('object0_site') * dt
		obj0_velr = self.sim.data.get_site_xvelr('object0_site') * dt


		obj1_pos = self.sim.data.get_site_xpos('object1_site')
		obj1_rot = self.sim.data.get_joint_qpos('object1_joint0')[-4:]
		obj1_velp = self.sim.data.get_site_xvelp('object1_site') * dt
		obj1_velr = self.sim.data.get_site_xvelr('object1_site') * dt
		
		target_pos = obj0_pos.copy()
		target_pos[2] += self.object_height

		
		addition_values = np.array([
			self.goal_distance(obj1_pos, target_pos, self.use_xyz),
			gripper_angle
		])

		state = np.concatenate([
			eef_pos, eef_velp, goal_pos,
			obj0_pos, obj0_rot, obj0_velp, obj0_velr, # object0 state
			obj1_pos, obj1_rot, obj1_velp, obj1_velr, # object1 state
			addition_values
		], axis=0)

		return state


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
		return np.squeeze(self.sim.data.get_site_xpos('object1_site').copy())

 
	def _sample_object_pos(self):
		scale_random = 0.025 # 0.1 -> 0.05 -> 0.025

		# random first object
		object0_qpos = self.object0_qpos_init.copy() # xyz+quat
		object0_qpos[0] += self.np_random.uniform(-scale_random, scale_random, size=1)
		object0_qpos[1] += self.np_random.uniform(-scale_random, scale_random, size=1)
		self.sim.data.set_joint_qpos('object0_joint0', object0_qpos)

		# random second object
		object1_qpos = self.object1_qpos_init.copy() # xyz+quat
		object1_qpos[0] += self.np_random.uniform(-scale_random, scale_random, size=1)
		object1_qpos[1] += self.np_random.uniform(-scale_random, scale_random, size=1)
		self.sim.data.set_joint_qpos('object1_joint0', object1_qpos)


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