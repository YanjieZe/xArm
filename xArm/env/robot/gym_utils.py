from turtle import position
import numpy as np

from gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)



def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.

    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0: # torque control
                sim.data.ctrl[i] = action[i]
            else: # position control
                # print('i: {}, ctrl: {}, action: {}'.format(i, sim.data.ctrl[i], action[i]))
                sim.data.ctrl[i] = action[i]
                # action_threshold = 0.2
                # action_scale = 1
                # if action[i] > action_threshold:
                #     position_ctrl = action_scale
                # elif action[i] < -action_threshold:
                #     position_ctrl = - action_scale
                # else:
                #     position_ctrl = 0.0
                # sim.data.ctrl[i] = position_ctrl
            


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]
        reset_mocap2body_xpos(sim) # actually reset the quat. Here is the understanding. originally: 1. reset: 1->0. recover: 0+1->1
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return

    # For all weld constraints
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        body2 = sim.model.body_id2name(obj2_id)
        if body2 == 'B0' or body2== 'B9' or body2 == 'B1':
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]

        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
