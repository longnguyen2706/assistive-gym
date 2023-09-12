import os
from datetime import datetime
from typing import Set, Optional

import numpy as np
import pybullet as p
from cma import CMA, CMAEvolutionStrategy
from cmaes import CMA
from torch.utils.hipify.hipify_python import bcolors

from assistive_gym.envs.utils.dto import HandoverObject, HandoverObjectConfig, MaximumHumanDynamics, OriginalHumanInfo
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.point_utils import fibonacci_evenly_sampling_range_sphere

LOG = get_logger()

objectTaskMapping = {
    HandoverObject.PILL: "comfort_taking_medicine",
    HandoverObject.CUP: "comfort_drinking",
    HandoverObject.CANE: "comfort_standing_up"
}

def create_point(point, size=0.01, color=[1, 0, 0, 1]):
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
    id = p.createMultiBody(baseMass=0,
                           baseCollisionShapeIndex=sphere,
                           basePosition=np.array(point))
    p.setGravity(0, 0, 0)
    return id


def draw_point(point, size=0.01, color=[1, 0, 0, 1]):
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
    p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=sphere,
                      basePosition=np.array(point))

def get_task_from_handover_object(object_name):
    if not object_name:
        return None
    object_type = HandoverObject.from_string(object_name)
    task = objectTaskMapping[object_type]
    return task


def solve_ik(env, target_pos, end_effector="right_hand"):
    human = env.human
    ee_idx = human.human_dict.get_dammy_joint_id(end_effector)
    ik_joint_indices = human.find_ik_joint_indices()
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices, max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution


def cal_energy_change(human, original_link_positions, end_effector):
    g = 9.81  # gravitational acceleration
    potential_energy_initial = 0
    potential_energy_final = 0

    link_indices = human.human_dict.get_real_link_indices(end_effector)
    current_link_positions = human.get_link_positions(True, end_effector_name=end_effector)

    # Iterate over all links
    for idx, j in enumerate(link_indices):
        mass = p.getDynamicsInfo(human.body, j)[0]
        LOG.debug(f"{bcolors.WARNING}link {j}, mass: {mass}{bcolors.ENDC}")

        # Calculate initial potential energy
        potential_energy_initial += mass * g * original_link_positions[idx][2]  # z axis
        potential_energy_final += mass * g * current_link_positions[idx][2]
        # Add changes to the total energy change
    total_energy_change = potential_energy_final - potential_energy_initial
    LOG.debug(
        f"Total energy change: {total_energy_change}, Potential energy initial: {potential_energy_initial}, Potential energy final: {potential_energy_final}")

    return total_energy_change, potential_energy_initial, potential_energy_final


def generate_target_points(env, num_points, ee_pos):
    human = env.human
    ee_pos, _ = human.get_ee_pos_orient("right_hand")
    points = fibonacci_evenly_sampling_range_sphere(ee_pos, [0.25, 0.5], num_points)
    return get_valid_points(env, points)


def get_initial_guess(env, target=None):
    if target is None:
        return np.zeros(len(env.human.controllable_joint_indices))  # no of joints
    else:
        # x0 = env.human.ik_chain(target)
        x0 = solve_ik(env, target, end_effector="right_hand")
        print(f"{bcolors.BOLD}x0: {x0}{bcolors.ENDC}")
        return x0


def debug_solution():
    # ee_pos, _, _= env.human.fk(["right_hand_limb"], x0)
    # ee_pos = env.human.fk_chain(x0)
    # print("ik error 2: ", eulidean_distance(ee_pos, target))
    # env.human.set_joint_angles(env.human.controllable_joint_indices, x0)

    # right_hand_ee = env.human.human_dict.get_dammy_joint_id("right_hand")
    # ee_positions, _ = env.human.forward_kinematic([right_hand_ee], x0)
    # print("ik error: ", eulidean_distance(ee_positions[0], target))
    #
    # for _ in range(1000):
    #     p.stepSimulation()
    # time.sleep(100)
    pass


def test_collision():
    # print("collision1: ", human.check_self_collision())
    # # print ("collision1: ", perform_collision_check(human))
    # x1 = np.random.uniform(-1, 1, len(human.controllable_joint_indices))
    # human.set_joint_angles(human.controllable_joint_indices, x1)
    # # for i in range (100):
    # #     p.stepSimulation(physicsClientId=human.id)
    # #     print("collision2: ", human.check_self_collision())
    # p.performCollisionDetection(physicsClientId=human.id)
    # # print("collision2: ", perform_collision_check(human))
    # print("collision2: ", human.check_self_collision())
    # time.sleep(100)
    pass


def cal_mid_angle(lower_bounds, upper_bounds):
    return (np.array(lower_bounds) + np.array(upper_bounds)) / 2


def count_new_collision(old_collisions: Set, new_collisions: Set, human, end_effector, penetration_threshold) -> int:
    # TODO: remove magic number (might need to check why self colllision happen in such case)
    # TODO: return number of collisions instead and use that to scale the cost
    link_ids = set(human.human_dict.get_real_link_indices(end_effector))
    # print ("link ids", link_ids)

    # convert old collision to set of tuples (link1, link2), remove penetration
    initial_collision_map = dict()
    for o in old_collisions:
        initial_collision_map[(o[0], o[1])] = o[2]

    collision_set = set()  # list of collision that is new or has deep penetration
    for collision in new_collisions:
        link1, link2, penetration = collision
        if not link1 in link_ids and not link2 in link_ids:
            continue  # not end effector chain collision, skip
        # TODO: fix it, since link1 and link2 in collision from different object, so there is a slim chance of collision
        if (link1, link2) not in initial_collision_map or (link2, link1) not in initial_collision_map:  # new collision:
            if abs(penetration) > penetration_threshold[
                "new"]:  # magic number. we have penetration between spine4 and shoulder in pose 5
                print("new collision: ", collision)
                collision_set.add((collision[0], collision[1]))
        else:
            # collision in old collision
            initial_depth = initial_collision_map[(link1, link2)] if (link1, link2) in initial_collision_map else \
                initial_collision_map[(link2, link1)]
            if abs(penetration) > max(penetration_threshold["old"],
                                      initial_depth):  # magic number. we have penetration between spine4 and shoulder in pose 5
                print("old collision with deep penetration: ", collision)
                collision_set.add((link1, link2))

    return len(collision_set)


def cal_dist_to_bedside(env, end_effector):
    human, bed = env.human, env.furniture
    ee_pos, _ = human.get_ee_pos_orient(end_effector)

    bed_bb = p.getAABB(bed.body, physicsClientId=env.id)
    bed_pos = p.getBasePositionAndOrientation(bed.body, physicsClientId=env.id)[0]
    side = "right" if ee_pos[0] > bed_pos[0] else "left"
    bed_xx, bed_yy, bed_zz = bed_bb[1] if side == "right" else bed_bb[0]
    bed_xx = bed_xx + 0.1 if side == "right" else bed_xx - 0.1
    # print ('bed size: ', np.array(bed_bb[1]) - np.array(bed_bb[0]))
    # print ("bed_xx: ", bed_xx, "ee_pos: ", ee_pos, "side: ", side)
    if side == "right":
        return 0 if ee_pos[0] > bed_xx else abs(ee_pos[0] - bed_xx)
    else:
        return 0 if ee_pos[0] < bed_xx else abs(ee_pos[0] - bed_xx)


def cal_angle_diff(cur, target):
    # print ("cur: ", len(cur), 'target: ', len(target))
    diff = np.sqrt(np.sum(np.square(np.array(cur) - np.array(target)))) / len(cur)
    # print ("diff: ", diff)
    return diff


def get_valid_points(env, points):
    point_ids = []
    for point in points:
        id = create_point(point, size=0.01)
        point_ids.append(id)
    p.performCollisionDetection(physicsClientId=env.id)

    valid_points = []
    valid_ids = []
    for idx, point in enumerate(points):
        id = point_ids[idx]
        contact_points = p.getContactPoints(bodyA=id, physicsClientId=env.id)
        if len(contact_points) == 0:
            valid_points.append(point)
            # valid_ids.append(id)
        # else:
        #     p.removeBody(id)
        p.removeBody(id)
    return valid_points, valid_ids


def cal_torque_magnitude(human, end_effector):
    def pretty_print_torque(human, torques, end_effector):
        link_names = human.human_dict.joint_chain_dict[end_effector]
        # print ("link_names: ", link_names)
        # print ("torques: ", len(torques))
        for i in range(0, len(torques), 3):
            LOG.debug(f"{link_names[i // 3]}: {torques[i: i + 3]}")

    # torques = human.inverse_dynamic(end_effector)
    # print ("torques ee: ", len(torques), torques)
    # torques = human.inverse_dynamic()
    # print ("torques: ", len(torques), torques)
    torques = human.inverse_dynamic(end_effector)
    # print("torques ee: ", torques)
    # print ("----------------------------------")
    pretty_print_torque(human, torques, end_effector)

    torque_magnitude = 0
    for i in range(0, len(torques), 3):
        torque = np.sqrt(np.sum(np.square(torques[i:i + 3])))
        torque_magnitude += torque
    LOG.debug(f"torques: {torques}, torque magnitude: {torque_magnitude}")
    return torque_magnitude


def get_actions_dict_key(handover_obj, robot_ik):
    return handover_obj + "-robot_ik" if robot_ik else handover_obj + "-no_robot_ik"


def get_save_dir(save_dir, env_name, person_id, smpl_file, timestamp=False):
    smpl_name = smpl_file.split('/')[-1].split('.')[0]
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if timestamp:
        return os.path.join(save_dir, env_name, person_id, smpl_name, time)
    else:
        return os.path.join(save_dir, env_name, person_id, smpl_name)


def get_max_torque(env, end_effector="right_hand"):
    human = env.human
    human.set_joint_angles(human.controllable_joint_indices, len(human.controllable_joint_indices) * [0])
    torque = cal_torque_magnitude(human, end_effector)
    print("max torque: ", torque)
    return torque


def max_torque_cost_fn(human, end_effector):
    torque = cal_torque_magnitude(human, end_effector)
    return 1.0 / torque


def max_manipulibity_cost_fn(human, end_effector, joint_angles):
    manipulibility = human.cal_chain_manipulibility(joint_angles, end_effector)
    return 1.0 / manipulibility


def max_energy_cost_fn(human, end_effector, original_link_positions):
    _, _, energy_final = cal_energy_change(human, original_link_positions, end_effector)
    return 1.0 / energy_final


def move_robot(env):  # for debugging purpose
    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool
    target_joint_angles = np.random.uniform(-1, 1, len(robot.right_arm_joint_indices)) * np.pi

    for i in range(100):
        # random_val = np.random.uniform(-1, 1, len(robot.controllable_joint_indices))
        robot.control(robot.right_arm_joint_indices, np.array(target_joint_angles), 0.1, 100)
        p.stepSimulation()

    print("tool mass: ", p.getDynamicsInfo(tool.body, -1)[0])


def get_human_link_robot_collision(human, end_effector):
    human_link_robot_collision = []
    for ee in human.human_dict.end_effectors:
        human_link_robot_collision.extend([link for link in human.human_dict.get_real_link_indices(ee)])
    # ignore collision with end effector and end effector's parent link
    parent_ee = human.human_dict.joint_to_parent_joint_dict[end_effector]
    link_to_ignores = [human.human_dict.get_dammy_joint_id(end_effector),
                       human.human_dict.get_dammy_joint_id(parent_ee)]
    human_link_robot_collision = [link for link in human_link_robot_collision if link not in link_to_ignores]
    # print("human_link: ", human_link_robot_collision)
    return human_link_robot_collision


def choose_upward_hand(human):
    right_offset = abs(-np.pi / 2 - human.get_roll_wrist_orientation(end_effector="right_hand"))
    left_offset = abs(-np.pi / 2 - human.get_roll_wrist_orientation(end_effector="left_hand"))

    if right_offset > np.pi / 2 and left_offset < np.pi / 2:
        return "left_hand"
    elif right_offset < np.pi / 2 and left_offset > np.pi / 2:
        return "right_hand"
    else:
        return None


def choose_upper_hand(human):
    right_pos = human.get_link_positions(True, end_effector_name="right_hand")
    left_pos = human.get_link_positions(True, end_effector_name="left_hand")
    right_shoulder_z = right_pos[1][2]
    left_shoulder_z = left_pos[1][2]
    print("right_shoulder_z: ", right_shoulder_z, "\nleft_shoudler_z: ", left_shoulder_z)
    diff = right_shoulder_z - left_shoulder_z
    if diff > 0.1:
        return "right_hand"
    elif diff < -0.1:
        return "left_hand"
    else:
        return None


def choose_closer_bedside_hand(env):
    right_dist = cal_dist_to_bedside(env, "right_hand")
    left_dist = cal_dist_to_bedside(env, "left_hand")
    print("right_dist: ", right_dist, "\nleft_dist: ", left_dist)
    return "right_hand" if right_dist < left_dist else "left_hand"


def build_original_human_info(human, env_object_ids, end_effector) -> OriginalHumanInfo:
    # original value
    original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
    original_link_positions = human.get_link_positions(center_of_mass=True, end_effector_name=end_effector)
    original_self_collisions = human.check_self_collision()
    original_env_collisions = human.check_env_collision(env_object_ids)
    original_info = OriginalHumanInfo(original_joint_angles, original_link_positions, original_self_collisions,
                                      original_env_collisions)
    return original_info


def translate_wrt_human_pelvis(human, pos, orient):
    print("pos: ", pos, "orient: ", orient)
    pelvis_pos, pelvis_orient = human.get_pos_orient(human.human_dict.get_fixed_joint_id("pelvis"), center_of_mass=True)
    # print("pelvis_pos: ", pelvis_pos, "pelvis_orient: ", pelvis_orient)
    pelvis_pos_inv, pelvis_orient_inv = p.invertTransform(pelvis_pos, pelvis_orient, physicsClientId=human.id)
    if len(orient) == 0:
        orient = [0, 0, 0, 1]
    else:
        orient = orient if len(orient) == 4 else human.get_quaternion(orient)

    new_pos, new_orient = p.multiplyTransforms(pelvis_pos_inv, pelvis_orient_inv, pos, orient)
    return new_pos, new_orient


def init_optimizer(x0, sigma, lower_bounds, upper_bounds):  # for cmaes library
    opts = {}
    opts['tolfun'] = 1e-2
    opts['tolx'] = 1e-2

    for i in range(x0.size):
        if x0[i] < lower_bounds[i]:
            x0[i] = lower_bounds[i]
        if x0[i] > upper_bounds[i]:
            x0[i] = upper_bounds[i]
    for i in range(len(lower_bounds)):
        if lower_bounds[i] == 0:
            lower_bounds[i] = -1e-9
        if upper_bounds[i] == 0:
            upper_bounds[i] = 1e-9
    # bounds = [lower_bounds, upper_bounds]
    # opts['bounds'] = bounds
    es = CMAEvolutionStrategy(x0, sigma, opts)
    return es


def init_optimizer2(x0, sigma, lower_bounds, upper_bounds):  # for cma library
    # opts = {}
    # opts['tolfun'] = 1e-9
    # opts['tolx'] = 1e-9
    bounds = [[l, u] for l, u in zip(lower_bounds, upper_bounds)]
    bounds = np.array(bounds)
    # print ("bounds: ", bounds.shape, x0.shape, x0.size)
    print("bounds: ", bounds)
    print("x0: ", x0)
    for i in range(x0.size):
        if x0[i] < bounds[i][0]:
            x0[i] = bounds[i][0]
        if x0[i] > bounds[i][1]:
            x0[i] = bounds[i][1]
    es = CMA(x0, sigma, bounds=np.array(bounds))
    return es
