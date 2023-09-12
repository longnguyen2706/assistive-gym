import pickle
import time

import argparse
import gym
import importlib

from assistive_gym.envs.utils.dto import RobotSetting
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.point_utils import eulidean_distance
from assistive_gym.envs.utils.train_utils import *
from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_folderpath

LOG = get_logger()

COLLISION_PENETRATION_THRESHOLD = {
    "self_collision": {
        "old": 0.015,  # 1.5cm
        "new": 0.005
    },
    "env_collision": {
        "old": 0.005,  # 1.5cm
        "new": 0.005
    }
}

COLLISION_OBJECT_RADIUS = {
    "pill": 0.0,
    "cup": 0.05,
    "cane": 0.1
}

OBJECT_PALM_OFFSET = {
    "pill": 0.05,
    "cup": 0.1,
    "cane": 0.05
}

MAX_ITERATION = 100


def find_ee_ik_goal(human, end_effector, handover_obj):
    ee_pos, ee_orient = human.get_ee_pos_orient(end_effector)
    ee_norm_vec = human.get_ee_normal_vector(end_effector)
    target_pos = np.array(ee_pos) + ee_norm_vec * OBJECT_PALM_OFFSET[
        handover_obj]  # need to depends on the size of the object as well
    return ee_pos, target_pos


def find_robot_ik_solution(env, end_effector: str, handover_obj: str):
    """
    Find robot ik solution with TOC. Place the robot in best base position and orientation.
    :param env:
    :param end_effector: str
    :param human_link_robot_collision: dict(agent, [link1, link2, ...]) to check for collision with robot
    :return:
    """

    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool

    robot_base_pos, robot_base_orient, side = find_robot_start_pos_orient(env, end_effector)

    ee_pos, target_pos = find_ee_ik_goal(human, end_effector, handover_obj)
    p.addUserDebugLine(ee_pos, target_pos, [1, 0, 0], 5, 0.1)

    best_position, best_orientation, best_joint_angles = robot.position_robot_toc2(robot_base_pos, side,
                                                                                   [(target_pos, None)],
                                                                                   [(target_pos, None)], human,
                                                                                   base_euler_orient=robot_base_orient,
                                                                                   attempts=5,
                                                                                   random_position=0.3,
                                                                                   max_ik_iterations=50,
                                                                                   collision_objects={
                                                                                       furniture: None,
                                                                                       human: None},
                                                                                   tool=tool)

    # TODO: reuse best_poses (ik solution) from toc instead of resolving ik
    is_success, robot_joint_angles, penetrations, dist_to_target, gripper_orient = robot.ik_random_restarts2(right=True,
                                                                                                             target_pos=target_pos,
                                                                                                             target_orient=None,
                                                                                                             max_iterations=100,
                                                                                                             randomize_limits=False,
                                                                                                             collision_objects={
                                                                                                                 furniture: None,
                                                                                                                 human: None},
                                                                                                             tool=tool)
    if is_success:
        # print("robot ik solution found")
        robot.set_joint_angles(robot.right_arm_joint_indices, robot_joint_angles, use_limits=True)
        tool.reset_pos_orient()

    return is_success, robot_joint_angles, best_position, best_orientation, side, penetrations, dist_to_target, gripper_orient


def find_max_val(human, cost_fn, original_joint_angles, original_link_positions, end_effector="right_hand"):
    x0 = np.array(original_joint_angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    timestep = 0
    while not optimizer.stop():
        fitness_values = []
        timestep += 1
        solutions = optimizer.ask()

        for s in solutions:
            if cost_fn == max_torque_cost_fn:
                human.set_joint_angles(human.controllable_joint_indices, s)
                cost = max_torque_cost_fn(human, end_effector)
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

            elif cost_fn == max_manipulibity_cost_fn:
                cost = max_manipulibity_cost_fn(human, end_effector, s)

            elif cost_fn == max_energy_cost_fn:
                human.set_joint_angles(human.controllable_joint_indices, s)
                cost = max_energy_cost_fn(human, end_effector, original_link_positions)
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

            fitness_values.append(cost)
        optimizer.tell(solutions, fitness_values)

    human.set_joint_angles(human.controllable_joint_indices, optimizer.best.x)
    return optimizer.best.x, 1.0 / optimizer.best.f


def find_robot_start_pos_orient(env, end_effector="right_hand"):
    # find bed bb
    bed = env.furniture
    bed_bb = p.getAABB(bed.body, physicsClientId=env.id)
    bed_pos = p.getBasePositionAndOrientation(bed.body, physicsClientId=env.id)[0]

    # find ee pos
    ee_pos, _ = env.human.get_ee_pos_orient(end_effector)
    # print ("ee real pos: ", ee_real_pos)

    # find the side of the bed
    side = "right" if ee_pos[0] > bed_pos[0] else "left"
    bed_xx, bed_yy, bed_zz = bed_bb[1] if side == "right" else bed_bb[0]

    # find robot base and bb
    robot_bb = p.getAABB(env.robot.body, physicsClientId=env.id)
    robot_x_size, robot_y_size, robot_z_size = np.subtract(robot_bb[1], robot_bb[0])
    # print("robot: ", robot_bb)
    base_pos = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]

    # new pos: side of the bed, near end effector, with z axis unchanged
    if side == "right":
        pos = (
            bed_xx + robot_x_size / 2 + 0, ee_pos[1] + robot_y_size / 2,
            base_pos[2])  # TODO: change back to original 0.3
        orient = env.robot.get_quaternion([0, 0, -np.pi / 2])
    else:  # left
        pos = (bed_xx - robot_x_size / 2 - 0, ee_pos[1] + robot_y_size / 2, base_pos[2])
        orient = env.robot.get_quaternion([0, 0, np.pi / 2])
    return pos, orient, side


def get_handover_object_config(object_name, env) -> Optional[HandoverObjectConfig]:
    if object_name == None:  # case: no handover object
        return None

    object_type = HandoverObject.from_string(object_name)
    if object_name == "pill":
        ee = choose_upward_hand(env.human)
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.27], end_effector=ee)  # original = 6
    elif object_name == "cup":
        ee = choose_closer_bedside_hand(env)
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.23], end_effector=ee)  # original = 6
    elif object_name == "cane":
        ee = choose_closer_bedside_hand(env)
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.23], end_effector=ee)  # original = 6


def build_max_human_dynamics(env, end_effector, original_info: OriginalHumanInfo) -> MaximumHumanDynamics:
    """
    build maximum human dynamics by doing CMAES search
    will reset the env after all searches are done

    :param env:
    :param end_effector:
    :param original_info:
    :return:
    """
    human = env.human
    _, max_torque = find_max_val(human, max_torque_cost_fn, original_info.angles, original_info.link_positions,
                                 end_effector)
    _, max_manipubility = find_max_val(human, max_manipulibity_cost_fn, original_info.angles,
                                       original_info.link_positions,
                                       end_effector)
    _, max_energy = find_max_val(human, max_energy_cost_fn, original_info.angles, original_info.link_positions,
                                 end_effector)
    # max_torque, max_manipubility, max_energy = 10, 1, 100
    print("max torque: ", max_torque, "max manipubility: ", max_manipubility, "max energy: ", max_energy)
    max_dynamics = MaximumHumanDynamics(max_torque, max_manipubility, max_energy)

    env.reset()
    return max_dynamics


def detect_collisions(original_info: OriginalHumanInfo, self_collisions, env_collisions, human, end_effector):
    # check collision
    new_self_collision = count_new_collision(original_info.self_collisions, self_collisions, human, end_effector,
                                             COLLISION_PENETRATION_THRESHOLD["self_collision"])
    new_env_collision = count_new_collision(original_info.env_collisions, env_collisions, human, end_effector,
                                            COLLISION_PENETRATION_THRESHOLD["env_collision"])
    LOG.debug(f"self collision: {new_self_collision}, env collision: {new_env_collision}")

    return new_self_collision, new_env_collision


"""
TODO: 
1. step forward
2. check collision and stop on collsion 
3. check if the target angle is reached. break
"""
def step_forward(env, x0, env_object_ids, end_effector="right_hand"):
    human = env.human
    # p.setJointMotorControlArray(human.body, jointIndices=human.controllable_joint_indices,
    #                             controlMode=p.POSITION_CONTROL,
    #                             forces=[1000] * len(human.controllable_joint_indices),
    #                             positionGains=[0.01] * len(human.controllable_joint_indices),
    #                             targetPositions=x0,
    #                             physicsClientId=human.id)
    human.control(human.controllable_joint_indices, x0, 0.01, 100)

    # for _ in range(5):
    #     p.stepSimulation(physicsClientId=env.human.id)
    # p.setRealTimeSimulation(1)
    original_self_collisions = human.check_self_collision(end_effector=end_effector)
    original_env_collisions = human.check_env_collision(env_object_ids, end_effector=end_effector)

    # print ("target: ", x0)

    prev_angle = [0] * len(human.controllable_joint_indices)
    count = 0
    while True:
        p.stepSimulation(physicsClientId=human.id)  # step simulation forward

        self_collision = human.check_self_collision(end_effector=end_effector)
        env_collision = human.check_env_collision(env_object_ids, end_effector=end_effector)
        cur_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
        # print ("cur_joint_angles: ", cur_joint_angles)
        angle_dist = cal_angle_diff(cur_joint_angles, x0)
        count += 1
        if count_new_collision(original_self_collisions, self_collision, human, end_effector,
                               COLLISION_PENETRATION_THRESHOLD["self_collision"]) or count_new_collision(
            original_env_collisions,
            env_collision, human, end_effector, COLLISION_PENETRATION_THRESHOLD["env_collision"]):
            LOG.info(f"{bcolors.FAIL}sim step: {count}, collision{bcolors.ENDC}")
            return angle_dist, self_collision, env_collision, True

        if cal_angle_diff(cur_joint_angles, x0) < 0.05 or cal_angle_diff(cur_joint_angles, prev_angle) < 0.001:
            LOG.info(f"sim step: {count}, angle diff to prev: {cal_angle_diff(cur_joint_angles, prev_angle)}")
            return angle_dist, self_collision, env_collision, False
        prev_angle = cur_joint_angles


def make_env(env_name, person_id, smpl_file, object_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    env.set_smpl_file(smpl_file)

    human_urdf_path = get_urdf_filepath(get_urdf_folderpath(person_id))
    env.set_human_urdf(human_urdf_path)

    task = get_task_from_handover_object(object_name)
    env.set_task(task)
    return env


# TODO: better refactoring for seperating robot-ik/ non robot ik mode
def cost_fn(human, ee_name: str, angle_config: np.ndarray, ee_target_pos: np.ndarray, original_info: OriginalHumanInfo,
            max_dynamics: MaximumHumanDynamics, new_self_collision, new_env_collision, has_valid_robot_ik,
            robot_penetrations, robot_dist_to_target, angle_dist,
            object_config: Optional[HandoverObjectConfig], robot_ik_mode: bool, dist_to_bedside: float):
    # cal energy
    energy_change, energy_original, energy_final = cal_energy_change(human, original_info.link_positions, ee_name)

    # cal dist
    ee_pos, _ = human.get_ee_pos_orient(ee_name)
    dist = eulidean_distance(ee_pos, ee_target_pos)

    # cal torque
    torque = cal_torque_magnitude(human, ee_name)

    # cal manipulibility
    manipulibility = human.cal_chain_manipulibility(angle_config, ee_name)

    # cal angle displacement from mid angle
    mid_angle = cal_mid_angle(human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    mid_angle_displacement = cal_angle_diff(angle_config, mid_angle)
    # print("mid_angle_displacement: ", mid_angle_displacement)

    # cal reba
    reba = human.get_reba_score(end_effector=ee_name)
    max_reba = 9.0

    w = [1, 1, 4, 1, 1]
    cost = None

    if not object_config:  # no object handover case
        cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[
            2] * energy_final / max_dynamics.energy \
                + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement) / np.sum(w)
    else:
        if object_config.object_type == HandoverObject.PILL:
            # cal cost
            cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[
                2] * energy_final / max_dynamics.energy \
                    + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement) / np.sum(w)
        elif object_config.object_type in [HandoverObject.CUP, HandoverObject.CANE]:
            # cal cost
            cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[
                2] * energy_final / max_dynamics.energy \
                    + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement) / np.sum(w)
            if not robot_ik_mode:  # using raycast to calculate cost
                pass

    if new_self_collision:
        cost += 100 * new_self_collision
    if new_env_collision:
        cost += 100 * new_env_collision

    if robot_ik_mode:
        if not has_valid_robot_ik:
            cost += 1000
            # print('No valid ik solution found ', robot_dist_to_target)
            cost += 100 * robot_dist_to_target
        if robot_penetrations:
            # flatten list
            robot_penetrations = [abs(item) for sublist in robot_penetrations for item in sublist]
            # print(robot_penetrations)
            cost += 10 * sum(robot_penetrations)

    return cost, manipulibility, dist, energy_final, torque, reba


def train(env_name, seed=0, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl', person_id='p001',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False,
          robot_ik=False, handover_obj=None):
    start_time = time.time()
    # init
    env = make_env(env_name, person_id, smpl_file, handover_obj, coop=True)
    print("person_id: ", person_id, smpl_file)
    if render:
        env.render()
    env.reset()
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_file), [0, 0, 1], textColorRGB=[1, 0, 0])

    human, robot, furniture, plane = env.human, env.robot, env.furniture, env.plane

    # choose end effector
    handover_obj_config = get_handover_object_config(handover_obj, env)
    if handover_obj_config and handover_obj_config.end_effector:  # reset the end effector based on the object
        human.reset_controllable_joints(handover_obj_config.end_effector)
        end_effector = handover_obj_config.end_effector

    # init collision check
    env_object_ids = [furniture.body, plane.body]  # set env object for collision check
    human_link_robot_collision = get_human_link_robot_collision(human, end_effector)

    # init original info and max dynamics
    original_info = build_original_human_info(human, env_object_ids, end_effector)
    max_dynamics = build_max_human_dynamics(env, end_effector, original_info)

    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])

    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution, mean_reba = [], [], [], [], [], [], []

    # init optimizer
    x0 = np.array(original_info.angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)

    if not robot_ik:  # simulate collision
        ee_link_idx = human.human_dict.get_dammy_joint_id(end_effector)
        ee_collision_radius = COLLISION_OBJECT_RADIUS[handover_obj]  # 20cm range
        ee_collision_body = human.add_collision_object_around_link(ee_link_idx,
                                                                   radius=ee_collision_radius)  # TODO: ignore collision with hand

    smpl_name = os.path.basename(smpl_file)
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

    while timestep < MAX_ITERATION and not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        best_cost, best_angle, best_robot_setting = float('inf'), None, None
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []
        for s in solutions:

            if simulate_collision:
                # step forward env
                angle_dist, _, env_collisions, _ = step_forward(env, s, env_object_ids, end_effector)
                self_collisions = human.check_self_collision()
                new_self_collision, new_env_collision = detect_collisions(original_info, self_collisions,
                                                                          env_collisions, human, end_effector)
                # cal dist to bedside
                dist_to_bedside = cal_dist_to_bedside(env, end_effector)
                cost, m, dist, energy, torque = cost_fn(human, end_effector, s, original_ee_pos, original_info,
                                                        max_dynamics, new_self_collision, new_env_collision,
                                                        has_valid_robot_ik,
                                                        angle_dist, handover_obj_config, robot_ik, dist_to_bedside)
                env.reset_human(is_collision=True)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, angle_dist: {angle_dist} , dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
            else:
                # set angle directly
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle

                # check collision
                env_collisions, self_collisions = human.check_env_collision(
                    env_object_ids), human.check_self_collision()
                new_self_collision, new_env_collision = detect_collisions(original_info, self_collisions,
                                                                          env_collisions, human, end_effector)
                # move_robot(env)
                # cal dist to bedside
                dist_to_bedside = cal_dist_to_bedside(env, end_effector)
                if robot_ik:  # solve robot ik when doing training
                    has_valid_robot_ik, robot_joint_angles, robot_base_pos, robot_base_orient, robot_side, robot_penetrations, robot_dist_to_target, gripper_orient = find_robot_ik_solution(
                        env, end_effector, handover_obj)
                else:
                    ee_collision_body_pos, ee_collision_body_orient = human.get_ee_collision_shape_pos_orient(
                        end_effector, ee_collision_radius)
                    p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos,
                                                      ee_collision_body_orient, physicsClientId=env.id)
                    has_valid_robot_ik = True

                cost, m, dist, energy, torque, reba = cost_fn(human, end_effector, s, original_ee_pos, original_info,
                                                              max_dynamics, new_self_collision, new_env_collision,
                                                              has_valid_robot_ik, robot_penetrations,
                                                              robot_dist_to_target,
                                                              0, handover_obj_config, robot_ik, dist_to_bedside)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
                if cost < best_cost:
                    best_cost = cost
                    best_angle = s
                    best_robot_setting = RobotSetting(robot_base_pos, robot_base_orient, robot_joint_angles, robot_side,
                                                      gripper_orient)
                # restore joint angle
                # human.set_joint_angles(human.controllable_joint_indices, original_info.angles)

                # robot_ee = robot.get_pos_orient(robot.right_end_effector, center_of_mass=True)
                # robot_ee_transform = translate_wrt_human_pelvis(human, robot_ee[0], robot_ee[1])
                #
                # add_debug_line_wrt_parent_frame(robot_ee_transform[0], robot_ee_transform[1], human.body,
                #                           human.human_dict.get_fixed_joint_id("pelvis"))

            fitness_values.append(cost)
            dists.append(dist)
            manipus.append(m)
            energy_changes.append(energy)
            torques.append(torque)

        optimizer.tell(solutions, fitness_values)

        mean_evolution.append(np.mean(solutions, axis=0))
        mean_cost.append(np.mean(fitness_values, axis=0))
        mean_dist.append(np.mean(dists, axis=0))
        mean_m.append(np.mean(manipus, axis=0))
        mean_energy.append(np.mean(energy_changes, axis=0))
        mean_torque.append(np.mean(torques, axis=0))

    LOG.info(
        f"{bcolors.OKBLUE} Best cost: {best_cost}, best cost 2: {optimizer.best.f}, best angle: {best_angle}, best angle2: {optimizer.best.x}, best robot setting: {best_robot_setting}{bcolors.ENDC}, ")
    human.set_joint_angles(env.human.controllable_joint_indices, best_angle)

    robot.set_base_pos_orient(best_robot_setting.base_pos, best_robot_setting.base_orient)
    env.robot.set_joint_angles(
        env.robot.right_arm_joint_indices if best_robot_setting.robot_side == 'right' else env.robot.left_arm_joint_indices,
        best_robot_setting.robot_joint_angles)
    env.tool.reset_pos_orient()
    ee_pos, ik_target_pos = find_ee_ik_goal(human, end_effector, handover_obj)

    action = {
        "solution": optimizer.best.x,
        "cost": cost,
        "end_effector": end_effector,
        "m": m,
        "dist": dist,
        "mean_energy": mean_energy,
        "target": original_ee_pos,
        "mean_cost": mean_cost,
        "mean_dist": mean_dist,
        "mean_m": mean_m,
        "mean_evolution": mean_evolution,
        "mean_torque": mean_torque,
        "mean_reba": mean_reba,
        "robot_settings": {
            "joint_angles": robot_joint_angles,
            "base_pos": robot_base_pos,
            "base_orient": robot_base_orient,
            "side": robot_side
        },
        "wrt_pelvis": {
            'pelvis': human.get_pos_orient(human.human_dict.get_fixed_joint_id("pelvis"), center_of_mass=True),
            "ee": {
                'original': human.get_ee_pos_orient(end_effector),
                'transform': translate_wrt_human_pelvis(human, np.array(human.get_ee_pos_orient(end_effector)[0]),
                                                        np.array(human.get_ee_pos_orient(end_effector)[1])),
            },
            "ik_target": {
                'original': [np.array(ik_target_pos), np.array(gripper_orient)],  # [pos, orient
                'transform': translate_wrt_human_pelvis(human, np.array(ik_target_pos), np.array(gripper_orient)),
            },
            'robot': {
                'original': [np.array(robot_base_pos), np.array(robot_base_orient)],
                'transform': translate_wrt_human_pelvis(human, np.array(robot_base_pos), np.array(robot_base_orient)),
            },
            'robot_joint_angles': robot_joint_angles
        }
    }

    actions = {}
    key = get_actions_dict_key(handover_obj, robot_ik)
    actions[key] = action
    # plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque)
    # plot_mean_evolution(mean_evolution)

    save_train_result(save_dir, env_name, person_id, smpl_file, actions)

    print("training time (s): ", time.time() - start_time)
    env.disconnect()
    return env, actions, action


def save_train_result(save_dir, env_name, person_id, smpl_file, actions):
    save_dir = get_save_dir(save_dir, env_name, person_id, smpl_file)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "actions.pkl")):
        old_actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
        # merge
        if old_actions:
            for key in old_actions.keys():
                if key not in actions.keys():
                    actions[key] = old_actions[key]
    pickle.dump(actions, open(os.path.join(save_dir, "actions.pkl"), "wb"))


def render(env_name, person_id, smpl_file, save_dir, handover_obj, robot_ik: bool):
    print("rendering person {} and smpl file {}".format(person_id, smpl_file))

    save_dir = get_save_dir(save_dir, env_name, person_id, smpl_file)
    actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
    if not actions:
        raise Exception("no actions found for person {} and smpl file {}".format(person_id, smpl_file))
    if handover_obj == "all":
        for key in actions.keys():
            action = actions[key]
            print("key: ", key)
            handover_obj = key.split("-")[0]
            robot_ik = key.split("-")[1] == "robot_ik"
            print("handover obj: ", handover_obj, "robot_ik: ", robot_ik)
            robot_pose, robot_joint_angles = None, None
            try:
                robot_pose = action["wrt_pelvis"]["robot"]['original']
                robot_joint_angles = action["wrt_pelvis"]["robot_joint_angles"]
            except Exception as e:
                print("no robot pose found")

            render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik, robot_pose,
                          robot_joint_angles)
    else:
        key = get_actions_dict_key(handover_obj, robot_ik)
        if key not in actions:
            raise Exception("no action found for ", key)
        action = actions[key]
        robot_pose, robot_joint_angles = None, None

        try:
            robot_pose = action["wrt_pelvis"]["robot"]['original']
            robot_joint_angles = action["wrt_pelvis"]["robot_joint_angles"]
        except Exception as e:
            print("no robot pose found")
        render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik, robot_pose, robot_joint_angles)


def render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik: bool, robot_pose=None,
                  robot_joint_angles=None):
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=handover_obj, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    smpl_name = os.path.basename(smpl_file).split(".")[0]
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

    env.human.reset_controllable_joints(action["end_effector"])
    env.human.set_joint_angles(env.human.controllable_joint_indices, action["solution"])
    if robot_ik:
        print("robot pose: ", robot_pose, "robot_joint_angles: ", robot_joint_angles)
        if robot_pose is None or robot_joint_angles is None:
            find_robot_ik_solution(env, action["end_effector"], handover_obj)
        else:
            # TODO: fix this
            # find_robot_ik_solution(env, action["end_effector"], handover_obj)
            base_pos, base_orient, side = find_robot_start_pos_orient(env, action["end_effector"])
            env.robot.set_base_pos_orient(robot_pose[0], robot_pose[1])
            env.robot.set_joint_angles(
                env.robot.right_arm_joint_indices if side == 'right' else env.robot.left_arm_joint_indices,
                robot_joint_angles)
            env.tool.reset_pos_orient()
        # robot_settings = action["robot_settings"]
        # env.robot.set_base_pos_orient(robot_settings["base_pos"], robot_settings["base_orient"])
        # env.robot.set_joint_angles(env.robot.controllable_joint_indices, robot_settings["joint_angles"])
        # env.tool.reset_pos_orient()
    # plot_cmaes_metrics(action['mean_cost'], action['mean_dist'], action['mean_m'], action['mean_energy'],
    #                    action['mean_torque'])
    # plot_mean_evolution(action['mean_evolution'])

    while True:
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break
    env.disconnect()


def render_pose(env_name, person_id, smpl_file):
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=None, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    while True:
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break
    env.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    # env
    parser.add_argument('--env', default='',
                        help='Environment to train.py on (default: HumanComfort-v1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    # mode
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train.py a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    # train details
    parser.add_argument('--smpl-file', default='examples/data/slp3d/p002/s01.pkl', help='smpl file')
    parser.add_argument('--person-id', default='p002', help='person id')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--render-gui', action='store_true', default=False,
                        help='Whether to render during training')
    parser.add_argument('--end-effector', default='right_hand', help='end effector name')
    parser.add_argument("--simulate-collision", action='store_true', default=False, help="simulate collision")
    parser.add_argument("--robot-ik", action='store_true', default=False, help="solve robot ik during training")
    parser.add_argument("--handover-obj", default=None,
                        help="define if the handover object is default, pill, bottle, or cane")

    # replay
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    # verbosity
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        if args.handover_obj == 'all':  # train all at once
            handover_objs = ['pill', 'cup', 'cane']
            for handover_obj in handover_objs:
                train(args.env, args.seed, args.smpl_file, args.person_id, args.end_effector, args.save_dir,
                      args.render_gui, args.simulate_collision, args.robot_ik, handover_obj)
        else:
            _, actions = train(args.env, args.seed, args.smpl_file, args.person_id, args.end_effector, args.save_dir,
                               args.render_gui, args.simulate_collision, args.robot_ik, args.handover_obj)

    if args.render:
        render(args.env, args.person_id, args.smpl_file, args.save_dir, args.handover_obj, args.robot_ik)
