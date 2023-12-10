import pickle
import time

import argparse

from assistive_gym.envs.utils.dto import RobotSetting
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.point_utils import eulidean_distance
from assistive_gym.envs.utils.train_utils import *
from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_folderpath

import argparse
import json
import multiprocessing
import os
import time
from assistive_gym.envs.utils.dto import RobotSetting, InitRobotSetting, EnvConfig, SearchConfig, MainEnvInitResult, \
    SearchResult, MainEnvProcessInitTask, MainEnvProcessTask, MainEnvProcessTaskType, MainEnvProcessRenderTask, \
    MainEnvProcessGetHumanRobotInfoTask, TrialResult, HandoverValidity, BestKinematicResult, MeanKinematicResult
from assistive_gym.envs.utils.train_utils import *


LOG = get_logger()
MAX_ITERATION = 500
RENDER_UI = False
MAX_TRIAL = 1

def cal_object_specific_cost(env, handover_object, bedside, end_effector):
    if handover_object == 'cup':
        return get_gripper_z_angle_cost(env, bedside, GRIPPER_Z_ANGLE_LIMIT['cup'])
    elif handover_object == 'cane':
        return cal_ee_bedside_dist_cost(env, bedside, end_effector, GRIPPER_BEDSIDE_OFFSET['cane'])
    else:
        return 0
    
def render_robot(env, robot_setting):
    # print('render robot', robot_setting.robot_joint_angles)
    env.robot.set_base_pos_orient(robot_setting.base_pos, robot_setting.base_orient)
    env.robot.set_joint_angles(
        env.robot.right_arm_joint_indices if robot_setting.robot_side == 'right' else env.robot.left_arm_joint_indices,
        robot_setting.robot_joint_angles)
    env.tool.reset_pos_orient()

def check_validity(handover_validity: HandoverValidity):
    if len(handover_validity.new_env_penetrations) or len(handover_validity.new_self_penetrations):
        return False
    return True

def get_human_robot_info(env, result: TrialResult, env_config: EnvConfig): 
    human, robot= env.human, env.robot
    robot_setting, end_effector, handover_obj = result.robot_setting, env_config.end_effector, env_config.handover_obj
    
    human.set_joint_angles(human.controllable_joint_indices, result.joint_angles)
    render_robot(env, robot_setting)
    _, ik_target_pos = find_ee_ik_goal(human, end_effector, handover_obj)

    return {
        'pelvis': human.get_pos_orient(human.human_dict.get_fixed_joint_id("pelvis"), center_of_mass=True),
        'joint_angles': result.joint_angles,
        "ee": {
            'original': human.get_ee_pos_orient(end_effector),
            'transform': translate_wrt_human_pelvis(human, np.array(
                human.get_ee_pos_orient(end_effector)[0]),
                                                    np.array(
                                                        human.get_ee_pos_orient(end_effector)[
                                                            1])),
        },
        "ik_target": {
            'original': [np.array(ik_target_pos), np.array(robot_setting.gripper_orient)],  # [pos, orient
            'transform': translate_wrt_human_pelvis(human, np.array(ik_target_pos),
                                                    np.array(robot_setting.gripper_orient)),
        },
        'robot': {
            'original': [np.array(robot_setting.base_pos), np.array(robot_setting.base_orient)],
            'transform': translate_wrt_human_pelvis(human, np.array(robot_setting.base_pos),
                                                    np.array(robot_setting.base_orient)),
        },
        'robot_joint_angles': robot_setting.robot_joint_angles
    }


def get_search_result(env, joint_angles, search_config: SearchConfig):
    # TODO: remove reset_controllable_joints
    human, end_effector, handover_obj = env.human, search_config.handover_obj_config.end_effector, search_config.handover_obj
    # print("s: ", s, 'human', human.controllable_joint_indices, 'end_effector', end_effector, 'handover_obj', handover_obj)
    # set angle directly
    human.reset_controllable_joints(end_effector)
    human.set_joint_angles(human.controllable_joint_indices, joint_angles)  # force set joint angle

    # check collision
    env_collisions, self_collisions = human.check_env_collision(search_config.env_object_ids,
                                                                end_effector), human.check_self_collision(end_effector)
    new_self_penetrations, new_env_penetrations = detect_collisions(search_config.original_info, self_collisions,
                                                                    env_collisions,
                                                                    human,
                                                                    end_effector)
    # print ('end_effector', end_effector)
    # cal dist to bedside
    object_specific_cost = cal_object_specific_cost(env, handover_obj, search_config.initial_robot_setting.robot_side,
                                                    end_effector)
    if search_config.robot_ik:  # solve robot ik when doing training
        has_valid_robot_ik, robot_joint_angles, robot_base_pos, robot_base_orient, robot_side, robot_penetrations, robot_dist_to_target, gripper_orient = find_robot_ik_solution(
            env,
            end_effector,
            handover_obj, search_config.initial_robot_setting)
    else:
        ee_link_idx = human.human_dict.get_dammy_joint_id(end_effector)
        ee_collision_radius = COLLISION_OBJECT_RADIUS[search_config.handover_obj]  # 20cm range
        ee_collision_body = human.add_collision_object_around_link(ee_link_idx,
                                                                   radius=ee_collision_radius)  # TODO: ignore collision with hand`

        ee_collision_body_pos, ee_collision_body_orient = human.get_ee_collision_shape_pos_orient(end_effector,
                                                                                                  ee_collision_radius)
        p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos, ee_collision_body_orient,
                                          physicsClientId=env.id)
        has_valid_robot_ik = True

    cost, m, dist, energy, torque = cost_func(human, end_effector, joint_angles,
                                              search_config.original_info.original_ee_pos, search_config.original_info,
                                              search_config.max_dynamics, new_self_penetrations, new_env_penetrations,
                                              has_valid_robot_ik, robot_penetrations, robot_dist_to_target,
                                              0, search_config.handover_obj_config, search_config.robot_ik,
                                              object_specific_cost)

    robot_setting = RobotSetting(robot_base_pos, robot_base_orient, robot_joint_angles, robot_side,
                                 gripper_orient)
    handover_validity = HandoverValidity(new_self_penetrations, new_env_penetrations, robot_penetrations,
                                         robot_dist_to_target)
    # print ("sub process ", robot_setting.robot_joint_angles)
    # restore joint angle
    # human.set_joint_angles(human.controllable_joint_indices, original_info.angles)
    return SearchResult(joint_angles, cost, m, dist, energy, torque, robot_setting, handover_validity)


def run_trial(env, init_result: MainEnvInitResult, search_config: SearchConfig):
    # print (init_result, search_config)
    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution, mean_reba = [], [], [], [], [], [], []
    # init optimizer
    x0 = np.array(init_result.original_info.angles)
    optimizer = init_optimizer(x0, 0.05, init_result.joint_lower_limits, init_result.joint_upper_limits)

    best_cost, best_angle, best_robot_setting = float('inf'), None, None
    validity_count = 0

    while timestep < MAX_ITERATION and not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []
            

        for joint_angles in solutions:
            sr: SearchResult = get_search_result(env, joint_angles, search_config)
            # print (result)
            fitness_values.append(sr.cost)
            dists.append(sr.dist)
            manipus.append(sr.manipulability)
            energy_changes.append(sr.energy)
            torques.append(sr.torque)

            if check_validity(sr.handover_validity):
                validity_count += 1
            if sr.cost < best_cost and check_validity(sr.handover_validity):
                best_cost = sr.cost
                best_angle = sr.joint_angles
                best_robot_setting = sr.robot_setting
        if timestep%10 ==0: 
            LOG.info ("step: ", timestep)
        optimizer.tell(solutions, fitness_values)

        mean_evolution.append(np.mean(solutions, axis=0))
        mean_cost.append(np.mean(fitness_values, axis=0))
        mean_dist.append(np.mean(dists, axis=0))
        mean_m.append(np.mean(manipus, axis=0))
        mean_energy.append(np.mean(energy_changes, axis=0))
        mean_torque.append(np.mean(torques, axis=0))

        if timestep >50 and validity_count / len(solutions) / timestep < 0.25:  # stuck
            LOG.info(
                f"{bcolors.OKBLUE} Stuck at step: {timestep}, validity count: {validity_count}, validity ratio: {validity_count / len(solutions) / timestep}, {bcolors.ENDC}")
            break

    # # get the kinematic result for best solution
    sr: SearchResult = get_search_result(env, best_angle, search_config)
    best_dist, best_m, best_energy, best_torque = sr.dist, sr.manipulability, sr.energy, sr.torque

    best_kinematic_result = BestKinematicResult(best_energy, best_cost, best_dist, best_m, best_torque)
    mean_kinematic_result = MeanKinematicResult(mean_energy, mean_cost, mean_dist, mean_m, mean_torque, mean_evolution)
    result = TrialResult(best_angle, best_kinematic_result, mean_kinematic_result, sr.robot_setting,
                         sr.handover_validity)

    LOG.info(
        f"{bcolors.OKBLUE} Best cost: {optimizer.best.f} {best_cost} {bcolors.ENDC}")
    return result

def init_env(env, handover_obj): 
    
    env.reset()

    # time.sleep(100)
    human, robot, furniture, plane = env.human, env.robot, env.furniture, env.plane

    # choose end effector
    handover_obj_config = get_handover_object_config(handover_obj, env)
    # if handover_obj_config and handover_obj_config.end_effector:  # reset the end effector based on the object
    #     human.reset_controllable_joints(handover_obj_config.end_effector)
    #     end_effector = handover_obj_config.end_effector
    # reset the end effector based on the object
    end_effector = handover_obj_config.end_effector
    human.reset_controllable_joints(end_effector)
    robot_base, robot_orient, robot_side = find_robot_start_pos_orient(env, end_effector)

    print ("end effector: ", end_effector)
    robot_setting = InitRobotSetting(robot_base, robot_orient, robot_side)
    # init collision check
    env_object_ids = [furniture.body, plane.body]  # set env object for collision check
    human_link_robot_collision = get_human_link_robot_collision(human, end_effector)

    # init original info and max dynamics
    original_info = build_human_info(human, env_object_ids, end_effector)
    max_dynamics = build_max_human_dynamics(env, end_effector, original_info)

    if RENDER_UI:
        env.render()
    env.reset()
    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])
    original_info.original_ee_pos = original_ee_pos  # TODO: refactor

    return MainEnvInitResult(original_info, max_dynamics, env_object_ids, human_link_robot_collision, end_effector,
                             handover_obj_config,
                             human.controllable_joint_lower_limits, human.controllable_joint_upper_limits,
                             robot_setting)

def train(env_name, seed=0, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl', person_id='p001',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False,
          robot_ik=False, handover_obj=None):
    start_time = time.time()
    env = make_env(env_name, person_id, smpl_file, handover_obj, True)
    try: 
        init_result: MainEnvInitResult = init_env(env, handover_obj)

        env_config: EnvConfig =  EnvConfig(env_name, person_id, smpl_file, handover_obj, init_result.end_effector, True)

        # init sub env processes
        search_config = SearchConfig(robot_ik, init_result.env_object_ids, init_result.original_info,
                                    init_result.max_dynamics, handover_obj,
                                    init_result.handover_obj_config, init_result.robot_setting)


        # smpl_name = os.path.basename(smpl_file)
        # p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

        # best_mean_cost,  best_mean_dist, best_mean_m, best_mean_energy, best_mean_torque, best_mean_evolution, best_mean_reba = [], [], [], [], [], [], []
        best_trial_cost, best_trial_result = float('inf'), None
        for i in range(MAX_TRIAL):
            result = run_trial(env, init_result, search_config)
            if result.best_kinematic_result.cost < best_trial_cost:
                best_trial_cost = result.best_kinematic_result.cost
                best_trial_result = result
        
        human_robot_info = get_human_robot_info(env, best_trial_result, env_config)
        action = {
            "solution": best_trial_result.joint_angles,
            "cost": best_trial_result.best_kinematic_result.cost,
            "end_effector": end_effector,
            "m": best_trial_result.best_kinematic_result.m,
            "dist": best_trial_result.best_kinematic_result.dist,
            "energy": best_trial_result.best_kinematic_result.energy,
            "torque": best_trial_result.best_kinematic_result.torque,
            "mean_energy": best_trial_result.mean_kinematic_result.mean_energy,
            # "target": init_result.original_info.original_ee_pos,
            "mean_cost": best_trial_result.mean_kinematic_result.mean_cost,
            "mean_dist": best_trial_result.mean_kinematic_result.mean_dist,
            "mean_m": best_trial_result.mean_kinematic_result.mean_m,
            "mean_evolution": best_trial_result.mean_kinematic_result.mean_evolution,
            "mean_torque": best_trial_result.mean_kinematic_result.mean_torque,
            "initial_robot_settings": init_result.robot_setting,
            "wrt_pelvis": human_robot_info,
            "validity": best_trial_result.handover_validity
        }

        # print('human_robot_info:', human_robot_info)

        actions = {}
        key = get_actions_dict_key(handover_obj, robot_ik)
        actions[key] = action
        # # plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque)
        # # plot_mean_evolution(mean_evolution)
        save_train_result(save_dir, env_name, person_id, smpl_file, actions, key, handover_obj)

        print("training time (s): ", time.time() - start_time)
        return action
    except Exception as e:
        return e 
    finally: 
        destroy_env(env)
    


def save_train_result(save_dir, env_name, person_id, smpl_file, actions, key, handover_obj):
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

    dumped = json.dumps(actions[key]['wrt_pelvis'], cls=NumpyEncoder)
    with open(os.path.join(save_dir, handover_obj + ".json"), "w") as f:
        f.write(dumped)
    f.close()


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