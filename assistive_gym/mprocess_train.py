import argparse
import multiprocessing
import time
from copy import deepcopy

import numpy

from assistive_gym.envs.utils.dto import RobotSetting, InitRobotSetting
from assistive_gym.envs.utils.train_utils import *

LOG = get_logger()
NUM_WORKERS = 12
MAX_ITERATION = 200


class SubEnvProcess(multiprocessing.Process):
    def __init__(self, id, task_queue, result_queue, env_config, human_conf):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.env_config = env_config
        self.env = None
        self.human_conf = human_conf
        self.debug_id = id

    def run(self):
        while True:
            task = self.task_queue.get()

            if task is None:  # Sentinel value to indicate termination
                break
            result = self.perform_task(task)
            self.result_queue.put(result)

    def perform_task(self, joint_angles):
        if not self.env:
            env_name, person_id, smpl_file, handover_obj, coop = self.env_config
            self.env = make_env(env_name, person_id, smpl_file, handover_obj, coop)
            # self.env.render()
            self.env.reset()
        # print ('joint_angles: ', joint_angles)
        robot_ik, _, original_info, max_dynamics, handover_obj, handover_obj_config, initial_robot_setting = self.human_conf
        # print (handover_obj_config.end_effector)
        original_info, max_dynamics = deepcopy(original_info), deepcopy(max_dynamics)

        env_object_ids = [self.env.furniture.body, self.env.plane.body]
        # print ("pid: ", self.debug_id, env_object_ids, 'original info', original_info)
        conf = (self.env, np.array(joint_angles), robot_ik, env_object_ids, original_info, max_dynamics, handover_obj,
                handover_obj_config, initial_robot_setting)
        cost, m, dist, energy, torque, robot_setting = do_search(conf)
        return (joint_angles, cost, m, dist, energy, torque, robot_setting)


class MainEnvProcess(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, env_config):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.env_config = env_config
        self.env = None

    def run(self):
        while True:
            task = self.task_queue.get()

            if task is None:  # Sentinel value to indicate termination
                break
            result = self.perform_task(task)
            self.result_queue.put(result)

    def perform_task(self, task):
        if not self.env:
            env_name, person_id, smpl_file, handover_obj, coop = self.env_config
            self.env = make_env(env_name, person_id, smpl_file, handover_obj, coop)

        type, angle, robot_setting = task

        if type == 'init':
            print('init main env')
            return init_main_env(self.env, handover_obj)
        if type == 'render_step':
            # print ('render')
            env, human, robot = self.env, self.env.human, self.env.robot

            # print('robot_setting: ', robot_setting.robot_side, robot_setting.robot_joint_angles)
            if len(robot_setting.robot_joint_angles) == 0:
                return False
            human.set_joint_angles(human.controllable_joint_indices, angle)
            render_robot(env, robot_setting)
            return True

def render_robot(env, robot_setting):
    # print('render robot', robot_setting.robot_joint_angles)
    env.robot.set_base_pos_orient(robot_setting.base_pos, robot_setting.base_orient)
    env.robot.set_joint_angles(
        env.robot.right_arm_joint_indices if robot_setting.robot_side == 'right' else env.robot.left_arm_joint_indices,
        robot_setting.robot_joint_angles)
    env.tool.reset_pos_orient()
def init_main_env(env, handover_obj):
    env.reset()

    # time.sleep(100)
    human, robot, furniture, plane = env.human, env.robot, env.furniture, env.plane

    # choose end effector
    handover_obj_config = get_handover_object_config(handover_obj, env)
    if handover_obj_config and handover_obj_config.end_effector:  # reset the end effector based on the object
        human.reset_controllable_joints(handover_obj_config.end_effector)
        end_effector = handover_obj_config.end_effector

    robot_base, robot_orient, robot_side = find_robot_start_pos_orient(env, end_effector)
    robot_setting = InitRobotSetting(robot_base, robot_orient, robot_side)
    # init collision check
    env_object_ids = [furniture.body, plane.body]  # set env object for collision check
    human_link_robot_collision = get_human_link_robot_collision(human, end_effector)

    # init original info and max dynamics
    original_info = build_original_human_info(human, env_object_ids, end_effector)
    max_dynamics = build_max_human_dynamics(env, end_effector, original_info)

    if render:
        env.render()
    env.reset()
    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])
    original_info.original_ee_pos = original_ee_pos  # TODO: refactor

    return original_info, max_dynamics, env_object_ids, human_link_robot_collision, end_effector, handover_obj_config, \
        human.controllable_joint_lower_limits, human.controllable_joint_upper_limits, robot_setting


def do_search(conf):
    env, s, robot_ik, env_object_ids, original_info, max_dynamics, handover_obj, handover_obj_config, init_robot_setting = conf
    human, end_effector = env.human, handover_obj_config.end_effector
    # print("s: ", s, 'human', human.controllable_joint_indices, 'end_effector', end_effector, 'handover_obj', handover_obj)
    # set angle directly
    human.reset_controllable_joints(end_effector)
    human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle

    # check collision
    env_collisions, self_collisions = human.check_env_collision(env_object_ids,
                                                                end_effector), human.check_self_collision(end_effector)
    new_self_penetrations, new_env_penetrations = detect_collisions(original_info, self_collisions, env_collisions,
                                                                    human,
                                                                    end_effector)
    # print ('end_effector', end_effector)
    # move_robot(env)
    # move_robot(env)
    # cal dist to bedside
    dist_to_bedside = cal_dist_to_bedside(env, end_effector)
    if robot_ik:  # solve robot ik when doing training
        has_valid_robot_ik, robot_joint_angles, robot_base_pos, robot_base_orient, robot_side, robot_penetrations, robot_dist_to_target, gripper_orient = find_robot_ik_solution(env,
                                                                                                         end_effector,
                                                                                                         handover_obj, init_robot_setting)
    else:
        ee_link_idx = human.human_dict.get_dammy_joint_id(end_effector)
        ee_collision_radius = COLLISION_OBJECT_RADIUS[handover_obj]  # 20cm range
        ee_collision_body = human.add_collision_object_around_link(ee_link_idx,
                                                                   radius=ee_collision_radius)  # TODO: ignore collision with hand`

        ee_collision_body_pos, ee_collision_body_orient = human.get_ee_collision_shape_pos_orient(end_effector,
                                                                                                  ee_collision_radius)
        p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos, ee_collision_body_orient,
                                          physicsClientId=env.id)
        has_valid_robot_ik = True

    cost, m, dist, energy, torque = cost_fn(human, end_effector, s, original_info.original_ee_pos, original_info,
                                                  max_dynamics, new_self_penetrations, new_env_penetrations,
                                                  has_valid_robot_ik, robot_penetrations, robot_dist_to_target,
                                                  0, handover_obj_config, robot_ik, dist_to_bedside)


    robot_setting = RobotSetting(robot_base_pos, robot_base_orient, robot_joint_angles, robot_side,
                                      gripper_orient)
    print ("sub process ", robot_setting.robot_joint_angles)
    # restore joint angle
    # human.set_joint_angles(human.controllable_joint_indices, original_info.angles)
    return cost, m, dist, energy, torque, robot_setting

def init_main_env_process(env_config):
    # init main env process
    main_env_task_queue = multiprocessing.Queue()
    main_env_result_queue = multiprocessing.Queue()

    main_env_process = MainEnvProcess(main_env_task_queue, main_env_result_queue, env_config)
    main_env_process.start()

    return main_env_process, main_env_task_queue, main_env_result_queue

def init_sub_env_process(env_config, search_config):
    sub_env_task_queue = multiprocessing.Queue()
    sub_env_result_queue = multiprocessing.Queue()

    sub_env_workers = [SubEnvProcess(id, sub_env_task_queue, sub_env_result_queue, env_config, search_config) for id in
                       range(NUM_WORKERS)]
    for w in sub_env_workers:
        w.start()
    return sub_env_workers, sub_env_task_queue, sub_env_result_queue

def destroy_sub_env_process(sub_env_workers, sub_env_task_queue):
    # destroy sub env processes
    for _ in range(NUM_WORKERS):
        sub_env_task_queue.put(None)
    for w in sub_env_workers:
        w.join()

def mp_train(env_name, seed=0, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl', person_id='p001',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False,
          robot_ik=False, handover_obj=None):
    start_time = time.time()
    env_config = (env_name, person_id, smpl_file, handover_obj, True)

    # init main env process
    main_env_process, main_env_task_queue, main_env_result_queue = init_main_env_process(env_config)
    main_env_task_queue.put(('init', None, None))
    init_result = main_env_result_queue.get()
    original_info, max_dynamics, env_object_ids, human_link_robot_collision, end_effector, handover_obj_config, \
        controllable_joint_lower_limits, controllable_joint_upper_limits, initial_robot_setting = init_result

    # init sub env processes
    search_config = (robot_ik, env_object_ids, original_info, max_dynamics, handover_obj, handover_obj_config, initial_robot_setting)
    sub_env_workers, sub_env_task_queue, sub_env_result_queue = init_sub_env_process(env_config, search_config)

    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution, mean_reba = [], [], [], [], [], [], []

    # init optimizer
    x0 = np.array(original_info.angles)
    optimizer = init_optimizer(x0, 0.1, controllable_joint_lower_limits, controllable_joint_upper_limits)

    best_cost, best_angle, best_robot_setting = float('inf'), None, None
    while timestep < MAX_ITERATION and not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []

        for s in solutions:
            sub_env_task_queue.put(s)

        for _ in solutions:
            result = sub_env_result_queue.get()
            joint_angles, cost, dist, m, energy, torque, robot_setting = result
            # print (result)
            fitness_values.append(cost)
            dists.append(dist)
            manipus.append(m)
            energy_changes.append(energy)
            torques.append(torque)
            if cost < best_cost:
                best_cost = cost
                best_angle = joint_angles
                best_robot_setting = robot_setting
            print('best_cost: ', best_cost)
            main_env_task_queue.put(('render_step', joint_angles, robot_setting))
            main_env_result_queue.get()
        optimizer.tell(solutions, fitness_values)

        mean_evolution.append(np.mean(solutions, axis=0))
        mean_cost.append(np.mean(fitness_values, axis=0))
        mean_dist.append(np.mean(dists, axis=0))
        mean_m.append(np.mean(manipus, axis=0))
        mean_energy.append(np.mean(energy_changes, axis=0))
        mean_torque.append(np.mean(torques, axis=0))

    destroy_sub_env_process(sub_env_workers, sub_env_task_queue)
    # LOG.info(
    #     f"{bcolors.OKBLUE} Best cost: {optimizer.best.f}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")

    main_env_task_queue.put(('render_step', best_angle, best_robot_setting))

    main_env_result_queue.get()
    LOG.info(
        f"{bcolors.OKBLUE} Best cost: {optimizer.best.f} {best_cost} {bcolors.ENDC}")
    time.sleep(100)
    main_env_task_queue.put(None)
    main_env_process.join()
    # action = {
    #     "solution": optimizer.best.x,
    #     "cost": cost,
    #     "end_effector": end_effector,
    #     "m": m,
    #     "dist": dist,
    #     "mean_energy": mean_energy,
    #     "target": original_ee_pos,
    #     "mean_cost": mean_cost,
    #     "mean_dist": mean_dist,
    #     "mean_m": mean_m,
    #     "mean_evolution": mean_evolution,
    #     "mean_torque": mean_torque,
    #     "mean_reba": mean_reba,
    #     "robot_settings": {
    #         "joint_angles": robot_joint_angles,
    #         "base_pos": robot_base_pos,
    #         "base_orient": robot_base_orient,
    #         "side": robot_side
    #     }
    # }
    #
    # actions = {}
    # key = get_actions_dict_key(handover_obj, robot_ik)
    # actions[key] = action
    # # plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque)
    # # plot_mean_evolution(mean_evolution)
    #
    # env.disconnect()

    # save_train_result(save_dir, env_name, person_id, smpl_file, actions)

    print("training time (s): ", time.time() - start_time)
    return _, actions


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
