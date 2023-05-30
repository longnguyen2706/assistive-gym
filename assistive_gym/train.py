import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle
import time

import numpy as np
from cma import CMA, CMAEvolutionStrategy
from numpngw import write_apng
import pybullet as p
from matplotlib import pyplot as plt


def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    pass


def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    pass


def uniform_sample(pos, radius, num_samples):
    """
    Sample points uniformly from the given space
    :param pos: (x, y, z)
    :return:
    """
    # pos = np.array(pos)
    # points = np.random.uniform(low=pos-radius, high=pos + radius, size=(num_samples, 3))
    points = []
    for i in range(num_samples):
        r = np.random.uniform(radius / 2, radius)
        theta = np.random.uniform(0, np.pi / 2)
        phi = np.random.uniform(0, np.pi / 2)  # Only sample from 0 to pi/2

        # Convert from spherical to cartesian coordinates
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)

        # Add to original point
        x_new = pos[0] + dx
        y_new = pos[1] + dy
        z_new = pos[2] + dz
        points.append([x_new, y_new, z_new])
    return points


def draw_point(point, size=0.01):
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
    multiBody = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=sphere,
                                  basePosition=np.array(point))
    p.setGravity(0, 0, 0, multiBody)


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env


def solve_ik(env, target_pos, end_effector="right_hand"):
    human = env.human
    ee_idx = human.human_dict.get_dammy_joint_id(end_effector)
    ik_joint_indices = human.find_ik_joint_indices()

    print ("ik_joint_indices: ", ik_joint_indices)
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices,  max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution
    # ik_joint_poses = np.array(p.calculateInverseKinematics(human.body, ee_idx, targetPosition=target_pos,
    #                                                        maxNumIterations=10000, physicsClientId=human.id))
    # print('IK:', ik_joint_poses)
    # return ik_joint_poses[ik_joint_indices]


def cost_function(env, solution, target_pos, end_effector="right_hand"):
    human = env.human
    m = human.cal_manipulibility(solution, [end_effector])

    right_hand_ee = human.human_dict.get_dammy_joint_id(end_effector)
    # ee_positions, _ = human.forward_kinematic([right_hand_ee], solution)

    pos, _ = human.fk([end_effector], solution)
    dist = eulidean_distance(pos[0], target_pos)
    # cost = 10 / m + dist
    cost = dist
    print("manipubility: ", m, "distance: ", dist, "cost: ", cost)
    return cost, m, dist


def eulidean_distance(cur, target):
    print("current: ", cur, "target: ", target)
    # convert tuple to np array
    cur = np.array(cur)
    return np.sqrt(np.sum(np.square(cur - target)))


def generate_target_points(env):
    # init points
    # human_pos = p.getBasePositionAndOrientation(env.human.body, env.human.id)[0]
    # points = uniform_sample(human_pos, 0.5, 20)
    human = env.human
    right_hand_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id("right_hand"))[0]
    # points = uniform_sample(right_hand_pos, 0.3, 1)
    point = np.array(list(right_hand_pos))
    point[1] += 0.0
    point[0] += 0.2
    point[2] += 0.4
    points = [point]
    return points


def plot(vals, title, xlabel, ylabel):
    plt.figure()
    plt.plot(vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_CMAES_metrics(mean_cost, mean_dist, mean_m):
    # Plot the fitness values
    plot(mean_cost, "Fitness Values", "Iteration", "Fitness Value")

    # Plot the distance values
    plot(mean_dist, "Distance Values", "Iteration", "Distance Value")

    # Plot the manipubility values
    plot(mean_m, "Manipubility Values", "Iteration", "Manipubility Value")

def plot_mean_evolution(mean_evolution):
    # Plot the mean vector evolution
    mean_evolution = np.array(mean_evolution)
    plt.figure()
    for i in range(mean_evolution.shape[1]):
        plt.plot(mean_evolution[:, i], label=f"Dimension {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Value")
    plt.title("Mean Vector Evolution")
    plt.legend()
    plt.show()
def train(env_name, algo, timesteps_total=10, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0,
          extra_configs={}):
    env = make_env(env_name, coop=True)
    # agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs)
    # env.disconnect()
    env.render()
    env.reset()

    # init points
    points = generate_target_points(env)
    pickle.dump(points, open("points.pkl", "wb"))

    actions = {}
    best_action_idx = 0
    best_cost = 10 ^ 9
    cost = 0
    for (idx, target) in enumerate(points):
        draw_point(target, size=0.05)
        # x0 = np.zeros(len(env.human.controllable_joint_indices))  # no of joints
        x0 = solve_ik(env, target, end_effector="right_hand")
        # print("x0: ", x0.shape)
        # env.step({'robot': env.action_space_robot.sample(), 'human': x0})
        # for _ in range(5):
        #     p.stepSimulation()
        ee_pos, _= env.human.fk(["right_hand_limb"], x0)
        print("ik error 2: ", eulidean_distance(ee_pos[0], target))

        # right_hand_ee = env.human.human_dict.get_dammy_joint_id("right_hand")
        # ee_positions, _ = env.human.forward_kinematic([right_hand_ee], x0)
        # print("ik error: ", eulidean_distance(ee_positions[0], target))

        # for _ in range(1000):
        #     p.stepSimulation()
        # time.sleep(100)
        optimizer = init_optimizer(x0, sigma=0.1)

        timestep = 0
        actions[idx] = []
        mean_evolution = []
        dists = []
        manipus = []
        mean_cost = []
        mean_dist = []
        mean_m = []
        #
        while not optimizer.stop():
            timestep += 1
            solutions = optimizer.ask()  # TODO: change this?
            # print("solutions: ", solutions)
            fitness_values = []
            for s in solutions:
                cost, m, dist = cost_function(env, s, target)
                fitness_values.append(cost)
                dists.append(dist)
                manipus.append(m)
            optimizer.tell(solutions, fitness_values)
            # env.reset_human()
            # step forward
            action = {'robot': env.action_space_robot.sample(), 'human': np.mean(solutions, axis=0)}
            actions[idx].append(action)
            mean_evolution.append(action['human'])

            # env.step(action)
            # cost = cost_function(env, action['human'], target)
            print("timestep: ", timestep, "cost: ", cost)
            # optimizer.disp()
            cost = optimizer.best.f
            optimizer.result_pretty()
            mean_cost.append(np.mean(fitness_values))
            mean_dist.append(np.mean(dists))
            mean_m.append(np.mean(manipus))

        plot_CMAES_metrics(mean_cost, mean_dist, mean_m)
        plot_mean_evolution(mean_evolution)

        if cost < best_cost:
            best_cost = cost
            best_action_idx = idx

    env.disconnect()
    # save action to replay
    print("actions: ", len(actions))
    pickle.dump(actions, open("actions.pkl", "wb"))
    pickle.dump(best_action_idx, open("best_action_idx.pkl", "wb"))

    return env, actions


def init_optimizer(x0, sigma):
    es = CMAEvolutionStrategy(x0, sigma)
    # es.stop(termination_callback=cma.stoppers.VarianceStopping(tolx=1e-5, tolfun=1e-5))
    return es


def render(env, actions):
    # print("actions: ", actions)
    env.render()  # need to call reset after render
    env.reset()

    # init points
    points = pickle.load(open("points.pkl", "rb"))
    best_idx = pickle.load(open("best_action_idx.pkl", "rb"))
    for (idx, point) in enumerate(points):
        print(idx, point)

        if idx == best_idx:
            draw_point(point, size=0.05)
        else:
            draw_point(point)
    for a in actions[best_idx]:
        env.step(a)


def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60,
                             camera_width=1920 // 4, camera_height=1080 // 4)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    if not colab:
        env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
    env.disconnect()
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train.py on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train.py a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=10,
                        help='Number of simulation timesteps to train.py a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        _, actions = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir,
                           load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)

    if args.render:
        actions = pickle.load(open("actions.pkl", "rb"))
        env = make_env(args.env, coop=True)
        render(env, actions)
        # render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    # if args.evaluate:
    #     evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)
