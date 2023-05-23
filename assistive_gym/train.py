import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle

import numpy as np
from cma import CMA
from numpngw import write_apng
import pybullet as p

def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    pass

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    pass

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def cost_function(env, solution):
    # c_joints = env.human.get_controllable_joints()
    # print ("c_joints: ", c_joints)
    # env.human.set_joint_angles(c_joints, solution)
    return 1000.0/env.human.cal_manipulibility(solution)

def train(env_name, algo, timesteps_total=10, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}):
    env = make_env(env_name, coop=True)
    # agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs)
    # env.disconnect()
    env.reset()
    actions = []
    x0 = np.zeros(len(env.human.controllable_joint_indices)) #no of joints
    sigma = 0.1
    optimizer = init_optimizer(x0, sigma)
    timestep = 0

    while timestep<timesteps_total:
        timestep += 1

        # Train until we have performed the desired number of timesteps
        # env.render()
        solutions = optimizer.ask() # TODO: change this?
        print ("solutions: ", len(solutions))
        optimizer.tell(solutions, [cost_function(env, s) for s in solutions])

        # step forward
        action = {'robot': env.action_space_robot.sample(), 'human': np.mean(solutions, axis=0)}
        actions.append(action)
        env.step(action)

        optimizer.disp()
    optimizer.result_pretty()

    env.disconnect()
    #save action to replay
    print ("actions: ", len(actions))
    pickle.dump(actions, open("actions.pkl", "wb"))

    return env, actions

def init_optimizer(x0, sigma):
    return CMA(x0, sigma)

def render(env, actions):
    print("actions: ", actions)
    env.render() # need to call reset after render
    env.reset()
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.id)
    for a in actions:
        env.step(a)
    # while 1:
    #     env.step({'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()})


def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
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
        _, actions = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)

    if args.render:
        actions = pickle.load(open("actions.pkl", "rb"))
        env = make_env(args.env, coop=True)
        render(env, actions)
        # render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    # if args.evaluate:
    #     evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)

