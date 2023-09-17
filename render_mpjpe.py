import time
import os
import pickle as pkl
import numpy as np

from assistive_gym.train import render, render_pose, make_env
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
# from error.error_estimation import mpjpe
from error.heatmap_plot import build_map_pkl

    

SMPL_FILES = [ 's01']
PERSON_IDS = ['p001', 'p002' ]
OBJECTS = ['cane', 'cup', 'pill']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1"
SEED = 1001
SAVE_DIR = 'trained_models'
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 1

def get_dynamic_configs():
    configs =[]
    for p in PERSON_IDS:
        for s in SMPL_FILES:

            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o, s))
    return configs

def do_render(config):
    p, s, o = config
    print (p, s, o)
    render(ENV, p, s, SAVE_DIR, o, ROBOT_IK)

def render_result(env_name, person_id, smpl_file, pose_id='00'):
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=None, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    # smpl_name = os.path.basename(smpl_file).split(".")[0]
    # p.addUserDebugText("person: {}, smpl: {}".format(person_id,smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

    # loading the pose and render in sim
    # actions = pkl.load(open(smpl_file, "rb"))
    # print("actions: ", actions)
    # env.human.set_joint_angles(env.human.controllable_joint_indices, action)

    # collect joint angles to write to pkl file
    humanURDF = HumanUrdfDict()
    joints = ["left_hip", "pelvis", "right_hip", "left_shoulder", # 0 - 3
        "neck", "right_shoulder", "left_elbow", "right_elbow", # 4 - 7
        "left_hand", "right_hand", "left_knee", "right_knee", # 8 - 11
        "left_ankle", "right_ankle", "spine_2", "spine_3", "spine_4"] # 12 - 16
    data = []
    for joint in joints:
        data.append(np.array(env.human.get_link_positions_id(humanURDF.get_dammy_joint_id(joint))))
        print("data: ", data)

    env.disconnect()
    
    print("data: ", data)
    
    # saving to pickle file for comparison
    save = "error/" + str(person_id) + "_" + str(pose_id) + ".pkl"
    # os.makedirs(save, exist_ok=True)
    pkl.dump(data, open(save, "wb"))
    print("file saved")
    # time.sleep(5)

    # env.disconnect()
    slp_body = pkl.load(open("error/p1_s1/s01.pkl", "rb"))
    # slp_body = slp_body['body_pose']
    print("\n\nslp_body: ", slp_body)
    s_body = [np.array([0, 0, 0])] # pre-add pelvis?
    i = 0
    while i + 2 < len(slp_body):
        s_body.append(np.array([slp_body[i], slp_body[i + 1], slp_body[i + 2]]))
        i+=3
    # print("s_body: ", s_body)
    # build_map_pkl([], 0, 0, 0, body_pts=s_body, ag_body_pts=data, go=False)
    build_map_pkl([], 0, 0, 0, body_pts=s_body, go=False)



if __name__ == '__main__':
    configs = get_dynamic_configs()

    displayed = set()
    p, smpl, o, s = configs[0]
    render_result(ENV, p, smpl, pose_id=s)

    # for config in configs:
    #     p, smpl, o, s = config
    #     if (p, smpl) not in displayed:
    #         displayed.add((p, s))
    #     render_result(ENV, p, smpl, pose_id=s)

# Note for me: in env.human.get_link_positions(), try loading in the joints in the order prespecific in error_estimation.py
