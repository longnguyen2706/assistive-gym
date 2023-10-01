import time
import os
import pickle as pkl
import numpy as np

from assistive_gym.train import render, render_pose, make_env
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from error.error_estimation import mpjpe
from error.heatmap_plot import build_map_pkl
from assistive_gym.envs.utils.smpl_geom import generate_geom2
from assistive_gym.envs.utils.urdf_utils import SMPLData, load_smpl

    

PERSON_IDS = ['p002']
SMPL_FILES = [ 's01', 's02', 's03']
OBJECTS = [None]
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
    print("person_id: ", person_id, "smpl_file: ", smpl_file)
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=None, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    # collect joint angles to write to pkl file
    humanURDF = HumanUrdfDict()
    joints = ["left_hip", "pelvis", "right_hip", "left_shoulder", # 0 - 3
        "neck", "right_shoulder", "left_elbow", "right_elbow", # 4 - 7
        "left_hand", "right_hand", "left_knee", "right_knee", # 8 - 11
        "left_ankle", "right_ankle", "spine_2", "spine_3", "spine_4"] # 12 - 16
    data = []
    for joint in joints:
        if joint == "pelvis": data.append(np.array(env.human.get_link_positions_id(0, center_of_mass=False)))
        else:
            print("joint: ", joint, " -> dammy joint id: ", humanURDF.get_dammy_joint_id(joint))
            data.append(np.array(env.human.get_link_positions_id(humanURDF.get_dammy_joint_id(joint), center_of_mass=False)))
            print(joint, ": ", env.human.get_link_positions_id(humanURDF.get_dammy_joint_id(joint)))


    env.disconnect()
    
    # env.disconnect()
    slp_body = pkl.load(open("error/" + person_id + "/" + pose_id + ".pkl", "rb"))
    smpl_path = "examples/data/SMPL_FEMALE.pkl"
    smpl_data = SMPLData(slp_body['body_pose'], slp_body['betas'], slp_body['global_orient'])
    
    _, joint_pos, _ = generate_geom2(smpl_path, smpl_data, "assistive_gym/error/error_geom")
    s_body = []
    joints = ["L_Hip", "Pelvis", "R_Hip", "L_Shoulder", # 0 - 3
        "Neck", "R_Shoulder", "L_Elbow", "R_Elbow", # 4 - 7
        "L_Hand", "R_Hand", "L_Knee", "R_Knee", # 8 - 11
        "L_Ankle", "R_Ankle", "Spine1", "Spine2", "Spine3"] # 12 - 16
    for joint in joints:
        s_body.append(joint_pos[joint])
    build_map_pkl([], 0, 0, 0, smpl_ag_body_pts=s_body, ag_body_pts=data, go=False)

    file_w = open("error/mpjpe_values.txt", "a")
    file_w.write(person_id + "/" + pose_id + ".pkl: " + str(mpjpe(s_body, data)))
    file_w.close()
    print("mpjpe: ", mpjpe(s_body, data))



if __name__ == '__main__':
    configs = get_dynamic_configs()

    displayed = set()
    # p, smpl, o, s = configs[0]
    # render_result(ENV, p, smpl, pose_id=s)

    for config in configs:
        p, smpl, o, s = config
        if (p, smpl) not in displayed:
            displayed.add((p, s))
        render_result(ENV, p, smpl, pose_id=s)

# Note for me: in env.human.get_link_positions(), try loading in the joints in the order prespecific in error_estimation.py
