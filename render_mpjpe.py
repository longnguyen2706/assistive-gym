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
from scripts.icp import icp 

    

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
    
    # SETUP
    print("person_id: ", person_id, "smpl_file: ", smpl_file)
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=None, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    # BUILD: collect joint angles from sim to compare to reference file
    humanURDF = HumanUrdfDict()
    joints = ["left_hip", "pelvis", "right_hip", "left_shoulder", # 0 - 3
        "neck", "right_shoulder", "left_elbow", "right_elbow", # 4 - 7
        "left_hand", "right_hand", "left_knee", "right_knee", # 8 - 11
        "left_ankle", "right_ankle", "spine_2", "spine_3", "spine_4", # 12 - 16
        "left_foot", "right_foot", "head", "left_clavicle", "right_clavicle",  # 17 - 21
        "left_lowarm", "right_lowarm" # 22 - 23
        ]
    data = []
    for joint in joints:
        if joint == "pelvis": data.append(np.array(env.human.get_link_positions_id(0, center_of_mass=False)))
        else:
            # print("joint: ", joint, " -> dammy joint id: ", humanURDF.get_dammy_joint_id(joint))
            data.append(np.array(env.human.get_link_positions_id(humanURDF.get_dammy_joint_id(joint), center_of_mass=False)))
            # print(joint, ": ", env.human.get_link_positions_id(humanURDF.get_dammy_joint_id(joint)))

    env.disconnect()
    
    # OPEN: SMPL body
    slp_body = pkl.load(open("error/" + person_id + "/" + pose_id + ".pkl", "rb"))
    print("slp body: ", slp_body)
    smpl_path = "examples/data/SMPL_FEMALE.pkl"
    smpl_data = SMPLData(slp_body['body_pose'], slp_body['betas'], slp_body['global_orient'])
    
    # BUILD: joint locations from SMPL body
    _, joint_pos, _ = generate_geom2(smpl_path, smpl_data, "assistive_gym/error/error_geom")
    s_body = []
    joints = ["L_Hip", "Pelvis", "R_Hip", "L_Shoulder", # 0 - 3
        "Neck", "R_Shoulder", "L_Elbow", "R_Elbow", # 4 - 7
        "L_Hand", "R_Hand", "L_Knee", "R_Knee", # 8 - 11
        "L_Ankle", "R_Ankle", "Spine1", "Spine2", "Spine3", # 12 - 16
        "L_Foot", "R_Foot", "Head", "L_Collar", "R_Collar", # 17 - 21
        "L_Wrist", "R_Wrist" # 22 - 23
    ]
    for joint in joints:
        s_body.append(joint_pos[joint])
    
    # ICP: to align AG body with SMPL
    smpl = np.array(s_body) - s_body[1] # put pelvis at the origin
    ag = np.reshape(np.array(data), (24,3))
    ag = ag - ag[1]
    rot, dists, _ = icp(ag, smpl, max_iterations=45)
    
    # ROTATE: assitive gym body to match smpl
    ag_trans = np.ones((24, 4))
    ag_trans[:,  0:3] = np.copy(ag)

    ag_trans = np.dot(rot, ag_trans.T).T
    ag_trans = ag_trans[:, :-1]

    # for i,row in enumerate(ag):
    #     # print("pre transform: ", row)
    #     ag_trans[i] = row[0:-1] / row[-1]
    #     print("ag: ", ag_trans[i], "-> smpl: ", smpl[i])
    

    # SHOW
    print()
    build_map_pkl([], 0, 0, 0, smpl_ag_body_pts=smpl, ag_body_pts=ag_trans, go=False) # not building a heatmap, just body plotting

    # WRITE
    file_w = open("error/mpjpe_values.txt", "a")
    file_w.write(person_id + "/" + pose_id + ".pkl: " + str(mpjpe(smpl, ag_trans)))
    file_w.close()
    
    # PRINT
    print("\n\nmpjpe: ", mpjpe(smpl, ag_trans))
    print("final distances: ", dists)



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
