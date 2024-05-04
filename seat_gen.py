import os
import pickle as pkl
import numpy as np
import pybullet as p
import torch

import experimental.generate_human_urdf

from scripts.rewrite_pose_pkl import rewrite
from assistive_gym.mprocess_train import mp_train, mp_load
from assistive_gym.train import train

#### Define dynamic configs ####
### testing config ###
PERSON_IDS = ['p100', 'p102']
SMPL_FILES = ['s01']
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
#               's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26',
#               's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',]
# # PERSON_IDS = [ 'p002' ]
OBJECTS = ['pill']



#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'SeatedPose-v1'
SEED = 1001
SAVE_DIR = 'trained_models'
RENDER_GUI = False
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 1

if __name__ == '__main__':
    file_list = open("fp") # WRITE later
    # ensure files include: file path and person id

    for file in file_list:
        fp, pid, sid = file.split(".")
        # rewrite raw file and generate urdf
        rewrite(fp, save_original=True)
        mp_load(ENV, SEED, fp, pid, render=RENDER_GUI)
        # THIS should be it - load external and then the saving happens in the environment, or can be and external function


    

    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     results = executor.map(do_train, configs)
    # for num, result in enumerate(results):
    #     print('Done training for {} {}'.format(num, result))
