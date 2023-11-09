import os
import pickle as pkl
import numpy as np
import pybullet as p
import torch
from assistive_gym.mprocess_train import mp_train, mp_load
from assistive_gym.train import train

#### Define dynamic configs ####
### testing config ###
PERSON_IDS = ['p000']
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

def get_dynamic_configs():
    configs =[]
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs

import concurrent.futures

def do_train(config):
    p, s, o = config
    print (p, s, o)
    mp_load(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
    return "Done training for {} {} {}".format(p, s, o)


if __name__ == '__main__':
    configs = get_dynamic_configs()
    # ORIGINAL: leave unchanged
    # if ENV == 'SeatedPose-v1':
    do_train(configs[0])
    

    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     results = executor.map(do_train, configs)
    # for num, result in enumerate(results):
    #     print('Done training for {} {}'.format(num, result))