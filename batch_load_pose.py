import os
import argparse
import pickle as pkl
import numpy as np
import pybullet as p
import torch
from assistive_gym.mprocess_train import mp_train, mp_load
from assistive_gym.train import train

SMPL_FILES = []
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

def set_smpl_files(min):
    print("min: ", int(min[0]))
    for i in range(155):
        if i < int(min[0]) or i in [13, 14, 20, 37, 38, 44, 52, 53, 54, 59, # remove faulty poses
             75, 76, 77, 85, 88, 90, 91, 92, 94, 95, 114, 
             130, 131, 137, 138, 149, 150, 152]: continue
        text = 's'
        if i < 10: text += '00' + str(i)
        elif i < 100: text += '0' + str(i)
        else: text += str(i)
        SMPL_FILES.append(text)
    print("SMPL_FILES: ", SMPL_FILES)
    input("continue?")

def get_dynamic_configs(person_ids):
    configs = [] 
    for p in person_ids:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs


def do_train(config):
    p, s, o = config
    print (p, s, o)
    mp_load(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
    return # "Done training for {} {} {}".format(p, s, o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--startPose', nargs='+', required=True)
    args = parser.parse_args()
    set_smpl_files(args.startPose)
    configs = get_dynamic_configs(args.ids)
    # configs = get_dynamic_configs(PERSON_IDS)
    for c in configs:
        print("c: ", c)
        do_train(c)
    
