import time
import pickle as pkl
import numpy as np
import torch
from assistive_gym.mprocess_train import mp_train
from assistive_gym.train import train

#### Define dynamic configs ####
### testing config ###
# PERSON_IDS = ['p001', 'p002']
# SMPL_FILES = ['s01', 's08', 's13', 's18', 's19', 's20', 's24', 's30', 's36', 's40', 's44']
PERSON_IDS = ['p000']
SMPL_FILES = ['s01']
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
#               's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26',
#               's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',]
# # PERSON_IDS = [ 'p002' ]
OBJECTS = ['pill']

# PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p009', 'p010']
# # PERSON_IDS = ['p004']
# # SMPL_FILES = ['s01' ]
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
#               's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26',
#               's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
#               's41', 's42', 's43', 's44', 's45']
# # SMPL_FILES = [ 's01', 's19', 's45']
# # SMPL_FILES = [ 's44']
# # PERSON_IDS = [ 'p001', 'p002']
# # SMPL_FILES = [ 's19', 's20', 's44', 's45']
# OBJECTS = ['cane', 'cup', 'pill']

#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'HumanComfort-v1'
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
    mp_train(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
    return "Done training for {} {} {}".format(p, s, o)


if __name__ == '__main__':
    configs = get_dynamic_configs()
    # OPEN and PRINT: compare new pkl to regular
    # smpl_file_test = pkl.load(open('examples/data/slp3d/p001/s00.pkl', 'rb'))
    # smpl_file_norm = pkl.load(open('examples/data/slp3d/p001/s01.pkl', 'rb'))
    # print("smpl_file_norm: ", smpl_file_norm['global_orient'])
    # print("smpl_file_test: ", smpl_file_test['global_orient'])
    # print("\n\ns00 smpl body_pose: ", smpl_file_test['body_pose'], "\njoint #: ", len(smpl_file_test['body_pose'][0])/3)
    
    # # APPEND: add [0, 0, 0] for pelvis, left hand, right_hand
    # body_pose = np.append(np.array([0, 0, 0]), smpl_file_test['body_pose'][0])
    # body_pose = np.append(body_pose, np.array([0, 0, 0, 0, 0, 0]))
    # print("body_pose edited: ", body_pose)
    # print("joint #: ", len(smpl_file_test['body_pose'][0])/3)
    
    # # WRITE: append and save edited body pose to pkl file
    # smpl_file_test['body_pose'] = torch.from_numpy(body_pose)
    # smpl_file_test['gloabl_orient'] = smpl_file_test['global_orient'][0]
    # smpl_file_test['transl'] = smpl_file_test['transl'][0]
    # # print("edited: ", smpl_file_test['gloabl_orient'])
    # pkl.dump(smpl_file_test, open("examples/data/slp3d/p001/s0.pkl", 'wb'))
    # input("continue?")
    # ORIGINAL: leave unchanged
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(do_train, configs)
    for num, result in enumerate(results):
        print('Done training for {} {}'.format(num, result))
