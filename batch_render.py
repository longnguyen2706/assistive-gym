import time

from assistive_gym.train import render

#### Define dynamic configs ####
# PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005']
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05']
PERSON_IDS = ['p001', 'p002']
SMPL_FILES = ['s01', 's02']
OBJECTS = ['cane', 'cup', 'pill']

#### Define static configs ####
SMPL_DIR = 'examples/data/fits/'
ENV = 'HumanComfort-v1'
SEED = 1001
SAVE_DIR = 'trained_models'
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 20

def get_dynamic_configs():
    configs =[]
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs

def do_render(config):
    p, s, o = config
    print (p, s, o)
    render(ENV, p, s, SAVE_DIR,o, ROBOT_IK)

if __name__ == '__main__':
    configs = get_dynamic_configs()

    for c in configs:
        do_render(c)
