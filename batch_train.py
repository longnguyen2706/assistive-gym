import time

from assistive_gym.mprocess_train import mp_train
from assistive_gym.train import train

#### Define dynamic configs ####
PERSON_IDS = ['p001', 'p002']
# SMPL_FILES = ['s19', 's05']
# PERSON_IDS = ['p004']
SMPL_FILES = ['s01', 's10', 's12', 's16', 's17', 's18', 's19', 's20', 's21', 's22',
              's30', 's31', 's32', 's36', 's37', 's38', 's39', 's40', 's44', 's45']
# PERSON_IDS = [ 'p002' ]
OBJECTS = ['pill']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'HumanComfort-v1_2510'
SEED = 1001
SAVE_DIR = 'trained_models'
RENDER_GUI = True
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 18

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

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(do_train, configs)
    for num, result in enumerate(results):
        print('Done training for {} {}'.format(num, result))
