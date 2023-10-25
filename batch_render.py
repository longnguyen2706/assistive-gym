import time

from assistive_gym.train import render, render_pose

#### Define dynamic configs ####

# SMPL_FILES = ['s19']
# PERSON_IDS = ['p001']
# OBJECTS = ['pill']
# PERSON_IDS = [ 'p001']

PERSON_IDS = ['p001', 'p002']
SMPL_FILES = ['s01', 's08', 's13', 's18', 's19', 's20', 's24', 's30', 's36', 's40', 's44']

# SMPL_FILES = [ 's44']
# PERSON_IDS = [ 'p001', 'p002']
# SMPL_FILES = [ 's19', 's20', 's44', 's45']
OBJECTS = ['cane', 'cup', 'pill']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1_param_1509_yaml"
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
                configs.append((p, smpl_file, o))
    return configs

def do_render(config):
    p, s, o = config
    print (p, s, o)
    print("save dir: ", SAVE_DIR)
    render(ENV, p, s, SAVE_DIR,o, ROBOT_IK)

if __name__ == '__main__':
    configs = get_dynamic_configs()

    displayed = set()
    for config in configs:
        p, s, o = config
        if (p, s) not in displayed:
            displayed.add((p, s))
            do_render(config)
            # render_pose(ENV, p, s)
        # do_render(config)
