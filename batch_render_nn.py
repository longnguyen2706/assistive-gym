import concurrent
import json
import os
import time

from assistive_gym.envs.utils.train_utils import get_save_dir, render_nn_result

#### Define dynamic configs ####
PERSON_IDS = ['p001']
SMPL_FILES = ['s13','s14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26']
#
# PERSON_IDS = ['p047']
# SMPL_FILES = ['s15']

OBJECTS = ['pill']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1"
# SAVE_DIR = 'trained_models'
NN_OUTPUT_DIR = 'deepnn/data/output'
SOURCE_DIR = 'deepnn/data/input/searchoutput'

def get_dynamic_configs():
    configs = []
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)

                configs.append((p, smpl_file, o))
    return configs

def get_folders(config):
    p, s, o = config
    nn_output_dir = get_save_dir(NN_OUTPUT_DIR, "", p, s)
    original_output_dir = get_save_dir(SOURCE_DIR, "", p, s)
    return nn_output_dir, original_output_dir

def do_render_nn(config):
    p, s, o, is_real = config
    print(p, s, o)
    if is_real:
        data_dir = get_save_dir(SOURCE_DIR, "", p, s)
    else:
        data_dir = get_save_dir(NN_OUTPUT_DIR, "", p, s)
    # load data from json file. filename = object name, file dir = save_dir/p/s
    data = json.load(open(os.path.join(data_dir, o + ".json"), "r"))
    data['end_effector'] ='right_hand'

    render_nn_result(ENV, data, p, s, o, is_real)


if __name__ == '__main__':
    configs = get_dynamic_configs()
    for config in configs:
        nn_output_dir, original_output_dir = get_folders(config)
        settings = [True, False]
        arr = [[config[0], config[1], config[2], settings[0]], [config[0], config[1], config[2], settings[1]]]
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(do_render_nn, config): (i, config) for i, config in enumerate(arr)
            }
            for future in concurrent.futures.as_completed(futures):
                res = futures[future]
                try:
                    print('Done rendering for {}'.format(res))
                    del futures[future]
                except Exception as exc:
                    print('%r generated an exception: %s' % (res, exc))

        executor.shutdown(wait=True)

