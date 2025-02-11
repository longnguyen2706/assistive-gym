import json
import time

from assistive_gym.train import train
import concurrent.futures
import os

class FileFilter:
    def __init__(self, pref, lower, upper):
        self.pref = pref
        self.lower = lower
        self.upper = upper

    def is_valid(self, item):
        if item.startswith(self.pref):
            idx = int(item.split('_')[-1])
            return self.lower <= idx <= self.upper
        return False

#### Define path configs ####
SYNTHETIC_DATA= '/home/louis/Documents/hrl/synthetic_dataset/'
SMPL_DIR = os.path.join(SYNTHETIC_DATA, 'smpl/smpl_data')
URDF_DIR = os.path.join(SYNTHETIC_DATA, 'urdf')

#### Define static configs ####
OBJECTS = ['cane']
FILES_FILTERS = [FileFilter('f_2', 0, 1999), FileFilter('m_11', 0, 1999)] 

ENV = 'HumanComfort-v1_augmented_cane'
SEED = 1001
SAVE_DIR = 'trained_models'
RENDER_GUI = False
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

exception_file = os.path.join(SAVE_DIR, ENV, 'exception.txt')

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS =24

def get_aug_files():
    files = []
    dirs = os.listdir(URDF_DIR)
    print(len(dirs))
    for d in dirs:
        for filter in FILES_FILTERS:
            if filter.is_valid(d):
                files.append(d)
    # print (len(files), files)
    return files

# def clean():
#     dirs = os.listdir(URDF_DIR)
#     for d in dirs:
#         if not d.startswith('f') or not d.startswith('m'):
#             os.system('rm -rf ' + os.path.join(URDF_DIR, d))

def get_dynamic_configs(rerun_missing=False):
    # invalid_cases = get_invalid_cases()
    configs =[]
    filenames = []
    filenames =  get_missing_files() if rerun_missing else get_aug_files()
   
    for f in filenames:
        smpl_file = os.path.join(SMPL_DIR,  f + '.pkl')
        for o in OBJECTS:
            configs.append((f, smpl_file, o))
    configs.sort()
    print (len(configs), configs)
    return configs


# def get_invalid_cases():
#     # read the invalid cases from the json file
#     cases = json.loads(open('invalid_cases.json').read())
#     invalid_cases = set()
#     for case in cases:
#         invalid_cases.add(tuple(case))

#     return invalid_cases

def get_missing_files():
    result_dir = os.path.join(SAVE_DIR, ENV)
    result_files = os.listdir(result_dir)
    miss_count = 0
    processed_count =0
    missed_files = []
    for f in get_aug_files():
        if f not in result_files:
            # print (f)
            miss_count +=1
            missed_files.append(f)
        else:
            processed_count +=1
    print ("processed: ", processed_count, "missed: ", miss_count)
    return missed_files

def do_train(config):
    p, s, o = config
    print (p, s, o)
    try:
        train(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o, is_augmented=True)
        # mp_read(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
        return "Done training for {} {} {}".format(p, s, o)
    except Exception as e: 
        message= "Exception for {} {} {}, cause {} \n".format(p, s, o, e)
        with open(exception_file, 'a') as f: 
            f.write(message)
        f.close()

if __name__ == '__main__':
    counter = 0

    configs = get_dynamic_configs(rerun_missing=False)
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(do_train, config): (i, config) for i, config in enumerate(configs)
        }
        for future in concurrent.futures.as_completed(futures):
            res = futures[future]
            counter +=1
            try:
                print('Done training for {}'.format(res), 'progress: {} / {}'.format(counter, len(configs)) )
                del futures[future]
            except Exception as exc:
                print('%r generated an exception: %s' % (res, exc))
    executor.shutdown()
    end = time.time()
    print("Total time taken: {}".format(end - start))
