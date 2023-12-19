import gc
import json
import os
import time

from assistive_gym.train import train
import concurrent.futures

#### Define dynamic configs ####
PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p009', 'p010', 'p011', 'p012',
              'p013', 'p014', 'p015', 'p016', 'p017', 'p018', 'p019', 'p020', 'p021', 'p022', 'p023', 'p024', 'p025',
              'p026', 'p027', 'p028', 'p029', 'p030', 'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037', 'p038',
              'p039', 'p040', 'p041', 'p042', 'p043', 'p044', 'p045',
              'p046', 'p047', 'p048', 'p049', 'p050', 'p051', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p058',
              'p059', 'p060', 'p061', 'p062', 'p063', 'p064', 'p065', 'p066', 'p067', 'p068', 'p069', 'p070', 'p071',
              'p072', 'p073', 'p074', 'p075', 'p076', 'p077', 'p078', 'p079', 'p080', 'p081', 'p082', 'p083', 'p084',
              'p085', 'p086', 'p087', 'p088', 'p089', 'p090', 'p091', 'p092', 'p093', 'p094', 'p095', 'p096', 'p097',
              'p098', 'p099', 'p100', 'p101', 'p102']

SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
              's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25',
              's26', 's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38',
              's39', 's40', 's41', 's42', 's43', 's44', 's45']

OBJECTS = ['cane']

#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'HumanComfort-v1_rerun_dec19_cane'
SEED = 1001
SAVE_DIR = 'trained_models'
RENDER_GUI = False
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

exception_file = os.path.join(SAVE_DIR, ENV, 'exception.txt') # dumps all exception message to this file

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 1


def get_dynamic_configs(re_run_failed_cases=False):
    # invalid_cases = get_invalid_cases()
    configs = []
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                if os.path.exists(os.path.join(SAVE_DIR, ENV, p, s, o, ".json")):
                    print("Already trained for {} {} {}".format(p, s, o))
                    continue
                # if re_run_failed_cases and (p, s, o) not in invalid_cases:
                #     continue
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    # print(len(configs), configs)
    return configs


def get_invalid_cases():
    # read the invalid cases from the json file
    cases = json.loads(open('invalid_ik_cases_cup.json').read())
    invalid_cases = set()

    for case in cases:
        p, s, o = case
        # if os.path.join(SAVE_DIR, ENV, p, s, o, ".json"):
        #     print("Already trained for {} {} {}".format(p, s, o))
        #     continue
        # if p == 'p003' and s == 's13':
        smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
        invalid_cases.add((p, smpl_file, o))
    invalid_cases = list(invalid_cases)
    invalid_cases.sort()
    return invalid_cases


def do_train(config):
    p, s, o = config
    print(p, s, o)
    try:
        train(ENV, SEED, s, p, END_EFFECTOR, SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
        # mp_read(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
        return "Done training for {} {} {}".format(p, s, o)
    except Exception as e:
        message = "Exception for {} {} {}, cause {} \n".format(p, s, o, e)
        with open(exception_file, 'a') as f:
            f.write(message)
        f.close()


if __name__ == '__main__':
    counter = 0

    configs = get_dynamic_configs(re_run_failed_cases=False)
    # configs = get_invalid_cases()
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(do_train, config): (i, config) for i, config in enumerate(configs)
        }
        for future in concurrent.futures.as_completed(futures):
            res = futures[future]
            counter += 1
            try:
                print('Done training for {}'.format(res), 'progress: {} / {}'.format(counter, len(configs)))
                del futures[future]
                gc.collect()  # just to be sure gc is called
            except Exception as exc:
                print('%r generated an exception: %s' % (res, exc))
    executor.shutdown()
    end = time.time()
    print("Total time taken: {}".format(end - start))
