import concurrent
import json
import time

from assistive_gym.train import render, render_pose

#### Define dynamic configs ####
"""
p013', 'p014', 'p015', 'p016', 'p017', 'p018', 'p019', 'p020', 'p021', 'p022', 'p023', 'p024',
              'p025', 'p026', 'p027', 'p028', 'p029', 'p030', 'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037',
              'p038', 'p039', 'p040', 'p041', 'p042', 'p043', 'p044', 'p045', 'p046', 'p047', 'p048', 'p049', 'p050',
              'p051', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p058', 'p059', 'p060', 'p061', 'p062',
              'p063', 'p064', 'p065', 'p066', 'p067', 'p068', 'p069','p070', 'p071', 'p072', 'p073', 'p074', 'p075', 'p076',
              'p077', 'p078', 'p079', 'p080', 'p081', 'p082', 'p083', 'p084', 'p085', 'p086', 'p087', 'p088', 'p089',
              'p090', 'p091', 'p092', 'p093', 'p094', 'p095', 'p096', 'p097', 'p098', 'p099', 'p100', 'p101', 'p102'
"""
PERSON_IDS = ['p050','p051', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p058',
              'p059', 'p060', 'p061', 'p062', 'p063', 'p064', 'p065', 'p066', 'p067', 'p068', 'p069', 'p070', 'p071',
              'p072', 'p073', 'p074', 'p075', 'p076', 'p077', 'p078', 'p079', 'p080', 'p081', 'p082', 'p083', 'p084',
              'p085', 'p086', 'p087', 'p088', 'p089', 'p090', 'p091', 'p092', 'p093', 'p094', 'p095', 'p096', 'p097',
              'p098', 'p099', 'p100', 'p101', 'p102']
#
SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
              's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25',
              's26', 's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38',
              's39', 's40', 's41', 's42', 's43', 's44', 's45']
# SMPL_FILES = ['s02']

OBJECTS = ['cane']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1_rerun_dec15_cane" #"HumanComfort-v1_rerun_dec15_cup"
SEED = 1001
# SAVE_DIR = 'trained_models'
SAVE_DIR = '/home/louis/Documents/hrl/results'
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 32


def get_dynamic_configs():
    configs = []
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs


def do_render(config):
    p, s, o = config
    print(p, s, o)
    render_pose(ENV, p, s)
    # render(ENV, p, s, SAVE_DIR, o, ROBOT_IK, save_to_file=False, save_metrics=False)


def get_invalid_cases(filepath):
    configs = []
    with open(filepath, 'r') as f:
        cases = json.load(f)
        for case in cases:
            p, s, o = case
            smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
            configs.append((p, smpl_file, o))
    return configs


if __name__ == '__main__':
    configs = get_dynamic_configs()
    # configs = get_invalid_cases('invalid_ik_cases_cup.json')
    displayed = set()
    # for config in configs:
    #     p, s, o = config
    #     if (p, s) not in displayed:
    #         displayed.add((p, s))
    #         # render_pose(ENV, p, s)
    #     do_render(config)

    # TODO: delete this. we need this one to regenerate weakly label data from actions.pkl
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(do_render, config): (i, config) for i, config in enumerate(configs)
        }
        # futures = executor.map(do_render, configs)
        for future in concurrent.futures.as_completed(futures):
            res = futures[future]
            try:
                print('Done rendering for {}'.format(res))
                del futures[future]
            except Exception as exc:
                print('%r generated an exception: %s' % (res, exc))
    executor.shutdown()
