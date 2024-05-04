import time
import os
import numpy as np
import argparse
import pickle as pk

def rewrite(file, folder, save_original=None, betas=True):
    # PULL: information
    if folder is None:
        o = open(file, 'rb')
        f = pk.load(o)
        o.close()
        
        # SAVE ORIGINAL DATA
        if save_original:
            # input("continue to original save?")
            fn = file.split(".")[0] + "_original.pkl"
            pk.dump(f, open(str(fn), 'wb'))
            print("original saved")
        else: print("WARNING: original is not being saved")
        # CONFIGURE: data for new file
        f['betas'] = f['betas'][0]
        f['global_orient'] = f['global_orient'][0]
        f['body_pose'] = np.append(f['body_pose'][0], [0, 0, 0, 0, 0, 0])
        print("type of body_pose: ", type(f['body_pose']))
        f['transl'] = f['transl'][0]
        print("f: ", f)
        
        # SAVE: save reconfigured data
        # input("continue to save?")
        pk.dump(f, open(file, 'wb'))
    elif not betas:
        file_list = os.listdir(folder)
        for file in file_list:
            file = folder + "/" + file
            o = open(file, 'rb')
            f = pk.load(o)
            o.close()
            
            # SAVE ORIGINAL DATA
            if save_original:
                # input("continue to original save?")
                fn = file.split(".")[0] + "_original.pkl"
                pk.dump(f, open(str(fn), 'wb'))
                print("original saved")
            else: print("WARNING: original is not being saved")
            # CONFIGURE: data for new file
            f['betas'] = f['betas'][0]
            f['global_orient'] = f['global_orient'][0]
            f['body_pose'] = np.append(f['body_pose'][0], [0, 0, 0, 0, 0, 0])
            f['transl'] = f['transl'][0]
            
            # SAVE: save reconfigured data
            # input("continue to save?")
            pk.dump(f, open(file, 'wb'))
    else:
        file_list = os.listdir(folder)
        for file in file_list:
            file = folder + "/" + file
            o = open(file, 'rb')
            f = pk.load(o)
            o.close()
            
            # SAVE ORIGINAL DATA
            if save_original:
                # input("continue to original save?")
                fn = file.split(".")[0] + "_original.pkl"
                pk.dump(f, open(str(fn), 'wb'))
                print("original saved")
            else: print("WARNING: original is not being saved")
            # CONFIGURE: data for new file
            print("f[betas]: ", f['betas'])
            f['betas'] = f['betas'][0]
            print("changed: ", f['betas'])
            # SAVE: save reconfigured data
            # input("continue to save?")
            pk.dump(f, open(file, 'wb'))


def open_amass(folder):
    template = open("examples/data/slp3d/p100/s001.pkl", 'rb')
    t = pk.load(template)

    print("t: ", t)
    print("t[betas]: ", t['betas'])
    
    folders = os.listdir(folder)
    betas = {}
    for f in folders:
        fp = folder + "/" + f + "/betas.npy"
        betas[f] = np.load(fp)
        print("\n", f, "\n", betas[f])
    
    p_s = ['p120', 'p121', 'p122', 'p123', 'p124', 'p125', 'p126', 'p127', 'p128']
    i = 0

    for key in betas:
        temp = t
        temp['betas'] = betas[key]
        print("new b: ", temp['betas'])
        input("continue?")
        file = 'examples/data/slp3d/p000_originals/' + p_s[i] + '.pkl'
        with open(file, 'wb') as f: pk.dump(temp, f)
        i += 1

def write_beta_files(folder):
    files = os.listdir(folder)
    template = open("examples/data/slp3d/p100/s001.pkl", 'rb')
    t = pk.load(template)

    for f in files:
        fp = folder + "/" + f
        beta_file = open(fp, 'rb')
        betas = pk.load(beta_file)
        beta_file.close()

        temp = t
        print("original betas: ", temp['betas'])
        temp['betas'] = betas['betas']
        print("new betas: ", temp['betas'])
        input("continue?")
        with open(fp, 'wb') as f: pk.dump(temp, f) # OVERWRITE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--folder')
    parser.add_argument('--wo', action='store_true')
    args = parser.parse_args()
    # dp = rewrite(args.file, args.folder, args.wo)
    write_beta_files('examples/data/slp3d/people_files')