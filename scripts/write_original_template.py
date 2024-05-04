import time
import os
import numpy as np
import argparse
import pickle as pk

def rewrite(file, folder, save_original=None):
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
            f['betas'] = f['betas'][0]
            f['global_orient'] = f['global_orient'][0]
            f['body_pose'] = np.append(f['body_pose'][0], [0, 0, 0, 0, 0, 0])
            f['transl'] = f['transl'][0]
            
            # SAVE: save reconfigured data
            # input("continue to save?")
            pk.dump(f, open(file, 'wb'))

def clear_original(file):
    f = open(file, 'wb')
    data = None
    pk.dump(data, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    # clear_original(args.file)
    check = pk.load(open(args.file, 'rb'))
    print("c: ", check)
