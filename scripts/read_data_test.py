import time
import os
import numpy as np
import argparse
import pickle as pk

def clean_poses(folder):
    d = [13, 14, 20, 37, 38, 52, 53, 54, 59, 75, 76, 77, 85, 88, 
             90, 91, 92, 94, 95, 130, 131, 137, 138, 149, 150, 152, 
             155, 156, 157]
    
    for pose in d:
        fp = folder + "/"
        if pose < 100: fp += "0"
        fp += str(pose)
        depth_fp = fp + "_depth.pkl"
        png_fp = fp + ".png"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    f = open(args.file, 'rb')
    data = pk.load(f)
    f.close()
    i = 0
    for key in data:
        if i > 10: break
        print(key, ": ", data[key])
        i += 1