import pybullet as p
import argparse


if __name__ == '__main__':
    p.connect(p.DIRECT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='+', required=True)
    args = parser.parse_args()
    name_in = args.file[0]
    name_out = "new_vhacd.obj"
    name_log = "log.txt"
    p.vhacd(name_in, name_out, name_log)