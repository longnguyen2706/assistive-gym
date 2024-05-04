import argparse
import pybullet as p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Util script for furniture file parsing')
    parser.add_argument('--obj-file')
    args = parser.parse_args()
    p.connect(p.DIRECT)
    name_in = args.obj_file + ".obj"
    name_out =  args.obj_file + "_vhacd.obj"
    name_log = "log.txt"
    p.vhacd(name_in, name_out, name_log)

# Note for me: in env.human.get_link_positions(), try loading in the joints in the order prespecific in error_estimation.py
