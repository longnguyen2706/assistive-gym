import argparse
import os
import shutil

import pybullet as p
import pybullet_data

from assistive_gym.envs.utils.urdf_utils import generate_human_mesh
from urdf_name_resolver import get_urdf_folderpath, get_urdf_filepath, get_urdf_ref_filepath

GENDER_DATAPATH = 'examples/data/slp3d/gender.txt'


def read_gender_data(filepath):  # currently works with slp3d
    gender_data = {}
    with open(filepath, "r") as f:
        text = f.read()
        lines = text.strip().split("\n")

        for line in lines:
            sample, gender = line.split(": ")
            gender_data[int(sample)] = gender

        return gender_data


def generate_urdf(args):
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    urdf_folder = get_urdf_folderpath(args.person_id)
    os.makedirs(urdf_folder, exist_ok=True)

    # copy the urdf ref file
    urdf_ref_file = get_urdf_ref_filepath(urdf_folder)
    shutil.copy(args.ref_urdf_file, urdf_ref_file)

    gender_data = read_gender_data(os.path.join(os.getcwd(), GENDER_DATAPATH))
    gender = gender_data[int(args.person_id[1:])]

    generate_human_mesh(physic_client_id, gender, urdf_ref_file, urdf_folder, args.smpl_file)
    print(
        f"Generated urdf file from ref urdf: {args.ref_urdf_file} with smpl file: {args.smpl_file}, out folder: {urdf_folder}")
    # remove the ref file from the folder
    os.remove(urdf_ref_file)
    # Disconnect from the simulation
    p.disconnect()


# generate urdfs for all the people in the slp3d dataset
# TODO: refactor or remove
def generate_urdfs():
    for i in range(1, 103):  # TODO: fix why p046 has no gender
        args.person_id = f'p{i:03d}'
        args.smpl_file = f'examples/data/slp3d/{args.person_id}/s01.pkl'
        try:
            generate_urdf(args)
        except Exception as e:
            print(f'Error generating urdf for {args.person_id}: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Util script for human urdf generation')
    parser.add_argument('--ref-urdf-file', default='assistive_gym/envs/assets/human/ref_mesh.urdf',
                        help='Path to reference urdf file')
    parser.add_argument('--person-id', default='p001', help='Person id')
    parser.add_argument('--smpl-file', default='examples/data/slp3d/p001/s01.pkl',
                        help='Path to smpl file')
    parser.add_argument('--mode', default='generate', help='Mode: generate or test')
    args = parser.parse_args()
    if args.mode == 'generate':
        # generate_urdf(args)
        generate_urdfs()
    else:
        raise NotImplementedError()
