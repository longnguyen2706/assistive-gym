import argparse

import pybullet as p
import pybullet_data

from assistive_gym.envs.utils.urdf_utils import generate_human_mesh
from urdf_name_resolver import get_urdf_folder, get_urdf_file

def generate_urdf(args):
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    urdf_folder = get_urdf_folder('', args.person_id)

    generate_human_mesh(physic_client_id, args.ref_urdf_file, urdf_folder, args.smpl_file)
    print(f"Generated urdf file from ref urdf: { args.ref_urdf_file} with smpl file: {args.smpl_file}, out folder: {urdf_folder}")

    # Disconnect from the simulation
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Util script for human urdf generation')
    parser.add_argument('--ref-urdf-file', default='assistive_gym/envs/assets/human/ref_mesh.urdf',  help='Path to reference urdf file')
    parser.add_argument('--person-id', default='p001', help='Person id')
    parser.add_argument('--smpl-file', default='examples/data/fits/p005/s02.pkl',
                        help='Path to smpl file')
    parser.add_argument('--mode', default='generate', help='Mode: generate or test')
    args = parser.parse_args()
    if args.mode == 'generate':
        generate_urdf(args)
    else:
        raise NotImplementedError()

