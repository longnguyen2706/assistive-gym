import os

BASEDIR = os.path.dirname(os.path.realpath(__file__))

def get_urdf_folder(dir, person_id):
    return os.path.join(BASEDIR, dir, person_id)

def get_urdf_file(urdf_folder):
    return os.path.join(urdf_folder, 'human.urdf')

def get_urdf_mesh_folder(urdf_folder):
    return os.path.join(urdf_folder, 'meshes')
