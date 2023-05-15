import pickle

import numpy as np
import pybullet as p
import torch
from pytorch3d import transforms as t3d

from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.urdf_editor import UrdfEditor

def load_smpl(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)

    # print("data keys", data.keys())
    print(
        "body_pose: ",
        data["body_pose"].shape,
        "betas: ",
        data["betas"].shape,
        "global_orient: ",
        data["global_orient"].shape,
    )

    return data


# TODO: move to utils
def convert_aa_to_euler(aa):
    aa = np.array(aa)
    mat = t3d.axis_angle_to_matrix(torch.from_numpy(aa))
    # print ("mat", mat)
    quats = t3d.matrix_to_quaternion(mat)
    euler = t3d.matrix_to_euler_angles(mat, "XYZ")
    return euler


# TODO: move to utils
def euler_convert_np(q, from_seq='XYZ', to_seq='XYZ'):
    r"""
    Convert euler angles into different axis orders. (numpy, single/batch)

    :param q: An ndarray of euler angles (radians) in from_seq order. Shape [3] or [N, 3].
    :param from_seq: The source(input) axis order. See scipy for details.
    :param to_seq: The target(output) axis order. See scipy for details.
    :return: An ndarray with the same size but in to_seq order.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(from_seq, q).as_euler(to_seq)


def mul_tuple(t, multiplier):
    return tuple(multiplier * elem for elem in t)

def scale_body_part(human_id, physic_client_id, scale=10):
    editor = UrdfEditor()
    editor.initializeFromBulletBody(human_id, physic_client_id) # load all properties to editor
    # scaling the robot
    for link in editor.urdfLinks:
        for v in link.urdf_visual_shapes:
            if v.geom_type ==  p.GEOM_BOX:
                v.geom_extents = mul_tuple(v.geom_extents, 10)
            if v.geom_type ==  p.GEOM_SPHERE:
                v.geom_radius *= 10
            if v.geom_type ==  p.GEOM_CAPSULE:
                v.geom_radius *= 10
                v.geom_length *= 10
            v.origin_xyz= mul_tuple(v.origin_xyz, 10)

        for c in link.urdf_collision_shapes:
            if c.geom_type == p.GEOM_BOX:
                c.geom_extents = mul_tuple(c.geom_extents, 10)
            if c.geom_type == p.GEOM_SPHERE:
                c.geom_radius *= 10
            if c.geom_type == p.GEOM_CAPSULE:
                c.geom_radius *= 10
                c.geom_length *= 10
            c.origin_xyz = mul_tuple(c.origin_xyz, 10)
    for j in editor.urdfJoints:
        j.joint_origin_xyz = mul_tuple(j.joint_origin_xyz, 10)
    editor.saveUrdf("test10.urdf", True)

def reposition_body_part(human_id, physic_client_id, pos_dict):
    editor = UrdfEditor()
    editor.initializeFromBulletBody(human_id, physic_client_id) # load all properties to editor
    human_dict = HumanPipDict()

    # scaling the robot
    # for link in editor.urdfLinks:
    #     # remove _limb from left_arm_limb
    #     last_underscore_idx = link.link_name.rfind("_")
    #     name = link.link_name[:last_underscore_idx]
    #     print ("link_name", link.link_name, name)
    #     # TODO: update inertia
    #     if name in human_dict.urdf_to_smpl_dict.keys():
    #         urdf_name = human_dict.urdf_to_smpl_dict[name]
    #         for v in link.urdf_visual_shapes:
    #             v.origin_xyz = pos_dict[urdf_name]
    #         for c in link.urdf_collision_shapes:
    #             c.origin_xyz = pos_dict[urdf_name]

    for link in editor.urdfLinks:
        for v in link.urdf_visual_shapes:
            v.origin_xyz = (0, 0, 0)
        for c in link.urdf_collision_shapes:
            c.origin_xyz =  (0, 0, 0)

    for j in editor.urdfJoints:
        last_underscore_idx = j.joint_name.rfind("_")
        name = j.joint_name[:last_underscore_idx]
        if name in human_dict.urdf_to_smpl_dict.keys():
            urdf_name = human_dict.urdf_to_smpl_dict[name]
            # j.joint_origin_xyz = pos_dict[urdf_name]
            j.joint_origin_xyz = (0, 0, 0)
    editor.saveUrdf("test_mesh.urdf", True)