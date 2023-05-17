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
    print("body_pose: ", data["body_pose"].shape, "betas: ", data["betas"].shape, "global_orient: ",
          data["global_orient"].shape)
    return data


def convert_aa_to_euler_quat(aa, seq="YZX"):
    aa = np.array(aa)
    mat = t3d.axis_angle_to_matrix(torch.from_numpy(aa))
    # print ("mat", mat)
    quat = t3d.matrix_to_quaternion(mat)
    euler = t3d.matrix_to_euler_angles(mat, seq)
    return euler, quat


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
    editor.initializeFromBulletBody(human_id, physic_client_id)  # load all properties to editor
    # scaling the robot
    for link in editor.urdfLinks:
        for v in link.urdf_visual_shapes:
            if v.geom_type == p.GEOM_BOX:
                v.geom_extents = mul_tuple(v.geom_extents, 10)
            if v.geom_type == p.GEOM_SPHERE:
                v.geom_radius *= 10
            if v.geom_type == p.GEOM_CAPSULE:
                v.geom_radius *= 10
                v.geom_length *= 10
            v.origin_xyz = mul_tuple(v.origin_xyz, 10)

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


def get_bodypart_name(urdf_name):
    """
    :param urdf_name: joint or link name in urdf
    :return:
    """
    # TODO: check for _
    last_underscore_idx = urdf_name.rfind("_")
    name = urdf_name[:last_underscore_idx]
    suffix = urdf_name[last_underscore_idx + 1:]
    return name, suffix


def reposition_body_part(human_id, physic_client_id, pos_dict):
    editor = UrdfEditor()
    editor.initializeFromBulletBody(human_id, physic_client_id)  # load all properties to editor
    human_dict = HumanPipDict()

    for link in editor.urdfLinks:
        name, suffix = get_bodypart_name(link.link_name)
        if name in human_dict.urdf_to_smpl_dict.keys():
            xyz = (0, 0, 0)
            for v in link.urdf_visual_shapes:
                v.origin_xyz = xyz
            for c in link.urdf_collision_shapes:
                c.origin_xyz = xyz

            # inertia
            link.urdf_inertial.origin_xyz = xyz
            if suffix == 'limb':
                link.urdf_inertial.mass = 2

    # smpl joint is spherical.
    # due to the limitation of pybullet (no joint limit for spherical joint),
    # we need to decompose it to 3 revolute joints (rx, ry, rz) and 1 fixed joint (rzdammy) rx -> ry -> rz -> rzdammy
    # the human body part is a link (_limb) that attached to the fixed joint
    # we need to move the joint origin to correct position as follow
    # - set rx joint origin = urdf_joint pos - parent urdf joint pos
    # - set ry, rz, rxdammy origin = 0 (superimpose on the rx joint)
    for j in editor.urdfJoints:
        name, suffix = get_bodypart_name(j.joint_name)
        if name in human_dict.urdf_to_smpl_dict.keys():
            urdf_name = human_dict.urdf_to_smpl_dict[name]
            if suffix == 'rx':
                pos = pos_dict[urdf_name]

                parent = human_dict.joint_to_parent_joint_dict[name]
                pos_parent = pos_dict[human_dict.urdf_to_smpl_dict[parent]]
                xyz = pos - pos_parent
            else:
                xyz = (0, 0, 0)
            j.joint_origin_xyz = xyz
        else:
            j.joint_origin_xyz = (0, 0, 0)
        j.joint_lower_limit = -3.14
        j.joint_upper_limit = 3.14

    editor.saveUrdf("test_mesh.urdf", True)
