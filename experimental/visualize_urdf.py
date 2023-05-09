# TODO: clean up this file
import pickle

import numpy as np
import pybullet as p
import pybullet_data
from experimental.urdf_editor import UrdfEditor

from pytorch3d import transforms as t3d
import torch
import os

from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict

smpl_dict = SMPLDict()
human_pip_dict = HumanPipDict()
def change_color(id_robot, color):
    r"""
    Change the color of a robot.

    :param id_robot: Robot id.
    :param color: Vector4 for rgba.
    """
    for j in range(p.getNumJoints(id_robot)):
        p.changeVisualShape(id_robot, j, rgbaColor=color)

def load_smpl(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)

    print("data keys", data.keys())
    print(
        "body_pose: ",
        data["body_pose"].shape,
        "betas: ",
        data["betas"].shape,
        "global_orient: ",
        data["global_orient"].shape,
    )

    return data
def convert_aa_to_euler(aa):
    aa = np.array(aa)
    mat = t3d.axis_angle_to_matrix(torch.from_numpy(aa))
    # print ("mat", mat)
    quats = t3d.matrix_to_quaternion(mat)
    euler = t3d.matrix_to_euler_angles(mat, "XYZ")
    return euler

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
def set_global_angle(robotId, pose):
    euler = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("pelvis")])
    # euler = euler_convert_np(euler, from_seq='XYZ', to_seq='ZYX')
    quat = np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=robotId))
    p.resetBasePositionAndOrientation(robotId, [0, 0, 0], quat)

def mul_tuple(t, multiplier):
    return tuple(multiplier * elem for elem in t)
def set_joint_angle(robotId, pose, smpl_joint_name, robot_joint_name):
    smpl_angles = convert_aa_to_euler(pose[smpl_dict.get_pose_ids(smpl_joint_name)])

    # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
    robot_joints = human_pip_dict.get_joint_ids(robot_joint_name)
    for i in range(0, 3):
        p.resetJointState(robotId, robot_joints[i], smpl_angles[i])
    # y, z, x
    # p.resetJointState(robotId, robot_joints[0], smpl_angles[1])
    # p.resetJointState(robotId, robot_joints[1], smpl_angles[2])
    # p.resetJointState(robotId, robot_joints[2], smpl_angles[0])
# Start the simulation engine
physicsClient = p.connect(p.GUI)

# Load the URDF file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("../assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])
# robotId = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf", [0, 0, 0], [0, 0, 0,1])
robotId = p.loadURDF("assistive_gym/envs/assets/human/test10.urdf", [0, 0, 0], [0, 0, 0,1])
# change_color(robotId, [1,0,0,1])

# print all the joints
for j in range(p.getNumJoints(robotId)):
    print (p.getJointInfo(robotId, j))
# Set the simulation parameters
p.setGravity(0,0,-9.81)
p.setTimeStep(1./240.)

# set the pose

# smpl_data = load_smpl( os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_3.pkl"))
# print ("global_orient", smpl_data["global_orient"])
# print ("pelvis", smpl_data["body_pose"][0:3])
# pose = smpl_data["body_pose"]
# # global_orient = smpl_data["global_orient"]
# # global_orient = convert_aa_to_euler(global_orient)
# # quat = np.array(p.getQuaternionFromEuler(np.array(global_orient), physicsClientId=robotId))
# # p.resetBasePositionAndOrientation(robotId, [0, 0, 0], quat)
# set_global_angle(robotId, pose)
#
# set_joint_angle(robotId, pose, "right_hip", "right_hip")
# set_joint_angle(robotId, pose, "lower_spine", "spine_2")
# set_joint_angle(robotId, pose, "middle_spine", "spine_3")
# set_joint_angle(robotId, pose, "upper_spine", "spine_4")
#
# set_joint_angle(robotId, pose, "left_hip", "left_hip")
# set_joint_angle(robotId, pose, "left_knee", "left_knee")
# set_joint_angle(robotId, pose, "left_ankle", "left_ankle")
# set_joint_angle(robotId, pose, "left_foot", "left_foot")
#
# set_joint_angle(robotId, pose, "right_hip", "right_hip")
# set_joint_angle(robotId, pose, "right_knee", "right_knee")
# set_joint_angle(robotId, pose, "right_ankle", "right_ankle")
# set_joint_angle(robotId, pose, "right_foot", "right_foot")
#
# set_joint_angle(robotId, pose, "right_collar", "right_clavicle")
# set_joint_angle(robotId, pose, "right_shoulder", "right_shoulder")
# set_joint_angle(robotId, pose, "right_elbow", "right_elbow")
# set_joint_angle(robotId, pose, "right_wrist", "right_lowarm")
# set_joint_angle(robotId, pose, "right_hand", "right_hand")
#
# set_joint_angle(robotId, pose, "left_collar", "left_clavicle")
# set_joint_angle(robotId, pose, "left_shoulder", "left_shoulder")
# set_joint_angle(robotId, pose, "left_elbow", "left_elbow")
# set_joint_angle(robotId, pose, "left_wrist", "left_lowarm")
# set_joint_angle(robotId, pose, "left_hand", "left_hand")
#
# set_joint_angle(robotId, pose, "neck", "neck")
# set_joint_angle(robotId, pose, "head", "head")

# move the robot
# p.setJointMotorControlArray(robotId, [0,1,2,3,4,5,6,7,8,9,10,11,12,13], p.POSITION_CONTROL, targetPositions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])

# editor = UrdfEditor()
# editor.initializeFromBulletBody(robotId, physicsClient) # load all properties to editor
# # scaling the robot
# for link in editor.urdfLinks:
#     for v in link.urdf_visual_shapes:
#         if v.geom_type ==  p.GEOM_BOX:
#             v.geom_extents = mul_tuple(v.geom_extents, 10)
#         if v.geom_type ==  p.GEOM_SPHERE:
#             v.geom_radius *= 10
#         if v.geom_type ==  p.GEOM_CAPSULE:
#             v.geom_radius *= 10
#             v.geom_length *= 10
#         v.origin_xyz= mul_tuple(v.origin_xyz, 10)
#
#     for c in link.urdf_collision_shapes:
#         if c.geom_type == p.GEOM_BOX:
#             c.geom_extents = mul_tuple(c.geom_extents, 10)
#         if c.geom_type == p.GEOM_SPHERE:
#             c.geom_radius *= 10
#         if c.geom_type == p.GEOM_CAPSULE:
#             c.geom_radius *= 10
#             c.geom_length *= 10
#         c.origin_xyz = mul_tuple(c.origin_xyz, 10)
# for j in editor.urdfJoints:
#     j.joint_origin_xyz = mul_tuple(j.joint_origin_xyz, 10)
# editor.saveUrdf("test10.urdf", True)


# Set the camera view
cameraDistance = 3
cameraYaw = 0
cameraPitch = -30
cameraTargetPosition = [0,0,1]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
while True:
    # p.stepSimulation()'
    pass
# Disconnect from the simulation
p.disconnect()


