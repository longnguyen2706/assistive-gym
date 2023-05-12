import colorsys
import os
import pickle

import numpy as np
import pybullet as p
import pybullet_data
import torch
from gym.utils import seeding
from pytorch3d import transforms as t3d

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict
from experimental.urdf_editor import UrdfEditor


class HumanUrdf(Agent):
    def __init__(self):
        super(HumanUrdf, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_pip_dict = HumanPipDict()
        self.controllable_joint_indices = list(range(0, 93)) #94 joints

    def change_color(self, color):
        r"""
        Change the color of a robot.
        :param color: Vector4 for rgba.
        """
        for j in range(p.getNumJoints(self.human_id)):
            p.changeVisualShape(self.human_id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])

    # TODO: move to utils
    def load_smpl(self, filename):
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
    def convert_aa_to_euler(self, aa):
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

    def set_global_angle(self, human_id, pose):
        euler = self.convert_aa_to_euler(pose[self.smpl_dict.get_pose_ids("pelvis")])
        # euler = euler_convert_np(euler, from_seq='XYZ', to_seq='ZYX')
        quat = np.array(p.getQuaternionFromEuler(np.array(euler), physic_client_idId=human_id))
        p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)

    #TODO: move to utils
    def mul_tuple(self, t, multiplier):
        return tuple(multiplier * elem for elem in t)
    def set_joint_angle(self, human_id, pose, smpl_joint_name, robot_joint_name):
        smpl_angles = self.convert_aa_to_euler(pose[self.smpl_dict.get_pose_ids(smpl_joint_name)])

        # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
        robot_joints = self.human_pip_dict.get_joint_ids(robot_joint_name)
        for i in range(0, 3):
            p.resetJointState(human_id, robot_joints[i], smpl_angles[i])

    def set_joint_angles(self, smpl_data):

        print ("global_orient", smpl_data["global_orient"])
        print ("pelvis", smpl_data["body_pose"][0:3])
        pose = smpl_data["body_pose"]
        # global_orient = smpl_data["global_orient"]
        # global_orient = convert_aa_to_euler(global_orient)
        # quat = np.array(p.getQuaternionFromEuler(np.array(global_orient), physic_client_idId=human_id))
        # p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)
        # set_global_angle(human_id, pose)

        self.set_joint_angle(self.human_id, pose, "right_hip", "right_hip")
        self.set_joint_angle(self.human_id, pose, "lower_spine", "spine_2")
        self.set_joint_angle(self.human_id, pose, "middle_spine", "spine_3")
        self.set_joint_angle(self.human_id, pose, "upper_spine", "spine_4")

        self.set_joint_angle(self.human_id, pose, "left_hip", "left_hip")
        self.set_joint_angle(self.human_id, pose, "left_knee", "left_knee")
        self.set_joint_angle(self.human_id, pose, "left_ankle", "left_ankle")
        self.set_joint_angle(self.human_id, pose, "left_foot", "left_foot")

        self.set_joint_angle(self.human_id, pose, "right_hip", "right_hip")
        self.set_joint_angle(self.human_id, pose, "right_knee", "right_knee")
        self.set_joint_angle(self.human_id, pose, "right_ankle", "right_ankle")
        self.set_joint_angle(self.human_id, pose, "right_foot", "right_foot")

        self.set_joint_angle(self.human_id, pose, "right_collar", "right_clavicle")
        self.set_joint_angle(self.human_id, pose, "right_shoulder", "right_shoulder")
        self.set_joint_angle(self.human_id, pose, "right_elbow", "right_elbow")
        self.set_joint_angle(self.human_id, pose, "right_wrist", "right_lowarm")
        self.set_joint_angle(self.human_id, pose, "right_hand", "right_hand")

        self.set_joint_angle(self.human_id, pose, "left_collar", "left_clavicle")
        self.set_joint_angle(self.human_id, pose, "left_shoulder", "left_shoulder")
        self.set_joint_angle(self.human_id, pose, "left_elbow", "left_elbow")
        self.set_joint_angle(self.human_id, pose, "left_wrist", "left_lowarm")
        self.set_joint_angle(self.human_id, pose, "left_hand", "left_hand")

        self.set_joint_angle(self.human_id, pose, "neck", "neck")
        self.set_joint_angle(self.human_id, pose, "head", "head")

    def get_skin_color(self):
        hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
        hsv[-1] = np.random.uniform(0.4, 0.8)
        skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return skin_color

    def get_mesh(self):
        pass
    # func to demonstrate scaling the body part in urdf
    def scale_body_part(self, human_id, physic_client_id, scale=10):
        editor = UrdfEditor()
        editor.initializeFromBulletBody(human_id, physic_client_id) # load all properties to editor
        # scaling the robot
        for link in editor.urdfLinks:
            for v in link.urdf_visual_shapes:
                if v.geom_type ==  p.GEOM_BOX:
                    v.geom_extents = self.mul_tuple(v.geom_extents, 10)
                if v.geom_type ==  p.GEOM_SPHERE:
                    v.geom_radius *= 10
                if v.geom_type ==  p.GEOM_CAPSULE:
                    v.geom_radius *= 10
                    v.geom_length *= 10
                v.origin_xyz= self.mul_tuple(v.origin_xyz, 10)

            for c in link.urdf_collision_shapes:
                if c.geom_type == p.GEOM_BOX:
                    c.geom_extents = self.mul_tuple(c.geom_extents, 10)
                if c.geom_type == p.GEOM_SPHERE:
                    c.geom_radius *= 10
                if c.geom_type == p.GEOM_CAPSULE:
                    c.geom_radius *= 10
                    c.geom_length *= 10
                c.origin_xyz = self.mul_tuple(c.origin_xyz, 10)
        for j in editor.urdfJoints:
            j.joint_origin_xyz = self.mul_tuple(j.joint_origin_xyz, 10)
        editor.saveUrdf("test10.urdf", True)

    def init(self, id, np_random):
        # TODO: no hard coding
        self.human_id = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf")
        super(HumanUrdf, self).init(self.human_id, id, np_random)

if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    #plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])

    #bed
    bed = Furniture()
    bed.init("hospital_bed","assistive_gym/envs/assets/", physic_client_id, np_random)
    bed.set_on_ground()

    # human
    human = HumanUrdf()
    human.init(physic_client_id, np_random)
    human.change_color(human.get_skin_color())

    # print all the joints
    # for j in range(p.getNumJoints(human_id)):
    #     print (p.getJointInfo(human_id, j))
    # Set the simulation parameters
    p.setGravity(0,0,-9.81)

    bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    smpl_data = human.load_smpl(os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl"))
    human.set_joint_angles(smpl_data)
    human.set_on_ground(bed_height)
    # p.setJointMotorControlArray(human_id, [0,1,2,3,4,5,6,7,8,9,10,11,12,13], p.POSITION_CONTROL, targetPositions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0,0,1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

    while True:
        p.stepSimulation()
        pass
    # Disconnect from the simulation
    p.disconnect()
