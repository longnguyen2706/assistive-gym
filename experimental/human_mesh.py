import colorsys
import os

import numpy as np
import pybullet as p
import pybullet_data
from gym.utils import seeding

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict

from assistive_gym.envs.utils.smpl_geom import generate_geom
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, reposition_body_part


class HumanUrdf(Agent):
    def __init__(self):
        super(HumanUrdf, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_pip_dict = HumanPipDict()
        self.controllable_joint_indices = list(range(0, 97)) #97 joints

    def change_color(self, color):
        r"""
        Change the color of a robot.
        :param color: Vector4 for rgba.
        """
        for j in range(p.getNumJoints(self.human_id)):
            p.changeVisualShape(self.human_id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])

    def set_joint_angles(self, smpl_data):

        print ("global_orient", smpl_data["global_orient"])
        print ("pelvis", smpl_data["body_pose"][0:3])
        pose = smpl_data["body_pose"]
        # global_orient = smpl_data["global_orient"]
        # global_orient = convert_aa_to_euler(global_orient)
        # quat = np.array(p.getQuaternionFromEuler(np.array(global_orient), physic_client_idId=human_id))
        # p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)
        # set_global_angle(human_id, pose)

        # self.set_joint_angle(self.human_id, pose, "R_Hip", "right_hip")
        # self.set_joint_angle(self.human_id, pose, "Spine1", "spine_2")
        # self.set_joint_angle(self.human_id, pose, "Spine2", "spine_3")
        # self.set_joint_angle(self.human_id, pose, "Spine3", "spine_4")
        #
        # self.set_joint_angle(self.human_id, pose, "L_Hip", "left_hip")
        # self.set_joint_angle(self.human_id, pose, "L_Knee", "left_knee")
        # self.set_joint_angle(self.human_id, pose, "L_Ankle", "left_ankle")
        # self.set_joint_angle(self.human_id, pose, "L_Foot", "left_foot")
        #
        # self.set_joint_angle(self.human_id, pose, "R_Hip", "right_hip")
        # self.set_joint_angle(self.human_id, pose, "R_Knee", "right_knee")
        # self.set_joint_angle(self.human_id, pose, "R_Ankle", "right_ankle")
        # self.set_joint_angle(self.human_id, pose, "R_Foot", "right_foot")
        #
        # self.set_joint_angle(self.human_id, pose, "R_Collar", "right_clavicle")
        # self.set_joint_angle(self.human_id, pose, "R_Shoulder", "right_shoulder")
        # self.set_joint_angle(self.human_id, pose, "R_Elbow", "right_elbow")
        # self.set_joint_angle(self.human_id, pose, "R_Wrist", "right_lowarm")
        # self.set_joint_angle(self.human_id, pose, "R_Hand", "right_hand")
        #
        # self.set_joint_angle(self.human_id, pose, "L_Collar", "left_clavicle")
        # self.set_joint_angle(self.human_id, pose, "L_Shoulder", "left_shoulder")
        # self.set_joint_angle(self.human_id, pose, "L_Elbow", "left_elbow")
        # self.set_joint_angle(self.human_id, pose, "L_Wrist", "left_lowarm")
        # self.set_joint_angle(self.human_id, pose, "L_Hand", "left_hand")
        #
        # self.set_joint_angle(self.human_id, pose, "Neck", "neck")
        # self.set_joint_angle(self.human_id, pose, "Head", "head")

    def get_skin_color(self):
        hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
        hsv[-1] = np.random.uniform(0.4, 0.8)
        skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return skin_color

    def set_global_angle(self, human_id, pose):
        euler, quat = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids("pelvis")])
        # euler = euler_convert_np(euler, from_seq='XYZ', to_seq='ZYX')
        # quat = np.array(p.getQuaternionFromEuler(np.array(euler), physic_client_idId=human_id))
        p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)

    def set_joint_angle(self, human_id, pose, smpl_joint_name, robot_joint_name):
        smpl_angles = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids(smpl_joint_name)])

        # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
        robot_joints = self.human_pip_dict.get_joint_ids(robot_joint_name)
        for i in range(0, 3):
            p.resetJointState(human_id, robot_joints[i], smpl_angles[i])

    def generate_human_mesh(self, id, physic_id, model_path):
        hull_dict, joint_pos_dict, joint_offset_dict = generate_geom(model_path)
        # now trying to scale the urdf file
        reposition_body_part(id, physic_id, joint_pos_dict)

        # p.loadURDF("test_mesh.urdf", [0, 0, 0])



    def init(self, id, np_random):
        # TODO: no hard coding
        # self.human_id = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf")
        self.human_id = p.loadURDF("ref_mesh.urdf")
        super(HumanUrdf, self).init(self.human_id, id, np_random)

if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    #plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])

    #bed
    # bed = Furniture()
    # bed.init("hospital_bed","assistive_gym/envs/assets/", physic_client_id, np_random)
    # bed.set_on_ground()

    # human
    human = HumanUrdf()
    human.init(physic_client_id, np_random)
    # human.change_color(human.get_skin_color())

    # print all the joints
    # for j in range(p.getNumJoints(human_id)):
    #     print (p.getJointInfo(human_id, j))
    # Set the simulation parameters
    p.setGravity(0,0,-9.81)

    # bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    smpl_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    smpl_data = load_smpl(smpl_path)
    # human.set_joint_angles(smpl_data)
    # human.set_on_ground(bed_height)
    human.generate_human_mesh(human.human_id, physic_client_id, smpl_path)

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
