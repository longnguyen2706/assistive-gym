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
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, generate_urdf, set_self_collisions


class HumanUrdf(Agent):
    def __init__(self):
        super(HumanUrdf, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_pip_dict = HumanPipDict()
        # self.controllable_joint_indices =  [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 19, 20, 21, 23, 24, 25, 27, 28, 29,
        #                                     31, 32, 33, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57,
        #                                     58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82,
        #                                     83, 85, 86, 87, 89, 90, 91, 93, 94, 95] # TODO: remove hardcoded
        self.controllable_joint_indices = list (range(0, 97))
        self.controllable = True
        self.motor_forces= 2000.0
        self.motor_gains = 0.05
        self.end_effectors = ['right_hand', 'left_hand', 'right_foot', 'left_foot', 'head']
    def change_color(self, color):
        r"""
        Change the color of a robot.
        :param color: Vector4 for rgba.
        """
        for j in range(p.getNumJoints(self.id)):
            p.changeVisualShape(self.id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])

    def set_joint_angles_with_smpl(self, smpl_data):

        print ("global_orient", smpl_data["global_orient"])
        print ("pelvis", smpl_data["body_pose"][0:3])
        pose = smpl_data["body_pose"]

        self.set_global_angle(self.body, pose)

        self._set_joint_angle(self.body, pose, "Spine1", "spine_2")
        self._set_joint_angle(self.body, pose, "Spine2", "spine_3")
        self._set_joint_angle(self.body, pose, "Spine3", "spine_4")

        self._set_joint_angle(self.body, pose, "L_Hip", "left_hip")
        self._set_joint_angle(self.body, pose, "L_Knee", "left_knee")
        self._set_joint_angle(self.body, pose, "L_Ankle", "left_ankle")
        self._set_joint_angle(self.body, pose, "L_Foot", "left_foot")

        self._set_joint_angle(self.body, pose, "R_Hip", "right_hip")
        self._set_joint_angle(self.body, pose, "R_Knee", "right_knee")
        self._set_joint_angle(self.body, pose, "R_Ankle", "right_ankle")
        self._set_joint_angle(self.body, pose, "R_Foot", "right_foot")

        self._set_joint_angle(self.body, pose, "R_Collar", "right_clavicle")
        self._set_joint_angle(self.body, pose, "R_Shoulder", "right_shoulder")
        self._set_joint_angle(self.body, pose, "R_Elbow", "right_elbow")
        self._set_joint_angle(self.body, pose, "R_Wrist", "right_lowarm")
        self._set_joint_angle(self.body, pose, "R_Hand", "right_hand")

        self._set_joint_angle(self.body, pose, "L_Collar", "left_clavicle")
        self._set_joint_angle(self.body, pose, "L_Shoulder", "left_shoulder")
        self._set_joint_angle(self.body, pose, "L_Elbow", "left_elbow")
        self._set_joint_angle(self.body, pose, "L_Wrist", "left_lowarm")
        self._set_joint_angle(self.body, pose, "L_Hand", "left_hand")

        self._set_joint_angle(self.body, pose, "Neck", "neck")
        self._set_joint_angle(self.body, pose, "Head", "head")

    def get_skin_color(self):
        hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
        hsv[-1] = np.random.uniform(0.4, 0.8)
        skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return skin_color

    def set_global_angle(self, human_id, pose):
        _, quat = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids("Pelvis")])
        # quat = np.array(p.getQuaternionFromEuler(np.array(euler)))
        p.resetBasePositionAndOrientation(human_id, [0, 0, 1], quat)

    def _set_joint_angle(self, human_id, pose, smpl_joint_name, robot_joint_name):
        smpl_angles, _ = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids(smpl_joint_name)])

        # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
        robot_joints = self.human_pip_dict.get_joint_ids(robot_joint_name)
        for i in range(0, 3):
            p.resetJointState(human_id, robot_joints[i], smpl_angles[i])

    def generate_human_mesh(self, id, physic_id, model_path):
        hull_dict, joint_pos_dict, _ = generate_geom(model_path)
        # now trying to scale the urdf file
        generate_urdf(id, physic_id, hull_dict, joint_pos_dict)
        # p.loadURDF("test_mesh.urdf", [0, 0, 0])

    def init(self, physics_id, np_random):
        # TODO: no hard coding
        # self.id = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf")
        # self.body = p.loadURDF("ref_mesh.urdf", useFixedBase=False) # enable self collision
        self.body = p.loadURDF("test_mesh.urdf", [0, 0, 0.1], flags=p.URDF_USE_SELF_COLLISION, useFixedBase=False)
        set_self_collisions(self.body, physics_id)
        super(HumanUrdf, self).init(self.body, physics_id, np_random)

    def get_movable_joints(self): # ignore all joints that are fixed
        pass
    def get_end_effector_indexes(self):
        ee_idxs = []
        for ee in self.end_effectors:
            ee_idxs.append(self.human_pip_dict.get_dammy_joint_id(ee)) # TODO: check if this is correct
        return ee_idxs
    def get_controllable_joints(self, joints=None):
        joint_states = p.getJointStates(self.body, self.all_joint_indices if joints is None else joints,
                                        physicsClientId=self.id)
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in
                       (self.all_joint_indices if joints is None else joints)]
        motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        return motor_indices

    def forward_kinematic(self, ee_idxs, joint_angles, joints=None):
        """
        :param link_idx: target link
        :param joint_angles: new joint angles
        :param joints: optional, if not given, use all controllable joints
        :return: Cartesian position of center of mass of the target link
        """
        original_angles = self.get_joint_angles(self.controllable_joint_indices)
        self.control(self.controllable_joint_indices, joint_angles, self.motor_gains, self.motor_forces) # forward to calculate fk
        _, motor_positions, _, _ = self.get_motor_joint_states()

        cos_positions = []
        for ee in ee_idxs:
            center_of_mass_pos = p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True,
                                            physicsClientId=self.id)[2]
            cos_positions.append(center_of_mass_pos) #TODO: check if motor_positions change after computeForwardKinematics
        self.set_joint_angles(self.controllable_joint_indices, original_angles) # reset to original angles
        return cos_positions, motor_positions

    def cal_manipulibility(self, joint_angles):
        # ee_idxs = self.get_end_effector_indexes()
        # J_arr = []
        # m_arr = [] # manipulibility
        # for ee in ee_idxs:
        #     _, motor_positions, _, _ = self.get_motor_joint_states()
        #     print ("motor_positions: ", motor_positions) #69
        #     joint_velocities = [0.0] * len(motor_positions)
        #     joint_accelerations = [0.0] * len(motor_positions)
        #     print ("body id ", self.body, " ee: ", ee, " physic client id: ", self.id)
        #     center_of_mass =  p.getLinkState(self.body, ee, computeLinkVelocity=False, computeForwardKinematics=False, physicsClientId=self.id)[
        #         2]
        #     print ("center of mass: ", center_of_mass)
        #     J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass,
        #                                               objPositions=motor_positions, objVelocities=joint_velocities,
        #                                               objAccelerations=joint_accelerations, physicsClientId=self.id)
        #     print("J linear: ", J_linear)
        #     J_linear = np.array(J_linear)
        #     J_angular = np.array(J_angular) # TODO: check if we only need part of it (right now it is 3* 75)
        #     J = np.concatenate([J_linear, J_angular], axis=0)
        #     m = np.sqrt(np.linalg.det(J @ J.T))
        #     J_arr.append(J)
        #     m_arr.append(m)
        #     print ("End effector idx: ", ee, "Jacobian_l: ", J_linear.shape, "Jacobian_r: ", J_angular.shape, "Manipulibility: ", m)
        # return np.mean(m_arr)

        J_arr = []
        m_arr = [] # manipulibility

        ee_idxes = self.get_end_effector_indexes()
        cos_positions, motor_positions = self.forward_kinematic(ee_idxes,  joint_angles)
        joint_velocities = [0.0] * len(motor_positions)
        joint_accelerations = [0.0] * len(motor_positions)

        for i in range(0, len(ee_idxes)):
            ee = ee_idxes[i]
            center_of_mass = cos_positions[i]
            J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass,
                                                          objPositions=motor_positions, objVelocities=joint_velocities,
                                                          objAccelerations=joint_accelerations, physicsClientId=self.id)
            # print("J linear: ", J_linear)
            J_linear = np.array(J_linear)
            J_angular = np.array(J_angular) # TODO: check if we only need part of it (right now it is 3* 75)
            J = np.concatenate([J_linear, J_angular], axis=0)
            m = np.sqrt(np.linalg.det(J @ J.T))
            J_arr.append(J)
            m_arr.append(m)
            # print ("End effector idx: ", ee, "Jacobian_l: ", J_linear.shape, "Jacobian_r: ", J_angular.shape, "Manipulibility: ", m)
        avg_manipubility = np.mean(m_arr)
        print ("manipubility: ", avg_manipubility)
        return avg_manipubility


if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)

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
    # for j in range(p.getNumJoints(id)):
    #     print (p.getJointInfo(id, j))
    # Set the simulation parameters
    p.setGravity(0,0,-9.81)

    # bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    smpl_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    smpl_data = load_smpl(smpl_path)
    # human.set_joint_angles_with_smpl(smpl_data)
    # human.set_on_ground(bed_height)
    human.generate_human_mesh(human.body, physic_client_id, smpl_path)

    # p.setJointMotorControlArray(id, [0,1,2,3,4,5,6,7,8,9,10,11,12,13], p.POSITION_CONTROL, targetPositions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0,0,1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

    # while True:
    #     p.stepSimulation()

    # Disconnect from the simulation
    p.disconnect()
