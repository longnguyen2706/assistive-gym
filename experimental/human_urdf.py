import colorsys
import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from numpy.linalg import norm
from cma import CMAEvolutionStrategy
from gym.utils import seeding
from kinpy import Transform
from ergonomics.reba import RebaScore

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.human_utils import set_self_collisions, change_dynamic_properties, check_collision, \
    set_joint_angles, set_global_orientation, set_joint_angles_2
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.plot_utils import plot
from assistive_gym.envs.utils.smpl_dict import SMPLDict
from scipy.spatial.transform import Rotation as R

from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, generate_urdf, SMPLData
import kinpy as kp

#######################################  Static settting ##########################################
# URDF_PATH = os.path.join(os.getcwd(), "test_mesh.urdf")

# generated by running self.print_all_joints(). TODO: automate this
all_controllable_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27,
                                  29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53,
                                  54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78,
                                  79, 81, 82, 83, 85, 86, 87, 89, 90, 91]
left_leg_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
right_leg_joint_indices = [17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
# left_arm_joint_indices = [53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71]
left_arm_joint_indices = [53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71] # added clavicle
# right_arm_joint_indices =  [77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91]
right_arm_joint_indices = [73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91]  # with clavicle
body_joint_indices = [33, 34, 35, 37, 38, 39, 41, 42, 43]
all_joint_indices = list(range(0, 93))
##################################################################################################

LOG = get_logger()
class HumanUrdf(Agent):
    def __init__(self):
        super(HumanUrdf, self).__init__()
        # variables for assistive gym agent
        self.controllable_joint_indices = right_arm_joint_indices
        self.controllable = True
        self.motor_forces = 1.0
        self.motor_gains = 0.005

        # variables for human urdf
        self.smpl_dict = SMPLDict()
        self.human_dict = HumanUrdfDict()
        self.urdf_path = None
        self.initial_self_collisions = set()  # collision due to initial pose

        # will be init in self.init()
        self.chain = None

    def set_urdf_path(self, urdf_path):
        self.urdf_path = urdf_path

    def find_ik_joint_indices(self):
        ik_indices = []
        for i in self.controllable_joint_indices:
            counter = 0
            for j in all_joint_indices:
                if i == j:
                    ik_indices.append(counter)
                joint_type = p.getJointInfo(self.body, j, physicsClientId=self.id)[2]
                if joint_type != p.JOINT_FIXED:
                    counter += 1
        return ik_indices

    def set_joint_angles_with_smpl(self, smpl_data: SMPLData, use_limits=True):
        # set_joint_angles(self.body, smpl_data.body_pose)
        # self.initial_self_collisions = self.check_self_collision()  # collision due to initial pose
        pose = smpl_data.body_pose
        self.set_joint_angle_with_limit(pose, "Spine1", "spine_2", use_limits)
        self.set_joint_angle_with_limit( pose, "Spine2", "spine_3", use_limits)
        self.set_joint_angle_with_limit( pose, "Spine3", "spine_4", use_limits)

        self.set_joint_angle_with_limit( pose, "L_Hip", "left_hip", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Knee", "left_knee", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Ankle", "left_ankle", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Foot", "left_foot", use_limits)

        self.set_joint_angle_with_limit( pose, "R_Hip", "right_hip", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Knee", "right_knee", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Ankle", "right_ankle", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Foot", "right_foot", use_limits)

        self.set_joint_angle_with_limit( pose, "R_Collar", "right_clavicle", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Shoulder", "right_shoulder", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Elbow", "right_elbow", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Wrist", "right_lowarm", use_limits)
        self.set_joint_angle_with_limit( pose, "R_Hand", "right_hand", use_limits)

        self.set_joint_angle_with_limit( pose, "L_Collar", "left_clavicle", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Shoulder", "left_shoulder", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Elbow", "left_elbow", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Wrist", "left_lowarm", use_limits)
        self.set_joint_angle_with_limit( pose, "L_Hand", "left_hand", use_limits)

        self.set_joint_angle_with_limit( pose, "Neck", "neck", use_limits)
        self.set_joint_angle_with_limit( pose, "Head", "head", use_limits)

    def set_joint_angles_with_smpl2(self, smpl_data: SMPLData):
        set_joint_angles_2(self.body, smpl_data.body_pose)
        # self.initial_self_collisions = self.check_self_collision()  # collision due to initial pose
        
    def set_joint_angle_with_limit(self, pose, smpl_joint_name, robot_joint_name, use_limits):
        smpl_dict = SMPLDict()
        smpl_angles, _ = convert_aa_to_euler_quat(pose[smpl_dict.get_pose_ids(smpl_joint_name)])

        robot_joints = self.human_dict.get_joint_ids(robot_joint_name)
        print ("joint name: ", smpl_joint_name, " angles: ", smpl_angles*180/np.pi)
        self.set_joint_angles(robot_joints, smpl_angles, use_limits=use_limits)

    def set_global_orientation(self, smpl_data: SMPLData, pos):
        set_global_orientation(self.body, smpl_data.global_orient, pos)

    def reset_controllable_joints(self, end_effector):
        if end_effector not in ['left_hand', 'right_hand']:
            raise ValueError("end_effector must be either 'left_hand' or 'right_hand'")
        self.controllable_joint_indices = left_arm_joint_indices if end_effector == 'left_hand' else right_arm_joint_indices
        self.controllable_joint_lower_limits = np.array([self.lower_limits[i] for i in self.controllable_joint_indices])
        self.controllable_joint_upper_limits = np.array([self.upper_limits[i] for i in self.controllable_joint_indices])

    def fit_joint_angle(self, target_angles, start_angles=None):
        def cost_fn(current_angles, target_angles):
            cost = np.sqrt(np.sum(np.square(current_angles - target_angles)))

            if len(self.check_self_collision()) > 0:
                cost += 100
            print("cost: ", cost)
            return cost

        def init_optimizer(x0, sigma):
            opts = {}
            opts['tolfun'] = 1e-3
            opts['tolx'] = 1e-3
            es = CMAEvolutionStrategy(x0, sigma, opts)
            return es

        if not start_angles:
            x0 = np.zeros(len(target_angles))
        else:
            x0 = np.array(start_angles)
        x0[0:3] = target_angles[0:3]  # global angle
        print("x0: ", x0)
        cma = init_optimizer(x0, 0.5)
        mean_cost = []
        mean_evolution = []
        while not cma.stop():
            solutions = cma.ask()
            costs = []
            for s in solutions:
                set_joint_angles(self.body, s)
                costs.append(cost_fn(s, target_angles))
                mean_evolution.append(np.mean(solutions, axis=0))
                mean_cost.append(np.mean(costs))
            cma.tell(solutions, costs)
            cma.LOG.add()
            cma.disp()
        self._set_joint_angles(cma.best.x)
        plot(mean_cost, "Cost Function", "Iteration", "Cost")
        plot(mean_evolution, "Mean Evolution", "Iteration", "Mean Evolution")

    def init(self, physics_id, np_random):
        self.body = p.loadURDF(self.urdf_path, [0, 0, 0],
                               flags=p.URDF_USE_SELF_COLLISION,
                               useFixedBase=False)
        self._init_kinematic_chain()

        # set_self_collisions(self.body, physics_id)

        # set contact damping
        num_joints = p.getNumJoints(self.body, physicsClientId=physics_id)
        change_dynamic_properties(self.body, list(range(0, num_joints)))

        # enable force torque sensor
        for i in self.controllable_joint_indices:
            p.enableJointForceTorqueSensor(self.body, i, enableSensor=True, physicsClientId=physics_id)

        super(HumanUrdf, self).init(self.body, physics_id, np_random)

    def _get_end_link_and_root_link_name(self, ee: str):
        if ee == "right_hand":
            end_link_name, root_link_name = "right_hand_limb", "spine_4_limb"
        elif ee == "left_hand":
            end_link_name, root_link_name = "left_hand_limb", "spine_4_limb"
        elif ee == "right_foot":
            end_link_name, root_link_name = "right_foot_limb", "pelvis_limb"
        elif ee == "left_foot":
            end_link_name, root_link_name = "left_foot_limb", "pelvis_limb"
        elif ee == "head":
            end_link_name, root_link_name = "head_limb", "spine_4_limb"
        else:
            raise NotImplementedError
        return end_link_name, root_link_name

    def _init_kinematic_chain(self):
        chain = {}
        chain['whole_body'] = kp.build_chain_from_urdf(open(self.urdf_path).read())
        for ee in self.human_dict.end_effectors:
            end_link_name, root_link_name = self._get_end_link_and_root_link_name(ee)
            chain[ee] = kp.build_serial_chain_from_urdf(open(self.urdf_path).read(),
                                                        end_link_name=end_link_name,
                                                        root_link_name=root_link_name)
        self.chain = chain

    def _get_end_effector_indexes(self, ee_names):
        ee_idxs = []
        for ee in ee_names:
            ee_idxs.append(self.human_dict.get_dammy_joint_id(ee))  # TODO: check if this is correct
        return ee_idxs

    def _get_controllable_joints(self, joints=None):
        joint_states = p.getJointStates(self.body, self.all_joint_indices if joints is None else joints,
                                        physicsClientId=self.id)
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in
                       (self.all_joint_indices if joints is None else joints)]
        motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        return motor_indices

    # this method wont be precise as the control api only move the joint in the time defined by 1 step only, so not neccessary the end position
    def forward_kinematic(self, ee_idxs, joint_angles):
        """
        :param link_idx: target link
        :param joint_angles: new joint angles
        :param joints: optional, if not given, use all controllable joints
        :return: Cartesian position of center of mass of the target link
        """
        original_angles = self.get_joint_angles(self.controllable_joint_indices)
        LOG.debug(f"original_angles: {original_angles.shape}, joint_angles: {joint_angles.shape}")
        self.control(self.controllable_joint_indices, joint_angles, 0, 0)  # forward to calculate fk
        self.step_simulation()
        _, motor_positions, _, _ = self.get_motor_joint_states()

        ee_positions = []
        for ee in ee_idxs:
            ee_pos = p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True,
                                    physicsClientId=self.id)[0]
            ee_positions.append(ee_pos)  # TODO: check if motor_positions change after computeForwardKinematics
        self.set_joint_angles(self.controllable_joint_indices, original_angles,
                              velocities=0.0)  # reset to original angles
        self.step_simulation()
        return ee_positions, motor_positions

    # inverse kinematic using kinpy
    # most of the case it wont be able to find a solution as good as pybullet
    def ik_chain(self, ee_pos, ee_quat=[1, 0, 0, 0], ee_name='right_hand'):
        """
        :param ee_pos:
        :param ee_quat:
        :return: ik solutions (angles) for all joints in chain
        """
        t = Transform(None, ee_pos) # TODO: check if this is correct t = Transform(ee_quat, ee_pos)
        chain = self.chain[ee_name]
        LOG.info(self.right_hand_chain.get_joint_parameter_names())
        return self.right_hand_chain.inverse_kinematics(t)

    # fk for a chain using kinpy
    def fk_chain(self, target_angles, ee: str):
        """
        :param target_angles:
        :return: pos of end effector
        """
        th = {}
        chain = self.chain[ee]
        for idx, joint in enumerate(chain.get_joint_parameter_names()):
            th[joint] = target_angles[idx]
        # print(th)
        ret = chain.forward_kinematics(th)
        return ret.pos

    # forward kinematics using kinpy
    def fk(self, ee_names, target_angles):
        th = {}
        chain = self.chain["whole_body"]
        for idx, joint in enumerate(chain.get_joint_parameter_names()):
            th[joint] = target_angles[idx]
        # print(th)
        g_pos, g_orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        # [x y z w] to [w x y z] format
        g_quat = [g_orient[3]] + list(g_orient[:3])
        # print (g_pos, g_orient, g_quat)
        ret = chain.forward_kinematics(th, world=Transform(g_quat, list(g_pos)))
        # ret = chain.forward_kinematics(th)
        # print(ret)

        j_angles = []
        for key in ret:
            q_rot = ret[key].rot
            # w x y z to x y z w
            rot = list(q_rot[1:]) + [q_rot[0]]
            j_angles.append(rot)
        ee_pos = []
        for ee_name in ee_names:
            ee_pos.append(ret[ee_name].pos)
        # J= self.chain.jacobian(target_angles)
        return ee_pos, j_angles, None

    def step_simulation(self):
        for _ in range(5):  # 5 is the number of skip steps
            p.stepSimulation(physicsClientId=self.id)


    def get_reba_score(self, end_effector="right_hand"):
        human_dict = HumanUrdfDict()
        rebaScore = RebaScore()
        # list joints in the order required for a reba score
        joints = ["head", "neck", "left_shoulder", "left_elbow", "left_lowarm", "right_shoulder", "right_elbow", "right_lowarm", # 7
            "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "left_hand", "right_hand"] # 15
        
        # obtain the links in the right order for the rebascore code
        dammy_ids = []
        for joint in joints:
            dammy_ids.append(human_dict.get_dammy_joint_id(joint))

        # use dammy ids to obtain the right link, use the right knee [12] as the root joint
        pose = []

        for i in dammy_ids:
            # get the location of each dammy joint and append to the pose list
            loc = p.getLinkState(self.body, i)[4]            
            pose.append(loc)
    
        pose = np.array(pose)
        # following code is from the ergonomic repo (https://github.com/rs9000/ergonomics/blob/master/ergonomics/reba.py
        if end_effector == "right_hand":
            arms_params = rebaScore.get_arms_angles_from_pose_right(pose)
        else:
            arms_params = rebaScore.get_arms_angles_from_pose_left(pose)

        # calculate scores
        rebaScore.set_arms(arms_params)
        _, partial_b = rebaScore.compute_score_b()
        arm_score = np.sum(partial_b)
        
        # return all info
        return arm_score


    def get_vertical_offset(self, end_effector="right_hand"):
        human_dict = HumanUrdfDict()
        # determine wrist index for the correct hand
        _, ee_orient = self.get_ee_pos_orient(end_effector)
        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 1]
        goal = [0, 0, 1]
        cosine = np.dot(ray_dir, goal)/(norm(ray_dir)*norm(goal))
        # print("Cosine Similarity:", cosine)

        return cosine

    def get_parallel_offset(self, end_effector="right_hand"):
        # determine wrist index for the correct hand
        _, ee_orient = self.get_ee_pos_orient(end_effector)
        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 2]
        goal = [0, 0, 1]
        cosine = np.dot(ray_dir, goal)/(norm(ray_dir)*norm(goal))
        # print("Cosine Similarity:", cosine)

        return cosine

    def get_yaw_wrist_orientation(self, end_effector="right_hand"):
        human_dict = HumanUrdfDict()
        # determine wrist index for the correct hand
        wrist_ind = human_dict.get_dammy_joint_id(end_effector)
        wrist_orientation = p.getLinkState(self.body, wrist_ind)[1]
        array = p.getEulerFromQuaternion(wrist_orientation)
        return array[2]

    
    def get_eyeline_offset(self, end_effector):
        fov = self.get_fov()
        right = fov[0]
        left = fov[1]
        hand_pos, _ = self.get_ee_pos_orient(end_effector)
        hand_x = hand_pos[0]

        # testing
        ee_pos, _ = self.get_ee_pos_orient("head")
        p.addUserDebugLine(ee_pos, left, [0, 1, 0])
        p.addUserDebugLine(ee_pos, right, [1, 0, 0])
        p.removeAllUserDebugItems
        print("left: ", left, "\nright: ", right)
        l = left[0]
        r = right[0]
        if hand_x >= l and hand_x <= r:
            return 0
        return min(abs(hand_x - l), abs(hand_x - r))

    def get_eyeline_cone(self, end_effector, l=0.5):
        ee_pos, ee_orient = self.get_ee_pos_orient("head")
        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 2]
        end_cone = [ee_pos[0] + (ray_dir[0]*l), ee_pos[1] + (ray_dir[1]*l), ee_pos[2] + (ray_dir[2]*l)]
        radius = 5
        perp = self.get_orthoganol(ee_orient)
        end_rad = [end_cone[0] + (perp[0]*radius), end_cone[1] + (perp[1]*radius), end_cone[2] + (perp[2]*radius)]
        end_rad0 = [end_cone[0] + (perp[0]*-radius), end_cone[1] + (perp[1]*-radius), end_cone[2] + (perp[2]*-radius)]
        perp = self.get_orthoganol(perp)
        end_rad2 = [end_cone[0] + (perp[0]*radius), end_cone[1] + (perp[1]*radius), end_cone[2] + (perp[2]*radius)]
        end_rad20 = [end_cone[0] + (perp[0]*-radius), end_cone[1] + (perp[1]*-radius), end_cone[2] + (perp[2]*-radius)]
        
        # ensure calculations were done correctly
        p.addUserDebugLine(ee_pos, end_cone, [1, 0, 0]) 
        p.addUserDebugLine(end_cone, end_rad, [0, 1, 0])
        p.addUserDebugLine(end_cone, end_rad0, [0, 0, 1])
        p.addUserDebugLine(end_cone, end_rad2, [0, 1, 0])
        p.addUserDebugLine(end_cone, end_rad20, [0, 0, 1])
        p.removeAllUserDebugItems()
        # idea, using the angle made between the head normal and [1, 0, 0] --> if greater or less than a certain threshold we can force
        # the hand in a certain direction or change how we model the cone of vision

        return False


    def get_fov(self, l=0.25):
        # casts field of view from head 0.5m outward to define a line of sight -- will check that 0.5m is reasonable
        ee_pos, ee_orient = self.get_ee_pos_orient("head")
        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 2]
        axis = np.cross(ray_dir, [1, 0, 0])
        left = self.rotate_3d(ray_dir, axis, 70)
        right = self.rotate_3d(ray_dir, axis, -70)
        ray_dir = left
        end_left = [ee_pos[0] + (ray_dir[0]*l), ee_pos[1] + (ray_dir[1]*l), ee_pos[2] + (ray_dir[2]*l)]
        ray_dir = right
        end_right = [ee_pos[0] + (ray_dir[0]*l), ee_pos[1] + (ray_dir[1]*l), ee_pos[2] + (ray_dir[2]*l)]
        return [end_left, end_right]

    def set_head_angle(self, l=0.25):
        ee_pos, ee_orient = self.get_ee_pos_orient("head")
        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 2]
        # head = np.array([ee_pos[0] + (ray_dir[0]*l), ee_pos[1] + (ray_dir[1]*l), ee_pos[2] + (ray_dir[2]*l)])
        head_u = ray_dir / np.linalg.norm(ray_dir)
        x = np.array([1, 0, 0])
        x_u = x / np.linalg.norm(x)
        angle = np.arccos(np.dot(head_u, x_u))
        # based on this angle, we can force the hand in the desired direction
        print("head anlge: ", angle)

        #idea 1
        end_norm = np.array([ee_pos[0] + (ray_dir[0]*l), ee_pos[1] + (ray_dir[1]*l), ee_pos[2] + (ray_dir[2]*l)])
        ray_dir = [1, 0, 0]
        end_l = np.array([end_norm[0] + (ray_dir[0]*l), end_norm[1] + (ray_dir[1]*l), end_norm[2] + (ray_dir[2]*l)])
        end_r = np.array([end_norm[0] + (ray_dir[0]*-l), end_norm[1] + (ray_dir[1]*-l), end_norm[2] + (ray_dir[2]*-l)])
        p.addUserDebugLine(ee_pos, end_norm, [1, 0 ,0])
        p.addUserDebugLine(end_norm, end_l, [0, 1, 0]) # left is green
        p.addUserDebugLine(end_norm, end_r, [0, 0, 1]) # right is blue
        self.head_coords = [end_l, end_norm, end_r] 
        self.head_angle = angle

    def get_head_angle_range(self, end_effector, l=0.25):
        end_l, end_norm, end_r = self.head_coords

        hand_pos, _ = self.get_ee_pos_orient(end_effector)
        hand = hand_pos[0]
        # center = end_norm[0]

        if hand > end_r[0] and hand < end_l[0]:
            return 0
            # return abs(0.5 * (hand - center)) # TRY LATER: add a slight bias toward the center
        return min(hand - end_r[0], hand - end_l[0])

    
    def rotate_3d(self, point, axis, angle_degrees):
        # Convert the angle to radians
        angle_rad = np.radians(angle_degrees)
        
        # Normalize the rotation axis
        axis = axis / np.linalg.norm(axis)
        
        # Create the rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])
        
        return np.dot(rotation_matrix, point)

    def get_orthoganol(self, vec):

        return [-vec[1], vec[0], 0]

    def cal_chain_manipulibility(self, joint_angles, ee: str):
        chain = self.chain[ee]
        J = chain.jacobian(joint_angles, end_only=True)
        J = np.array(J)
        # print("J: ", J.shape)
        # J = J[:, 6:]
        m = np.linalg.det(np.dot(J, J.T))
        return m

    # might need to remove this
    def cal_manipulibility(self, joint_angles, ee_pos_arr, manipulibity_ee_names=None):
        J_arr = []
        m_arr = []  # manipulibility
        ee_idxes = self._get_end_effector_indexes(manipulibity_ee_names)

        for i in range(0, len(ee_idxes)):
            ee = ee_idxes[i]
            ee_pos = ee_pos_arr[i]
            # print("ee_pos: ", ee_pos)
            J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=ee_pos,
                                                      objPositions=joint_angles, objVelocities=joint_velocities,
                                                      objAccelerations=joint_accelerations, physicsClientId=self.id)
            # print("J linear: ", J_linear)
            J_linear = np.array(J_linear)
            J_angular = np.array(J_angular)  # TODO: check if we only need part of it (right now it is 3* 75)
            J = np.concatenate([J_linear, J_angular], axis=0)
            m = np.sqrt(np.linalg.det(J @ J.T))
            J_arr.append(J)
            m_arr.append(m)
            LOG.debug(f"End effector idx: {ee}, Jacobian_l: {J_linear.shape}, Jacobian_r: {J_angular.shape}, Manipulibility: {m}")
        avg_manipubility = np.mean(m_arr)

        return avg_manipubility

    def check_self_collision(self, end_effector=None):
        """
        Check self collision
        :return: set of collision pairs
        """
        p.performCollisionDetection(physicsClientId=self.id)
        self_collision_pairs= check_collision(self.body, self.body)  # TODO: Check with initial collision
        if end_effector is None:
            for pair in self_collision_pairs:
                LOG.debug (f"Self collision: {pair}, {self.human_dict.limb_index_dict[pair[0]]},"
                              f" {self.human_dict.limb_index_dict[pair[1]]}")
            return self_collision_pairs
        else:
            link_indices = self.human_dict.get_real_link_indices(end_effector)
            self_collision_pairs = [pair for pair in self_collision_pairs if pair[0] in link_indices or pair[1] in link_indices]
            return self_collision_pairs

    # get link positions for all link in the chain/ all links in body
    def get_link_positions(self, center_of_mass= True, end_effector_name=None):
        link_positions = []
        if end_effector_name is None:
            for i in range(-1, p.getNumJoints(self.body)):  # include base
                pos, orient = self.get_pos_orient(i, center_of_mass=center_of_mass)
                link_positions.append(pos)
        # only for real link in chain
        else:
            for i in self.human_dict.get_real_link_indices(end_effector_name):
                pos, orient = self.get_pos_orient(i, center_of_mass=center_of_mass)
                link_positions.append(pos)

        return link_positions

    def inverse_dynamic(self, end_effector_name=None):
        # inverse dynamics will return the torque for base 7 DOF + all joints DOF (in our case of floating base + 23 joints, it will be 76)
        # we need pass all DOF positions to inverse dynamics method, then filter out the result
        # see: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/pd_controller_stable.py

        base_pos, base_orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        joint_positions = []
        joint_positions.extend(list(base_pos))
        joint_positions.extend(list(base_orn)) # DOF for base = 7

        default_velocity = 0.0
        joint_indices_map = {} # (joint_index, index)
        count = 6
        for j in self.all_joint_indices:
            if p.getJointInfo(self.body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                joint_state = p.getJointState(self.body, j)
                joint_positions.append(joint_state[0])
                count+=1
                joint_indices_map[j] = count
        # print("joint_positions: ", len(joint_positions))

        # need to pass flags=1 to overcome inverseDynamics error for floating base
        # see https://github.com/bulletphysics/bullet3/issues/3188
        torques = p.calculateInverseDynamics(self.body, objPositions=joint_positions,
                                             objVelocities=[default_velocity] * (len(joint_positions)),
                                             objAccelerations=[0] * (len(joint_positions)), physicsClientId=self.id, flags=1)
        if end_effector_name is None:
            torques = torques[7:]
        else:
            # filter out the torques for the end effector
            indices = []
            for j in self.controllable_joint_indices:
                indices.append(joint_indices_map[j])
            torques = [torques[i] for i in indices]
        # print("torques: ", len(torques))
        return torques

    def check_env_collision(self, body_ids, end_effector= None):
        """
        Check self collision
        :return: set of collision pairs
        """
        collision_pairs = set()
        p.performCollisionDetection(physicsClientId=self.id)
        # print ("env_objects: ", body_ids, [p.getBodyInfo(i, physicsClientId=self.id)[1].decode('UTF-8') for i in body_ids])
        if end_effector is None:
            for env_body in body_ids:
                collision_pairs.update(check_collision(self.body, env_body))
        else:
            joint_indices = self.human_dict.get_real_link_indices(end_effector)
            for env_body in body_ids:
                pairs = check_collision(self.body, env_body)
                for pair in pairs:
                    if  pair[0] in joint_indices or pair[1] in joint_indices:
                        collision_pairs.add( pair)

        return collision_pairs

    def ray_cast_perpendicular(self, end_effector: str, ray_length=0.17):
        ee_pos, ee_orient = self.get_ee_pos_orient(end_effector)

        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 1]
        # using midpoint as start pos so that the hand is not counted as a collision
        dist = -0.05 # how far from the hand should the ray start
        end = [ee_pos[0] + (ray_dir[0]*dist), ee_pos[1] + (ray_dir[1]*dist), ee_pos[2] + (ray_dir[2]*dist)]
        # ray start and end
        start_pos = [(ee_pos[0] + end[0])/2, (ee_pos[1] + end[1])/2, (ee_pos[2] + end[2])/2]
        to_pos = [start_pos[0] + (ray_dir[0]*-ray_length), start_pos[1] + (ray_dir[1]*-ray_length), start_pos[2] + (ray_dir[2]*-ray_length)]
        result = p.rayTest(start_pos, to_pos)

        # visualize the ray from 'from_pos' to 'to_pos'
        ray_id = p.addUserDebugLine(start_pos, to_pos, [0, 1, 0])  # the ray is green
        res_id = result[0][0]
        p.removeUserDebugItem(ray_id)  # remove the visualized ray

        return res_id > 0

    def ray_cast_parallel(self, end_effector: str, ray_length=0.5):
        ee_pos, ee_orient = self.get_ee_pos_orient(end_effector)

        rotation = np.array(p.getMatrixFromQuaternion(ee_orient))
        ray_dir = rotation.reshape(3, 3)[:, 2]
        # RAY 1
        # using midpoint as start pos so that the hand is not counted as a collision
        dist = -0.075 # how far from the hand should the ray start
        end = [ee_pos[0] + (ray_dir[0]*dist), ee_pos[1] + (ray_dir[1]*dist), ee_pos[2] + (ray_dir[2]*dist)]
        # ray start and end
        start_pos = [(ee_pos[0] + end[0])/2, (ee_pos[1] + end[1])/2, (ee_pos[2] + end[2])/2]
        to_pos = [start_pos[0] + (ray_dir[0]*-ray_length), start_pos[1] + (ray_dir[1]*-ray_length), start_pos[2] + (ray_dir[2]*-ray_length)]
        result = p.rayTest(start_pos, to_pos)

        # RAY 2
        dist = 0.075 # how far from the hand should the ray start
        end = [ee_pos[0] + (ray_dir[0]*dist), ee_pos[1] + (ray_dir[1]*dist), ee_pos[2] + (ray_dir[2]*dist)]
        # ray start and end
        start_pos_b = [(ee_pos[0] + end[0])/2, (ee_pos[1] + end[1])/2, (ee_pos[2] + end[2])/2]
        to_pos_b= [start_pos[0] + (ray_dir[0]*ray_length), start_pos[1] + (ray_dir[1]*ray_length), start_pos[2] + (ray_dir[2]*ray_length)]
        result_b = p.rayTest(start_pos_b, to_pos_b)

        # visualize the ray from 'from_pos' to 'to_pos'
        p.addUserDebugLine(start_pos, to_pos, [0, 0, 1])  # the ray is blue
        p.addUserDebugLine(start_pos_b, to_pos_b, [0, 0, 1])  # the ray is blue
        res_id = result[0][0]
        res_id_b = result_b[0][0]
        p.removeAllUserDebugItems() # remove all rays
        return res_id + res_id_b > -2

    def check_collision_radius(self, end_effector="right_hand", distance=0.05):
        human_dict = HumanUrdfDict()
        link = human_dict.get_dammy_joint_id(end_effector)
        out = p.getClosestPoints(self.body, self.body, distance, linkIndexA=link)
        print("out: ", out, "\n len(out) > 5: ", len(out) > 5)
        return len(out) > 0

    def get_ee_pos_orient(self, end_effector):
        ee_pos, ee_orient = p.getLinkState(self.body, self.human_dict.get_dammy_joint_id(end_effector),  computeForwardKinematics=True, physicsClientId=self.id)[0:2]
        return ee_pos, ee_orient

    def get_ee_bb_dimension(self, end_effector, draw_bb=False):
        """
        Return the AABB bounding box dimensions of the end effector
        :param end_effector:
        :return:
        """
        link_idx = self.human_dict.get_dammy_joint_id(end_effector)
        min_pos, max_pos = p.getAABB(self.body, link_idx, physicsClientId=self.id)
        # compute box lengths
        box_dims = [max_pos[i] - min_pos[i] for i in range(3)]
        if draw_bb: # fopr debugging
            # compute box position (which is the center of the AABB)
            box_pos, box_orient = p.getLinkState(self.body, link_idx, physicsClientId=self.id)[:2]

            # set the box halfExtents
            half_extends= [length/ 2 for length in box_dims]
            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extends)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=[1, 0, 0, 0.7], halfExtents=half_extends)

            # create a multi-body with baseMass=0 (making it static)
            box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id, basePosition=box_pos,
                                          baseOrientation=[0, 0, 0, 1], physicsClientId=self.id)

        return np.array(box_dims)

    def get_ee_collision_shape_pos_orient(self, end_effector, collision_shape_radius=0.05):
        """
        Return the position and orientation of the collision shape based on end effector position and orientation
        Note that for now, we try to move the collision shape to one side of the end effector, along the normal vector
        :param end_effector:
        :param collision_shape_radius:
        :return:
        """
        ee_pos, ee_orient = self.get_ee_pos_orient(end_effector)
        ee_norm_vec = self.get_ee_normal_vector(end_effector)
        pos_offset = ee_norm_vec * collision_shape_radius # create a displacement along the normal vector, and scale it by the radius
        return np.array(ee_pos) + pos_offset, ee_orient

    def get_ee_normal_vector(self, end_effector):
        """
        Return the normal vector of the end effector (normalized)
        :param end_effector:
        :return:
        """
        ee_pos, ee_orient = self.get_ee_pos_orient(end_effector)
        ee_rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orient)).reshape(3, 3)
        ee_norm_vec = -ee_rot_matrix[:, 1]  # perpendicular to the palm, pointing from palm outward
        return ee_norm_vec/np.linalg.norm(ee_norm_vec)

    def _print_joint_indices(self):
        """
        Getting the joint index for debugging purpose
        TODO: refactor for programmatically generate the joint index
        :return:
        """
        print(self._get_controllable_joints())
        human_dict = self.human_dict

        left_arms = human_dict.get_joint_ids("left_clavicle")
        for name in human_dict.joint_chain_dict["left_hand"]:
            left_arms.extend(human_dict.get_joint_ids(name))

        right_arms = human_dict.get_joint_ids("right_clavicle")
        for name in human_dict.joint_chain_dict["right_hand"]:
            right_arms.extend(human_dict.get_joint_ids(name))

        left_legs = []
        for name in human_dict.joint_chain_dict["left_foot"]:
            left_legs.extend(human_dict.get_joint_ids(name))

        right_legs = []
        for name in human_dict.joint_chain_dict["right_foot"]:
            right_legs.extend(human_dict.get_joint_ids(name))

        print("left arms: ", left_arms)
        print("right arms: ", right_arms)
        print("left legs: ", left_legs)
        print("right legs: ", right_legs)


if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    # plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0, 0, 0])

    # bed
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
    p.setGravity(0, 0, -9.81)

    # bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    SMPL_PATH = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    smpl_data = load_smpl(SMPL_PATH)
    human.set_joint_angles_with_smpl(smpl_data)
    height, base_height = human.get_heights()

    human.set_global_orientation(smpl_data, [0, 0, base_height])

    # human.set_on_ground(bed_height)
    # human.generate_human_mesh(human.body, physic_client_id, SMPL_PATH)
    # show_human_mesh(SMPL_PATH)
    # human._print_joint_indices()
    human.fit_joint_angle(smpl_data.body_pose)
    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0, 0, 1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

    # Disconnect from the simulation
    p.disconnect()
