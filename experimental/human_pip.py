import colorsys
import os
import pickle

import numpy as np
import pybullet as p
import pybullet_data
import smplx
import trimesh
from gym.utils import seeding
from smplx import lbs

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict

from assistive_gym.envs.smpl.serialization import load_model
from assistive_gym.envs.utils.smpl_geom import generate_geom
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler, load_smpl, reposition_body_part


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
        euler = convert_aa_to_euler(pose[self.smpl_dict.get_pose_ids("pelvis")])
        # euler = euler_convert_np(euler, from_seq='XYZ', to_seq='ZYX')
        quat = np.array(p.getQuaternionFromEuler(np.array(euler), physic_client_idId=human_id))
        p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)

    def set_joint_angle(self, human_id, pose, smpl_joint_name, robot_joint_name):
        smpl_angles = convert_aa_to_euler(pose[self.smpl_dict.get_pose_ids(smpl_joint_name)])

        # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
        robot_joints = self.human_pip_dict.get_joint_ids(robot_joint_name)
        for i in range(0, 3):
            p.resetJointState(human_id, robot_joints[i], smpl_angles[i])

    def generate_human_mesh(self, id, physic_id, model_path):
        hull_dict, joint_pos_dict, joint_offset_dict = generate_geom(model_path)
        # now trying to scale the urdf file
        reposition_body_part(id, physic_id, joint_pos_dict)

        # p.loadURDF("test_mesh.urdf", [0, 0, 0])

    def position_robot_toc(self, task, arms, start_pos_orient, target_pos_orients, human, base_euler_orient=np.zeros(3),
                           max_ik_iterations=200, max_ik_random_restarts=1, randomize_limits=False, attempts=100,
                           jlwki_restarts=1, step_sim=False, check_env_collisions=False, right_side=True,
                           random_rotation=30, random_position=0.5):
        # Continually randomize the robot base position and orientation
        # Select best base pose according to number of goals reached and manipulability
        if type(arms) == str:
            arms = [arms]
            start_pos_orient = [start_pos_orient]
            target_pos_orients = [target_pos_orients]
        a = 6  # Order of the robot space. 6D (3D position, 3D orientation)
        best_position = None
        best_orientation = None
        best_num_goals_reached = None
        best_manipulability = None
        best_start_joint_poses = [None] * len(arms)
        iteration = 0
        # Save human joint states for later restoring
        human_angles = human.get_joint_angles(human.controllable_joint_indices)
        while iteration < attempts or best_position is None:
            iteration += 1
            # Randomize base position and orientation
            random_pos = np.array(
                [self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position),
                 self.np_random.uniform(-random_position, random_position), 0])
            random_orientation = self.get_quaternion([base_euler_orient[0], base_euler_orient[1],
                                                      base_euler_orient[2] + np.deg2rad(
                                                          self.np_random.uniform(-random_rotation, random_rotation))])
            self.set_base_pos_orient(np.array([-0.85, -0.4, 0]) + self.toc_base_pos_offset[task] + random_pos,
                                     random_orientation)
            # Reset all robot joints to their defaults
            self.reset_joints()
            # Reset human joints in case they got perturbed by previous iterations
            human.set_joint_angles(human.controllable_joint_indices, human_angles)
            num_goals_reached = 0
            manipulability = 0.0
            start_joint_poses = [None] * len(arms)
            # Check if the robot can reach all target locations from this base pose
            for i, arm in enumerate(arms):
                right = (arm == 'right')
                ee = self.right_end_effector if right else self.left_end_effector
                ik_indices = self.right_arm_ik_indices if right else self.left_arm_ik_indices
                lower_limits = self.right_arm_lower_limits if right else self.left_arm_lower_limits
                upper_limits = self.right_arm_upper_limits if right else self.left_arm_upper_limits
                for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
                    best_jlwki = None
                    best_joint_positions = None
                    for k in range(jlwki_restarts):
                        # Reset state in case anything was perturbed from the last iteration
                        human.set_joint_angles(human.controllable_joint_indices, human_angles)
                        # Find IK solution
                        success, joint_positions_q_star = self.ik_random_restarts(right, target_pos, target_orient,
                                                                                  max_iterations=max_ik_iterations,
                                                                                  max_ik_random_restarts=max_ik_random_restarts,
                                                                                  success_threshold=0.03,
                                                                                  step_sim=step_sim,
                                                                                  check_env_collisions=check_env_collisions,
                                                                                  randomize_limits=randomize_limits)
                        if not success:
                            continue
                        _, motor_positions, _, _ = self.get_motor_joint_states()
                        joint_velocities = [0.0] * len(motor_positions)
                        joint_accelerations = [0.0] * len(motor_positions)
                        center_of_mass = \
                        p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True,
                                       physicsClientId=self.id)[2]
                        J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass,
                                                                  objPositions=motor_positions,
                                                                  objVelocities=joint_velocities,
                                                                  objAccelerations=joint_accelerations,
                                                                  physicsClientId=self.id)
                        J_linear = np.array(J_linear)[:, ik_indices]
                        J_angular = np.array(J_angular)[:, ik_indices]
                        J = np.concatenate([J_linear, J_angular], axis=0)
                        # Joint-limited-weighting
                        joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits,
                                                                          upper_limits)
                        # Joint-limited-weighted kinematic isotropy (JLWKI)
                        det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
                        jlwki = np.power(det, 1.0 / a) / (
                                    np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T)) / a)
                        if best_jlwki is None or jlwki > best_jlwki:
                            best_jlwki = jlwki
                            best_joint_positions = joint_positions_q_star
                    if best_jlwki is not None:
                        num_goals_reached += 1
                        manipulability += best_jlwki
                        if j == 0:
                            start_joint_poses[i] = best_joint_positions
                    if j < len(start_pos_orient[i]) and best_jlwki is None:
                        # Not able to find an IK solution to a start goal. We cannot use this base pose
                        num_goals_reached = -1
                        manipulability = None
                        break
                if num_goals_reached == -1:
                    break

            if num_goals_reached > 0:
                if best_position is None or num_goals_reached > best_num_goals_reached or (
                        num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
                    best_position = random_pos
                    best_orientation = random_orientation
                    best_num_goals_reached = num_goals_reached
                    best_manipulability = manipulability
                    best_start_joint_poses = start_joint_poses

            human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Reset state in case anything was perturbed
        human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Set the robot base position/orientation and joint angles based on the best pose found
        p.resetBasePositionAndOrientation(self.body, np.array([-0.85, -0.4, 0]) + np.array(
            self.toc_base_pos_offset[task]) + best_position, best_orientation, physicsClientId=self.id)
        for i, arm in enumerate(arms):
            self.set_joint_angles(self.right_arm_joint_indices if arm == 'right' else self.left_arm_joint_indices,
                                  best_start_joint_poses[i])
        return best_position, best_orientation, best_start_joint_poses

    def ik_random_restarts(self, right, target_pos, target_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.03, step_sim=False, check_env_collisions=False, randomize_limits=True, collision_objects=[]):
        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)
        orient_orig = target_orient
        best_ik_angles = None
        best_ik_distance = 0
        for r in range(max_ik_random_restarts):
            target_joint_angles = self.ik(self.right_end_effector if right else self.left_end_effector, target_pos, target_orient, ik_indices=self.right_arm_ik_indices if right else self.left_arm_ik_indices, max_iterations=max_iterations, half_range=self.half_range, randomize_limits=(randomize_limits and r >= 10))
            self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
            gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
            if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
                # if step_sim:
                #     # TODO: Replace this with getClosestPoints, see: https://github.gatech.edu/zerickson3/assistive-gym/blob/vr3/assistive_gym/envs/feeding.py#L156
                #     for _ in range(5):
                #         p.stepSimulation(physicsClientId=self.id)
                #     # if len(p.getContactPoints(bodyA=self.body, bodyB=self.body, physicsClientId=self.id)) > 0 and orient_orig is not None:
                #     #     # The robot's arm is in contact with itself. Continually randomize end effector orientation until a solution is found
                #     #     target_orient = self.get_quaternion(self.get_euler(orient_orig) + np.deg2rad(self.np_random.uniform(-45, 45, size=3)))
                # if check_env_collisions:
                #     for _ in range(25):
                #         p.stepSimulation(physicsClientId=self.id)

                # Check if the robot is colliding with objects in the environment. If so, then continue sampling.
                if len(collision_objects) > 0:
                    dists_list = []
                    for obj in collision_objects:
                        dists_list.append(self.get_closest_points(obj, distance=0)[-1])
                    if not all(not d for d in dists_list):
                        continue
                gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
                if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
                    self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
                    return True, np.array(target_joint_angles)
            if best_ik_angles is None or np.linalg.norm(target_pos - np.array(gripper_pos)) < best_ik_distance:
                best_ik_angles = target_joint_angles
                best_ik_distance = np.linalg.norm(target_pos - np.array(gripper_pos))
        self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, best_ik_angles)
        return False, np.array(best_ik_angles)

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
