import time
from datetime import date, datetime
import numpy as np
import pickle as pk
import pybullet as p
import cv2
import os
import torch

from gym.utils import seeding
from assistive_gym.envs.agents.stretch_dex import StretchDex
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.utils.human_utils import set_self_collisions, disable_self_collisions
from assistive_gym.envs.utils.urdf_utils import load_smpl , convert_aa_to_euler_quat, SMPLData
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.agents.human_mesh import HumanMesh
from experimental.human_urdf import HumanUrdf



class SeatedPoseEnv(AssistiveEnv):
    def __init__(self, mesh=False, robot=True):
        self.robot = None
        if robot: self.robot = StretchDex('wheel_right')
        if mesh: self.human = HumanMesh()
        else: self.human = HumanUrdf()

        super(SeatedPoseEnv, self).__init__(robot=self.robot, human=self.human, task='', obs_robot_len=0, 
                                         obs_human_len=len(self.human.controllable_joint_indices), render=False) #hardcoded
        self.target_pos = np.array([0, 0, 0])
        self.smpl_file = None
        self.task = None # task = 'comfort_standing_up', 'comfort_taking_medicine',  'comfort_drinking'

    def get_comfort_score(self):
        return np.random.rand() #TODO: implement this
    # TODO: refactor train to move the score return to env.step

    def set_smpl_file(self, smpl_file):
        self.smpl_file = smpl_file

    def set_human_urdf(self, urdf_path):
        self.human.set_urdf_path(urdf_path)

    def set_task(self, task):
        if not task: # default task
            task = "comfort_taking_medicine"
        self.task = task  # task = 'comfort_standing_up', 'comfort_taking_medicine',  'comfort_drinking'

    def step(self, action):
        if self.human.controllable:
            # print("action", action)
            action = np.concatenate([action['robot'], action['human']])

        self.take_step(action)

        obs = self._get_obs()
        return None

    def _get_obs(self, agent=None): # not needed
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos) 

        robot_obs = np.array([]) # TODO: fix
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs =np.array(human_joint_angles)
            if agent == 'human':
                return human_obs

            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def get_all_human_angles(self):
        dic = HumanUrdfDict()
        dict = dic.joint_xyz_dict
        inds = []
        for j in dict:
            inds.append(dict[j])
        angles = self.human.get_joint_angles(inds)
        return angles

    def config_mesh_angles(self, angles):
        dic = HumanUrdfDict()
        dict = dic.joint_xyz_dict
        ind = {}
        i = 0
        for joint in dict: 
            ind[joint] = angles[i]
            i += 1

        h = self.human
        angles_matched = [(h.j_left_hip_x, ind["left_hip_x"]), (h.j_left_hip_y, ind["left_hip_y"]), (h.j_left_hip_z, ind["left_hip_z"]), 
                  (h.j_right_hip_x, ind["right_hip_x"]), (h.j_right_hip_y, ind["right_hip_y"]), (h.j_right_hip_z, ind["right_hip_z"]), 
                  (h.j_left_knee_x, ind["left_knee_x"]), (h.j_left_knee_y, ind["left_knee_y"]), (h.j_left_knee_z, ind["left_knee_z"]),
                  (h.j_right_knee_x, ind["right_knee_x"]), (h.j_right_knee_y, ind["right_knee_y"]), (h.j_right_knee_z, ind["right_knee_z"]), 
                  (h.j_left_ankle_x, ind["left_ankle_x"]), (h.j_left_ankle_y, ind["left_ankle_y"]), (h.j_left_ankle_z, ind["left_ankle_z"]),
                  (h.j_right_ankle_x, ind["right_ankle_x"]), (h.j_right_ankle_z, ind["right_ankle_z"]), (h.j_right_ankle_z, ind["right_ankle_z"]), 
                  (h.j_left_shoulder_x, ind["left_shoulder_x"]), (h.j_left_shoulder_y, ind["left_shoulder_y"]), (h.j_left_shoulder_z, ind["left_shoulder_z"]), 
                  (h.j_right_shoulder_x, ind["right_shoulder_x"]), (h.j_right_shoulder_y, ind["right_shoulder_y"]), (h.j_right_shoulder_z, ind["right_shoulder_z"]), 
                  (h.j_left_elbow_x, ind["left_elbow_x"]), (h.j_left_elbow_y, ind["left_elbow_y"]), (h.j_left_elbow_z, ind["left_elbow_z"]), 
                  (h.j_right_elbow_x, ind["right_elbow_x"]), (h.j_right_elbow_y, ind["right_elbow_y"]), (h.j_right_elbow_z, ind["right_elbow_z"]), 
                  (h.j_left_wrist_x, ind["left_lowarm_x"]), (h.j_left_wrist_y, ind["left_lowarm_y"]), (h.j_left_wrist_z, ind["left_lowarm_z"]), 
                  (h.j_right_wrist_x, ind["right_lowarm_x"]), (h.j_right_wrist_y, ind["right_lowarm_y"]),(h.j_right_wrist_z, ind["right_lowarm_z"]),
                  (h.j_upper_neck_x, ind["head_x"]), (h.j_upper_neck_y, ind["head_y"]), (h.j_upper_neck_z, ind["head_z"]),
                  (h.j_lower_neck_x, ind["neck_x"]), (h.j_lower_neck_y, ind["neck_y"]), (h.j_lower_neck_z, ind["neck_z"]), 
                  (h.j_left_pecs_x, ind["left_clavicle_x"]), (h.j_left_pecs_y, ind["left_clavicle_y"]), (h.j_left_pecs_z, ind["left_clavicle_z"]), 
                  (h.j_right_pecs_x, ind["right_clavicle_x"]), (h.j_right_pecs_y, ind["right_clavicle_y"]), (h.j_right_pecs_z, ind["right_clavicle_z"]), 
                  (h.j_left_toes_x, ind["left_foot_x"]), (h.j_left_toes_y, ind["left_foot_y"]), (h.j_left_toes_z, ind["left_foot_z"]),
                  (h.j_right_toes_x, ind["right_foot_x"]), (h.j_right_toes_y, ind["right_foot_y"]),(h.j_right_toes_z, ind["right_foot_z"]), 
                  (h.j_waist_x, ind["spine_2_x"]), (h.j_waist_y, ind["spine_2_y"]), (h.j_waist_z, ind["spine_2_z"]), 
                  (h.j_chest_x, ind["spine_3_x"]), (h.j_chest_y, ind["spine_3_y"]), (h.j_chest_z, ind["spine_3_z"]), 
                  (h.j_upper_chest_x, ind["spine_4_x"]), (h.j_upper_chest_y, ind["spine_4_y"]), (h.j_upper_chest_z, ind["spine_4_z"])]
        # NEED head, neck relationship and spine 
        '''
        My guess -> upper neck = head, lower neck = neck, clavicle = pec, upper_chest, chest (or Upper neck!) = spine (2, 3, 4) 
        '''
        # [h.j_left_hip_x, h.j_left_hip_y, h.j_left_hip_z, h.j_left_knee_x, h.j_left_knee_y, h.j_left_knee_z,
        #  h.j_left_ankle_x, h.j_left_ankle_y, h.j_left_ankle_z, h.j_right_hip_x, h.j_right_hip_y, h.j_right_hip_z, 
        #  h.j_right_knee_x, h.j_right_knee_y, h.j_right_knee_z, h.j_right_ankle_x, h.j_right_ankle_y, h.j_right_ankle_z, 
        #  ]

        return angles_matched
    
    def get_mesh_angle(self, angles, jnts: list):
        dic = HumanUrdfDict()
        dict = dic.joint_xyz_dict
        ind = {}
        i = 0
        for joint in dict: 
            ind[joint] = angles[i]
            i += 1
        ref = []
        for j in jnts: ref.append(ind[j])
        return ref
           
    def reset_human(self, is_collision):
        if not is_collision:
            self.human.set_joint_angles_with_smpl(load_smpl(self.smpl_file)) #TODO: fix
        else:
            bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)
            smpl_data = load_smpl(self.smpl_file)
            self.human.set_joint_angles_with_smpl(smpl_data)
            height, base_height = self.human.get_heights()
            # print("human height ", height, base_height, "bed height ", bed_height, bed_base_height)
            self.human.set_global_orientation(smpl_data, [0, 0, bed_height])
            self.human.set_gravity(0, 0, -9.81)
            p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.id)

    def assign_smplx_features(self, smpl_data: SMPLData):
        smpl_info = pk.load(open(smpl_data.smpl_file, 'rb'))
    
        r_hand = smpl_info['right_hand_pose'][0][0:6]
        l_hand = smpl_info['left_hand_pose'][0][0:6]

        smpl_data.r_hand_pose = torch.tensor(r_hand)[None, :]
        smpl_data.l_hand_pose = torch.tensor(l_hand)[None, :]
        smpl_data.r_eye_pose = torch.tensor(smpl_info['reye_pose'][0])
        smpl_data.l_eye_pose = torch.tensor(smpl_info['leye_pose'][0])
        smpl_data.jaw_pose = torch.tensor(smpl_info['jaw_pose'][0])
        smpl_data.expression = torch.tensor(smpl_info['expression'][0])[None, :]
        return smpl_data
    
    def write_human_pkl(self, smpl_data):
        # get template smpl pkl file
        dir = os.getcwd()
        fpath = dir + '/examples/data/saved_seated_poses/template.pkl'
        template = pk.load(open(fpath, 'rb'))
        
        # reassign smpl parameters in template file
        template['global_orient'] = np.array(smpl_data.global_orient)
        template['betas'] = np.array(smpl_data.betas)

        # index the urdf joints in the correct order and overwrite template
        humanDict = HumanUrdfDict()
        joints = ['left_hip', 'right_hip', 'spine_2', 'left_knee', 'right_knee', 'spine_3', 'left_ankle',
                   'right_ankle', 'spine_4', 'left_foot', 'right_foot', 'neck', 'left_clavicle', 'right_clavicle', 'head', 
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_lowarm', 'right_lowarm', 'left_hand',
                   'right_hand']
        ordered_inds = []
        for joint in joints: 
                i = humanDict.joint_dict[joint]
                ordered_inds.append(i)
                ordered_inds.append(i + 1)
                ordered_inds.append(i + 2)
        angles = self.human.get_joint_angles(ordered_inds)
        all_angles = angles
        template['body_pose'] = torch.FloatTensor(all_angles)
        
        # write new file
        new_fn = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime("%H%M") + ".pkl"
        new_fpath = dir + '/examples/data/saved_seated_poses/' + new_fn
        pk.dump(template, open(new_fpath, 'ab'))
        print("human saved")
 
    def rotate_test(self, smpl_data):
        # ROTATE TEST LOOP
        pos = list(smpl_data.transl)
        pos[2] += 0.1

        p.addUserDebugText("rotate z", [0, 0, 1.5], [1, 0, 0]) # red text
        rots = [np.pi/8, np.pi/6, np.pi/4, np.pi/2, np.pi]
        for a in rots:
            ori = self.axis_rotate(smpl_data.global_orient, -a, axis='z')
            self.human.set_base_pos_orient(pos, ori)
            time.sleep(2)
        p.removeAllUserDebugItems()
            # reset
        self.human.set_base_pos_orient(pos, smpl_data.global_orient)
            # y rot
        p.addUserDebugText("rotate y", [0, 0, 1.5], [1, 0, 0]) # red text
        rots = [np.pi/8, np.pi/6, np.pi/4, np.pi/2, np.pi]
        for a in rots:
            ori = self.axis_rotate(smpl_data.global_orient, -a, axis='y')
            self.human.set_base_pos_orient(pos, ori)
            time.sleep(2)
        p.removeAllUserDebugItems()
            # reset
        self.human.set_base_pos_orient(pos, smpl_data.global_orient)
            # x rot
        p.addUserDebugText("rotate x", [0, 0, 1.5], [1, 0, 0]) # red text
        rots = [np.pi/8, np.pi/6, np.pi/4, np.pi/2, np.pi]
        for a in rots:
            ori = self.axis_rotate(smpl_data.global_orient, -a, axis='x')
            self.human.set_base_pos_orient(pos, ori)
            time.sleep(2)
        p.removeAllUserDebugItems()

        self.human.set_base_pos_orient(pos, smpl_data.global_orient)
        time.sleep(5)
        # END LOOP
    
    def reset(self):
        super(SeatedPoseEnv, self).reset()

        # magic happen here - now call agent.init()
        self.build_assistive_env('diningchair')

        bed_height, _ = self.furniture.get_heights(set_on_ground=True)

        # disable self collision before dropping on bed
        num_joints = p.getNumJoints(self.human.body, physicsClientId=self.id)
        disable_self_collisions(self.human.body, num_joints, self.id)
        smpl_data = load_smpl(self.smpl_file)
        self.human.set_joint_angles_with_smpl(smpl_data, False)
        new_x_or = self.human.align_chair()
        print("rotating: ", new_x_or)
        smpl_data.global_orient[0][0] += new_x_or
        smpl_data.global_orient = smpl_data.global_orient[0]

        self.human.set_global_orientation(smpl_data, [0, -0.05,  bed_height+0.2])

        self.robot.set_gravity(0, 0, -9.81)
        self.human.set_gravity(0, 0, -9.81)

        self.robot.set_joint_angles([4], [0.5]) # for stretch_dex: move the gripper upward

        # init tool
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True,
                       mesh_scale=[0.045] * 3, alpha=0.75)
        # # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task],
                                             set_instantly=True)


        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)
        p.setTimeStep(1/240., physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # drop human on bed
        for i in range(1000):
            p.stepSimulation(physicsClientId=self.id)

        # # enable self collision and reset joint angle after dropping on bed
        # human_pos = p.getBasePositionAndOrientation(self.human.body, physicsClientId=self.id)[0]

        # # REMOVED: the resetting of global orientation - it may be messing a bit with the chair interaction
        # self.human.set_global_orientation(smpl_data, human_pos)
        self.human.set_joint_angles_with_smpl(smpl_data, False)

        set_self_collisions(self.human.body, self.id)
        self.human.initial_self_collisions= self.human.check_self_collision()

        self.init_env_variables()
        smpl_data.transl = self.human.get_ee_pos_orient("spine_4")[0]
        angles = self.get_all_human_angles()
        p.addUserDebugText("final pose", [0, 0, 1.5], [1, 0, 0]) # red text
        self.write_human_pkl(smpl_data)
        time.sleep(5)
        p.removeAllUserDebugItems()
        smpl_data.smpl_file = self.smpl_file
        smpl_data = self.assign_smplx_features(smpl_data)
        return smpl_data, angles
    
    def reset_mesh(self, smpl_data: SMPLData, angles): 
        self.setup_camera([0, -1.65, 1.25], [0, 0, 0.75])
        super(SeatedPoseEnv, self).reset()
        angs = self.config_mesh_angles(angles)

        # magic happen here - now call agent.init()
        self.build_assistive_env('diningchair', human_angles=angs, smpl_data=smpl_data)

        # RESET: human
        pos = list(smpl_data.transl)
        pos[1] += 0.1

        orient = [0.0, 0.0, 0.0]
        orient[0] -= np.pi/8
        orient = convert_aa_to_euler_quat(orient)[0]
        self.human.set_base_pos_orient(pos, orient)

        # STEP: simulation
        p.setTimeStep(1/240., physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        
        # RUN: sim for viewing etc.
        while True:
            p.stepSimulation(physicsClientId=self.id)
            keys = p.getKeyboardEvents()
            if ord('q') in keys:
                break
        img, depth = self.get_camera_image_depth()
        depth = np.array(depth)
        d_shifted = depth - np.min(depth)
        d_scaled = (d_shifted / np.max(d_shifted)) * 255

        print("image taken")
        fn = date.today().strftime("%b-%d-%Y") + "_" + datetime.now().strftime("%H%M")
        rgb_fn = "images/sim_results/" + fn + "_rgb.png"
        depth_fn = "images/sim_results/" + fn + "_depth.png"
        cv2.imwrite(rgb_fn, img)
        cv2.imwrite(depth_fn, d_scaled)
        print("images_saved")
        return


