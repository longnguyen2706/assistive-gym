import time
from datetime import date, datetime
import numpy as np
import pickle as pk
import pybullet as p
import cv2
import os
import torch
import random
import math

# Angle Conversion
from scipy.spatial.transform import Rotation as R

from gym.utils import seeding
from assistive_gym.envs.agents.stretch_dex import StretchDex
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.utils.human_utils import set_self_collisions, disable_self_collisions
from assistive_gym.envs.utils.urdf_utils import load_smpl , convert_aa_to_euler_quat, get_aa_from_euler, euler_convert_np, batch_rot2aa, SMPLData
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.agents.human_mesh import HumanMesh
from experimental.human_urdf import HumanUrdf

GENDER_DATAPATH = 'examples/data/slp3d/gender.txt'
RANDOMIZE_CAMERA = True

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

    def config_mesh_angles(self, angles, transform=True):
        dic = HumanUrdfDict()
        dict = dic.joint_xyz_lists
        ind = {}
        i = 0
        for joint in dict:
            j = dict[joint] # the x, y, z coords for the joint angles
            if joint in ["left_elbow", "right_elbow"]: 
                euler = angles[i:i+3]
                r = R.from_euler("XYZ", euler, degrees=False)
                r = torch.tensor(r.as_matrix())[None, :]
                ind[joint] = batch_rot2aa(r)[0]
                i += 3
            elif joint == "pelvis": 
                ind[joint] = angles[i] 
                i += 1
            else:
                ind[joint] = angles[i:i+3]
                i += 3
        h = self.human
        angles_matched = [(h.j_left_hip_x, ind["left_hip"][0]), (h.j_left_hip_y, ind["left_hip"][1]), (h.j_left_hip_z, ind["left_hip"][2]), 
                  (h.j_right_hip_x, ind["right_hip"][0]), (h.j_right_hip_y, ind["right_hip"][1]), (h.j_right_hip_z, ind["right_hip"][2]), 
                  (h.j_left_knee_x, ind["left_knee"][0]), (h.j_left_knee_y, ind["left_knee"][1]), (h.j_left_knee_z, ind["left_knee"][2]),
                  (h.j_right_knee_x, ind["right_knee"][0]), (h.j_right_knee_y, ind["right_knee"][1]), (h.j_right_knee_z, ind["right_knee"][2]), 
                  (h.j_left_ankle_x, ind["left_ankle"][0]), (h.j_left_ankle_y, ind["left_ankle"][1]), (h.j_left_ankle_z, ind["left_ankle"][2]),
                  (h.j_right_ankle_x, ind["right_ankle"][0]), (h.j_right_ankle_z, ind["right_ankle"][1]), (h.j_right_ankle_z, ind["right_ankle"][2]), 
                  (h.j_left_shoulder_x, ind["left_shoulder"][0]), (h.j_left_shoulder_y, ind["left_shoulder"][1]), (h.j_left_shoulder_z, ind["left_shoulder"][2]), 
                  (h.j_right_shoulder_x, ind["right_shoulder"][0]), (h.j_right_shoulder_y, ind["right_shoulder"][1]), (h.j_right_shoulder_z, ind["right_shoulder"][2]), 
                  (h.j_left_elbow_x, ind["left_elbow"][0]), (h.j_left_elbow_y, ind["left_elbow"][1]), (h.j_left_elbow_z, ind["left_elbow"][2]), 
                  (h.j_right_elbow_x, ind["right_elbow"][0]), (h.j_right_elbow_y, ind["right_elbow"][1]), (h.j_right_elbow_z, ind["right_elbow"][2]),                 
                  (h.j_left_wrist_x, ind["left_lowarm"][0]), (h.j_left_wrist_y, ind["left_lowarm"][1]), (h.j_left_wrist_z, ind["left_lowarm"][2]), 
                  (h.j_right_wrist_x, ind["right_lowarm"][0]), (h.j_right_wrist_y, ind["right_lowarm"][1]),(h.j_right_wrist_z, ind["right_lowarm"][2]),
                  (h.j_upper_neck_x, ind["head"][0]), (h.j_upper_neck_y, ind["head"][1]), (h.j_upper_neck_z, ind["head"][2]),
                  (h.j_lower_neck_x, ind["neck"][0]), (h.j_lower_neck_y, ind["neck"][1]), (h.j_lower_neck_z, ind["neck"][2]), 
                  (h.j_left_pecs_x, ind["left_clavicle"][0]), (h.j_left_pecs_y, ind["left_clavicle"][1]), (h.j_left_pecs_z, ind["left_clavicle"][2]), 
                  (h.j_right_pecs_x, ind["right_clavicle"][0]), (h.j_right_pecs_y, ind["right_clavicle"][1]), (h.j_right_pecs_z, ind["right_clavicle"][2]), 
                  (h.j_left_toes_x, ind["left_foot"][0]), (h.j_left_toes_y, ind["left_foot"][1]), (h.j_left_toes_z, ind["left_foot"][2]),
                  (h.j_right_toes_x, ind["right_foot"][0]), (h.j_right_toes_y, ind["right_foot"][1]),(h.j_right_toes_z, ind["right_foot"][2]), 
                  (h.j_waist_x, ind["spine_2"][0]), (h.j_waist_y, ind["spine_2"][1]), (h.j_waist_z, ind["spine_2"][2]), 
                  (h.j_chest_x, ind["spine_3"][0]), (h.j_chest_y, ind["spine_3"][1]), (h.j_chest_z, ind["spine_3"][2]), 
                  (h.j_upper_chest_x, ind["spine_4"][0]), (h.j_upper_chest_y, ind["spine_4"][1]), (h.j_upper_chest_z, ind["spine_4"][2])]
        

        # ind["left_elbow_x"], ind['left_elbow_y'], ind['left_elbow_z'] = (0.11429025, -1.8572933, 0.34492198)
        # ind["right_elbow_x"], ind['right_elbow_y'], ind['right_elbow_z'] = (0.55436826, 1.83781826, -0.62086833)
        # angles_matched = [(h.j_left_hip_x, ind["left_hip_x"]), (h.j_left_hip_y, ind["left_hip_y"]), (h.j_left_hip_z, ind["left_hip_z"]), 
        #           (h.j_right_hip_x, ind["right_hip_x"]), (h.j_right_hip_y, ind["right_hip_y"]), (h.j_right_hip_z, ind["right_hip_z"]), 
        #           (h.j_left_knee_x, ind["left_knee_x"]), (h.j_left_knee_y, ind["left_knee_y"]), (h.j_left_knee_z, ind["left_knee_z"]),
        #           (h.j_right_knee_x, ind["right_knee_x"]), (h.j_right_knee_y, ind["right_knee_y"]), (h.j_right_knee_z, ind["right_knee_z"]), 
        #           (h.j_left_ankle_x, ind["left_ankle_x"]), (h.j_left_ankle_y, ind["left_ankle_y"]), (h.j_left_ankle_z, ind["left_ankle_z"]),
        #           (h.j_right_ankle_x, ind["right_ankle_x"]), (h.j_right_ankle_z, ind["right_ankle_z"]), (h.j_right_ankle_z, ind["right_ankle_z"]), 
        #           (h.j_left_shoulder_x, ind["left_shoulder_x"]), (h.j_left_shoulder_y, ind["left_shoulder_y"]), (h.j_left_shoulder_z, ind["left_shoulder_z"]), 
        #           (h.j_right_shoulder_x, ind["right_shoulder_x"]), (h.j_right_shoulder_y, ind["right_shoulder_y"]), (h.j_right_shoulder_z, ind["right_shoulder_z"]), 
        #           (h.j_left_elbow_x, ind["left_elbow_x"]), (h.j_left_elbow_y, ind["left_elbow_y"]), (h.j_left_elbow_z, ind["left_elbow_z"]), 
        #           (h.j_right_elbow_x, ind["right_elbow_x"]), (h.j_right_elbow_y, ind["right_elbow_y"]), (h.j_right_elbow_z, ind["right_elbow_z"]),                 
        #           (h.j_left_wrist_x, ind["left_lowarm_x"]), (h.j_left_wrist_y, ind["left_lowarm_y"]), (h.j_left_wrist_z, ind["left_lowarm_z"]), 
        #           (h.j_right_wrist_x, ind["right_lowarm_x"]), (h.j_right_wrist_y, ind["right_lowarm_y"]),(h.j_right_wrist_z, ind["right_lowarm_z"]),
        #           (h.j_upper_neck_x, ind["head_x"]), (h.j_upper_neck_y, ind["head_y"]), (h.j_upper_neck_z, ind["head_z"]),
        #           (h.j_lower_neck_x, ind["neck_x"]), (h.j_lower_neck_y, ind["neck_y"]), (h.j_lower_neck_z, ind["neck_z"]), 
        #           (h.j_left_pecs_x, ind["left_clavicle_x"]), (h.j_left_pecs_y, ind["left_clavicle_y"]), (h.j_left_pecs_z, ind["left_clavicle_z"]), 
        #           (h.j_right_pecs_x, ind["right_clavicle_x"]), (h.j_right_pecs_y, ind["right_clavicle_y"]), (h.j_right_pecs_z, ind["right_clavicle_z"]), 
        #           (h.j_left_toes_x, ind["left_foot_x"]), (h.j_left_toes_y, ind["left_foot_y"]), (h.j_left_toes_z, ind["left_foot_z"]),
        #           (h.j_right_toes_x, ind["right_foot_x"]), (h.j_right_toes_y, ind["right_foot_y"]),(h.j_right_toes_z, ind["right_foot_z"]), 
        #           (h.j_waist_x, ind["spine_2_x"]), (h.j_waist_y, ind["spine_2_y"]), (h.j_waist_z, ind["spine_2_z"]), 
        #           (h.j_chest_x, ind["spine_3_x"]), (h.j_chest_y, ind["spine_3_y"]), (h.j_chest_z, ind["spine_3_z"]), 
        #           (h.j_upper_chest_x, ind["spine_4_x"]), (h.j_upper_chest_y, ind["spine_4_y"]), (h.j_upper_chest_z, ind["spine_4_z"])]
        
        # print("left_elbow in matched: ", ind["left_elbow_x"], " ", ind["left_elbow_y"], " ", ind["left_elbow_z"])        

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

    def read_gender_data(self, filepath): # currently works with slp3d
        gender_data = {}
        with open(filepath, "r") as f:
            text = f.read()
            lines = text.strip().split("\n")


            for line in lines:
                sample, gender = line.split(": ")
                gender_data[int(sample)] = gender

            return gender_data
    
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
    
    def assign_smplx_gender(self, smpl_data:SMPLData):
        gender_data = self.read_gender_data(os.path.join(os.getcwd(), GENDER_DATAPATH))
        smpl_data.gender = gender_data[int(smpl_data.person_id[1:])]
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
 
    def write_train_data(self, smpl_data:SMPLData, angles):
        _, depth = self.get_camera_image_depth()
        depth = np.array(depth)
        d_shifted = depth - np.min(depth)
        # FOR VISUALIZATION: 
        d_scaled = (d_shifted / np.max(d_shifted)) * 255

        # FILE NAMES
        fp = "data_gen_results/" + smpl_data.person_id + "/"
        depth_fn = fp + smpl_data.pose_id + "_depth.pkl"
        label_fn = fp + "pose_labels.pkl"
        pic_fn = fp + smpl_data.pose_id + ".png"

        # FORMAT: label data
        label = []
        betas = list(smpl_data.betas)
        angles = list(angles)
        for i in range(len(betas)): label.append(betas[i])
        for j in range(len(angles)): label.append(angles[j])

        # OBTAIN: original data
        full_data = {}
        fr = open(label_fn, 'rb')
        original = pk.load(open(label_fn, 'rb'))
        fr.close()

        # APPEND: original and new data
        # print("not saving training data, check write_train_data in SeatedPoseEnv")
        if True:
            if original is None: full_data = {str(smpl_data.pose_id): label} # use the pose id as the data label
            else: 
                original[str(smpl_data.pose_id)] = label
                full_data = original

            # WRITE: data to files
            fw = open(label_fn, 'wb')
            pk.dump(depth, open(depth_fn, 'wb'))
            pk.dump(full_data, fw)
            fw.close()
            cv2.imwrite(pic_fn, d_scaled)
            
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
    
    def align_human(self, smpl_data):
        bed_height, _ = self.furniture.get_heights(set_on_ground=True)
        # SHIFT y-axis
        smpl_data.global_orient = [smpl_data.global_orient[0], 0.0, 0.0] # SET URDF orientation to 0 in yaw and roll
        self.human.set_global_orientation(smpl_data, [0, 0,  bed_height+0.2]) # SET the oritentation and base pos to 0 out y
        # SHIFT Pelvis
        p_y = self.human.get_ee_pos_orient("pelvis")[0][1]
        shift_y = -0.2 -p_y
        shift_dict = {'wheelchair2':-0.15, 'stool':0.15, 'stool2':0.05, 'couch':-0.45}
        if self.furniture.furniture_type in shift_dict.keys():
            shift_y += shift_dict[self.furniture.furniture_type]
        
        
        # ROTATE x-axis
        new_x_or = self.human.align_chair()
        print("rotating: ", new_x_or)
    
        if len(smpl_data.global_orient) == 1:
            smpl_data.global_orient[0][0] += new_x_or
            smpl_data.global_orient = smpl_data.global_orient[0]
        else: smpl_data.global_orient[0] += new_x_or

        x_pos = 0
        if self.furniture.furniture_type == 'couch': x_pos = (random.random() * 0.75) + 0.5 # range 0.5 to 1.25
        self.human.set_global_orientation(smpl_data, [x_pos, shift_y,  bed_height+0.2])
        return smpl_data
    
    def align_mesh(self, downward_shift=False, forward_shift=False):
        out = p.getContactPoints(self.human.body, self.furniture.body)
        if forward_shift:
            print("shifting mesh body out")
            totalShift = 0
            # using the left and right ankles to help define maximum shift
            left_ = self.human.get_pos_orient(7)[0][1]
            right = self.human.get_pos_orient(8)[0][1]
            while totalShift <= 0.2 and (left_ > -0.3 or right > -0.3):
                    pos, ori = p.getBasePositionAndOrientation(self.human.body)
                    copy_pos = (pos[0], pos[1] - 0.02, pos[2])
                    p.resetBasePositionAndOrientation(self.human.body, copy_pos, ori)
                    totalShift += 0.02
                    left_ = self.human.get_pos_orient(7)[0][1]
                    right = self.human.get_pos_orient(8)[0][1]
        else:
            if downward_shift:
                print("shifitng mesh body down")
                totalShift = 0
                while totalShift <= 0.1 and len(out) == 0:
                        pos, ori = p.getBasePositionAndOrientation(self.human.body)
                        copy_pos = (pos[0], pos[1], pos[2] - 0.01)
                        p.resetBasePositionAndOrientation(self.human.body, copy_pos, ori)
                        totalShift += 0.01
                        out = p.getContactPoints(self.human.body, self.furniture.body)
            else:
                if len(out) == 0: return
                dist = out[8]
                if dist < 0:
                    print("aligning mesh")
                    totalShift = 0
                    while totalShift <= 0.1 and dist < 0:
                        pos, ori = p.getBasePositionAndOrientation(self.human.body)
                        pos[2] += 0.01
                        p.resetBasePositionAndOrientation(self.human.body, pos, ori)
                        totalShift += 0.01
                        out = p.getContactPoints(self.human.body, self.furniture.body)
                        dist = out[8]
                
                out = p.getContactPoints(self.human.body, self.furniture.body)
                dist = out[8]
                
                if dist < 0:
                    pos, ori = p.getBasePositionAndOrientation(self.human.body)
                    pos[2] -= 0.05 # put it back halfway
                    p.resetBasePositionAndOrientation(self.human.body, pos, ori)
                    totalShift = 0
                    while totalShift <= 0.1 and dist < 0:
                        pos, ori = p.getBasePositionAndOrientation(self.human.body)
                        pos[1] -= 0.01
                        p.resetBasePositionAndOrientation(self.human.body, pos, ori)
                        totalShift += 0.01
                        out = p.getContactPoints(self.human.body, self.furniture.body)
                        dist = out[8]
                
                out = p.getContactPoints(self.human.body, self.furniture.body)
                dist = out[8]   
                if dist < 0:
                    pos, ori = p.getBasePositionAndOrientation(self.human.body)
                    pos[1] += 0.05 # put it back halfway
                    p.resetBasePositionAndOrientation(self.human.body, pos, ori)     
        
    def set_env_camera(self, randomize=RANDOMIZE_CAMERA):
        if randomize:
            distance_sq = 6.25 if self.furniture.furniture_type in ['stool', 'stool2', 'couch'] else 2.89
            if self.furniture.furniture_type == 'stool2':
                x_pos = (random.random() * 3) - 1.5
                y_pos = math.sqrt(distance_sq - (x_pos**2))
                self.setup_camera([x_pos, -y_pos, 1.75], [0, 0, 0.8], camera_height=1080, camera_width=1920)
            elif self.furniture.furniture_type == 'couch':
                x_pos = (random.random() * 3) - 1.5
                y_pos = math.sqrt(distance_sq - (x_pos**2))
                self.setup_camera([x_pos + 0.5, -y_pos, 1], [0.75, 0, 0.6], camera_height=1080, camera_width=1920)
            else: 
                x_pos = (random.random() * 3) - 1.5
                y_pos = math.sqrt(distance_sq - (x_pos**2))
                self.setup_camera([x_pos, -y_pos, 1.2], [0, 0, 0.65], camera_height=1080, camera_width=1920)
        else: 
            if self.furniture.furniture_type in ['stool']: self.setup_camera([0, -2, 1.5], [0, 0, 0.65], camera_height=1080, camera_width=1920)
            elif self.furniture.furniture_type in ['couch']: self.setup_camera([0.75, -2.5, 1], [0.75, 0, 0.6], camera_height=1080, camera_width=1920)  
            elif self.furniture.furniture_type == 'stool2': self.setup_camera([0, -2, 1.75], [0, 0, 0.8], camera_height=1080, camera_width=1920) 
            else: self.setup_camera([0, -1.7, 1.2], [0, 0, 0.65], camera_height=1080, camera_width=1920)
    
    def reset(self):
        super(SeatedPoseEnv, self).reset()

        # magic happen here - now call agent.init()
        # self.build_assistive_env('diningchair')
        self.build_assistive_env('couch')

        # disable self collision before dropping on bed
        num_joints = p.getNumJoints(self.human.body, physicsClientId=self.id)
        disable_self_collisions(self.human.body, num_joints, self.id)
        smpl_data = load_smpl(self.smpl_file)
        self.human.set_joint_angles_with_smpl(smpl_data, False)

        smpl_data = self.align_human(smpl_data)

        self.robot.set_gravity(0, 0, -9.81)
        self.human.set_gravity(0, 0, -9.81)

        self.robot.set_joint_angles([4], [0.5]) # for stretch_dex: move the gripper upward

    
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
        self.human.set_joint_angles_with_smpl(smpl_data, False) # DECIDE ABOUT THIS

        p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 10], [1, 1, 0]) # red + green

        set_self_collisions(self.human.body, self.id)
        self.human.initial_self_collisions= self.human.check_self_collision()

        self.init_env_variables()
        smpl_data.transl[2] = self.human.get_ee_pos_orient("spine_4")[0][2]
        smpl_data.transl[0] = self.human.get_ee_pos_orient("pelvis")[0][0] - 0.05
        smpl_data.transl[1] = 0.1
        if self.furniture.furniture_type == 'stool': smpl_data.transl[1] += 0.15
        elif self.furniture.furniture_type == 'stool2': smpl_data.transl[1] += 0.075
        angles = self.get_all_human_angles()
        
        p.removeAllUserDebugItems()
        
        # for i in range(5): 
            # p.addUserDebugText("final pose: " + str(3 - i), [0, 0, 1.5], [1, 0, 0]) # red text
            # time.sleep(1)
            # p.removeAllUserDebugItems()

        self.write_human_pkl(smpl_data)
        smpl_data.smpl_file = self.smpl_file
        smpl_data = self.assign_smplx_features(smpl_data)
        return smpl_data, angles
    
    def reset_mesh(self, smpl_data: SMPLData, angles): 
        super(SeatedPoseEnv, self).reset()
        angs = self.config_mesh_angles(angles)

        # magic happen here - now call agent.init()
        smpl_data = self.assign_smplx_gender(smpl_data)
        # self.build_assistive_env('diningchair', human_angles=angs, smpl_data=smpl_data)
        self.build_assistive_env('couch', human_angles=angs, smpl_data=smpl_data)
        self.set_env_camera()
        # self.align_mesh()

        # RESET: human
        if len(list(smpl_data.transl)) == 1: pos = list(smpl_data.transl[0])
        else: pos = list(smpl_data.transl)

        # if self.furniture.furniture_type == 'wheelchair2': pos[1] -= 0.1
        if self.furniture.furniture_type == 'stool':
            pos[2] = self.furniture.get_heights(set_on_ground=True)[0]  + 0.45
        if self.furniture.furniture_type == 'stool2':
            pos[2] = self.furniture.get_heights(set_on_ground=True)[0]  + 0.1
        elif self.furniture.furniture_type == 'wheelchair2':
            pos[2] -= 0.05
        elif self.furniture.furniture_type == 'couch':
            pos[1] -= 0.35
        
        orient = [0.0, 0.0, 0.0]
        orient[0] -= np.pi/4
        if self.furniture.furniture_type == 'stool':
            orient[0] = -np.pi/5
        orient = convert_aa_to_euler_quat(orient)[0]
        self.human.set_base_pos_orient(pos, orient)
        
        # check for floating human and adjust
        if self.furniture.furniture_type in ['stool', 'wheelchair2']: self.align_mesh(downward_shift=True)
        elif self.furniture.furniture_type == 'couch': self.align_mesh(forward_shift=True)

        # STEP: simulation
        p.setTimeStep(1/240., physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        
        # RUN: sim for viewing etc.
        # print("WARNING: not writing data, please change in seated_pose_env.py reset_mesh()")
        # self.write_train_data(smpl_data, angles)
        print("right ankle location: ", self.human.get_pos_orient(8))
        print("left  ankle location: ", self.human.get_pos_orient(7))

        # for i in range(10):
            # p.stepSimulation(physicsClientId=self.id)
            # keys = p.getKeyboardEvents()
            # if ord('q') in keys:
            #     pos, ori = p.getBasePositionAndOrientation(1)
            #     copy_pos = (pos[0], pos[1] - 0.1, pos[2])
            #     p.resetBasePositionAndOrientation(1, copy_pos, ori)
            # time.sleep(1)

        self.write_train_data(smpl_data, angles)
    
        return 0


