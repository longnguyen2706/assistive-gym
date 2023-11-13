# This is a Python script from https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
import cv2
import datetime
import numpy as np
import pickle as pkl
from error.heatmap_plot import build_map_pkl

# For smpl body:
# define the body line indexes
            # main_body = [0, 9, 12, 15]
            # shoulders = [16, 12, 17]
            # left_arm = [16, 18, 20, 22] # this works for SMPL body, commented out for testing Henry's model
            # right_arm = [17, 19, 21, 23] # same as left arm
            # hips = [1, 0, 2]
            # left_leg = [1, 4, 7, 10]
            # right_leg = [2, 5, 8, 11]

# For other body:
# define the body line indexes
            # main_body = [0, 1, 3]
            # shoulders = [6, 7]
            # left_arm = [1, 7, 9, 11] 
            # right_arm = [1, 6, 8, 10]
            # hips = [16, 3, 17]
            # left_leg = [17, 15, 19]
            # right_leg = [16, 14, 18]

# list: [lhip, pelvis, rhip, lshoulder, neck, rshoulder, lelbow, relbow, lwrist, rwrist, lknee, rknee, lankle, rankle]
# [smpl body, other]
lhip = [0, 0]
# pelvis = [0, 3]
rhip = [2, 2]
lshoulder = [3, 3]
neck = [4, 4]
rshoulder = [5, 5]
leblow = [6, 6]
relbow = [7, 7]
lwrist = [8, 8]
rwrist = [9, 9]
lknee = [10, 10]
rknee = [11, 11]
lankle = [12, 12]
rankle = [13, 13]
joint_map = [lhip, rhip, lshoulder, neck, rshoulder, leblow, relbow, lwrist, rwrist, lknee, rknee, lankle, rankle]

def mpjpe(smpl_body, ag_body):
    s = 0
    i = 0
    smpl_neck = smpl_body[0] # this is the pelvis
    ag_neck = ag_body[0]
    for i,s_joint in enumerate(smpl_body):
        smpl = s_joint - smpl_neck
        ag = ag_body[i] - ag_neck
        s += np.abs(np.linalg.norm(smpl - ag))
        i += 1
    return s/i

# function returns a dictionary with the specific joint and the squared error betweeen the two models
# note: the SMPL model must be passed in first to obtain the correct estimation
def squared_errors(bp_body, opt_body):
    # the order that joints will be added to the dictionary
    order = ["left_hip", "pelvis", "right_hip", "left_shoulder", "neck", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
    errors = {}
    i = 0
    for idx in joint_map:
        print(i, ": entered")
        # obtain squared difference
        joint_diff = bp_body[idx[0]] - opt_body[idx[1]]
        sq_joint_diff = joint_diff * joint_diff
        sse = sq_joint_diff[0] + sq_joint_diff[1] + sq_joint_diff[2]
        # put the data into the dictionary under the corresponding name
        print("sse: ", sse)
        errors[order[i]] = sse
        i = i + 1
    return errors


# function returns one value that is the sum of all the squared errors
# note: the SMPL body must be passed in as the bp_body
def sum_squared_errors(bp_body, opt_body):
    sum = 0
    for idx in joint_map:
        add = (bp_body[idx[0]] - opt_body[idx[1]])
        sum = sum + (add * add)
    return sum


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
