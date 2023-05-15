class HumanPipDict:
    def __init__(self):
        self.joint_dict = {
            "left_hip": 2,
            "left_knee": 6,
            "left_ankle": 10,
            "left_foot": 14,
            "left_toe": 17,
            "left_heel": 18,
            "right_hip": 19,
            "right_knee": 23,
            "right_ankle": 27,
            "right_foot": 31,
            "right_toe": 34,
            "right_heel": 35,
            "spine_2": 37,
            "spine_3": 41,
            "spine_4": 45,
            "neck": 49,
            "head": 53,
            "left_clavicle": 56,
            "left_shoulder": 60,
            "left_elbow": 64,
            "left_lowarm": 68,
            "left_hand": 72,
            "right_clavicle": 75,
            "right_shoulder": 79,
            "right_elbow": 83,
            "right_lowarm": 87,
            "right_hand": 91
        }

        self.urdf_to_smpl_dict  = {
            "left_hip": "L_Hip",
            "left_knee": "L_Knee",
            "left_ankle": "L_Ankle",
            "left_foot": "L_Foot",
            "right_hip": "R_Hip",
            "right_knee": "R_Knee",
            "right_ankle": "R_Ankle",
            "right_foot": "R_Foot",
            "spine_2": "Spine1",
            "spine_3": "Spine2",
            "spine_4": "Spine3",
            "neck": "Neck",
            "head": "Head",
            "left_clavicle": "L_Collar",
            "left_shoulder": "L_Shoulder",
            "left_elbow": "L_Elbow",
            "left_lowarm": "L_Wrist",
            "left_hand": "L_Hand",
            "right_clavicle": "R_Collar",
            "right_shoulder": "R_Shoulder",
            "right_elbow": "R_Elbow",
            "right_lowarm": "R_Wrist",
            "right_hand": "R_Hand"
        }


    def get_joint_ids(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        return [joint_id, joint_id + 1, joint_id + 2]

