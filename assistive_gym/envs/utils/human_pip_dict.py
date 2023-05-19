class HumanPipDict:
    def __init__(self):
        # TODO: dynamically generate this based on the URDF
        self.joint_dict = {
            "left_hip": 1,
            "left_knee": 5,
            "left_ankle": 9,
            "left_foot": 13,
            "left_toe": 17,
            "left_heel": 18,
            "right_hip": 19,
            "right_knee": 23,
            "right_ankle": 27,
            "right_foot": 31,
            "right_toe": 35,
            "right_heel": 36,
            "spine_2": 37,
            "spine_3": 41,
            "spine_4": 45,
            "neck": 49,
            "head": 53,
            "left_clavicle": 57,
            "left_shoulder": 61,
            "left_elbow": 65,
            "left_lowarm": 69,
            "left_hand": 73,
            "right_clavicle": 77,
            "right_shoulder": 81,
            "right_elbow": 85,
            "right_lowarm": 89,
            "right_hand": 93,
        }

        self.urdf_to_smpl_dict = {
            "pelvis": "Pelvis",
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
        # TODO: change the smpl and robot joint to same name
        self.joint_to_parent_joint_dict = {
            "pelvis": "pelvis",
            "left_hip": "pelvis",
            "left_knee": "left_hip",
            "left_ankle": "left_knee",
            "left_foot": "left_ankle",
            "right_hip": "pelvis",
            "right_knee": "right_hip",
            "right_ankle": "right_knee",
            "right_foot": "right_ankle",
            "spine_2": "pelvis",
            "spine_3": "spine_2",
            "spine_4": "spine_3",
            "neck": "spine_4",
            "head": "neck",
            "left_clavicle": "spine_4",
            "left_shoulder": "left_clavicle",
            "left_elbow": "left_shoulder",
            "left_lowarm": "left_elbow",
            "left_hand": "left_lowarm",
            "right_clavicle": "spine_4",
            "right_shoulder": "right_clavicle",
            "right_elbow": "right_shoulder",
            "right_lowarm": "right_elbow",
            "right_hand": "right_lowarm"
        }




    def get_joint_ids(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        return [joint_id, joint_id + 1, joint_id + 2]

    def get_dammy_joint_id(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        return joint_id + 3
