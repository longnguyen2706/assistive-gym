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

    def get_joint_ids(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        return [joint_id, joint_id + 1, joint_id + 2]
