class SMPLDict:
    def __init__(self):
        self.joint_dict = {
            "pelvis": 0,
            "left_hip": 1,
            "right_hip": 2,
            "lower_spine": 3,
            "left_knee": 4,
            "right_knee": 5,
            "middle_spine": 6,
            "left_ankle": 7,
            "right_ankle": 8,
            "upper_spine": 9,
            "left_foot": 10,
            "right_foot": 11,
            "neck": 12,
            "left_collar": 13,
            "right_collar": 14,
            "head": 15,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "left_elbow": 18,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
            "left_hand": 22,
            "right_hand": 23,
        }

    def get_pose_ids(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        base = 3 * joint_id
        return [base, base + 1, base + 2]
