import torch
from pytorch3d import transforms as t3d

from utils.misc import timing


# Copied from pytorch3d. We currently using 0.3.0, but the latest version is 0.7.5 and they only distribute it via conda
# We are using pip instead
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def transform_euler_to_aa_fn(euler_angle: torch.Tensor):
    """
    :param tensor: 3J
    :return:
    """
    mat = t3d.euler_angles_to_matrix(euler_angle, "XYZ")
    quat = t3d.matrix_to_quaternion(mat)
    aa = quaternion_to_axis_angle(quat)
    return aa

# TODO: make it more efficent by doing some labmda magic
# https://discuss.pytorch.org/t/apply-a-function-similar-to-map-on-a-tensor/51088/5
# @timing
def transform_euler_to_aa(euler_angles: torch.Tensor):
    """
    :param tensor: 3J
    :return:
    """
    euler_angles = torch.reshape(euler_angles, (-1, 3))
    res = []
    for eu_angle in euler_angles:
        aa = transform_euler_to_aa_fn(eu_angle)
        res.append(aa)
    # print ("res: ", res, torch.cat(res, dim=0))
    return torch.cat(res, dim=0)
    # euler_angles = torch.reshape(euler_angles, (-1, 3))
    # euler_angles.apply_(lambda angle: transform_euler_to_aa_fn(angle))
    # return torch.cat(euler_angles, dim=0)


def transform_aa_to_euler(tensor: torch.Tensor):
    angles = torch.reshape(tensor, (-1, 3))
    res = []
    for a in angles:
        mat = t3d.axis_angle_to_matrix(a)
        euler = t3d.matrix_to_euler_angles(mat, "XYZ")
        res.append(euler)
    return torch.cat(res, dim=0)

def get_device():
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

class ModelInput:
    def __init__(self, pose, betas):
        self.pose = pose  # len 72
        self.betas = betas  # len 10
        assert len(pose) == 72, "pose should be len 72"
        assert len(betas) == 10, "betas should be len 10"

    def to_tensor(self):
        data = self.pose + self.betas
        return torch.tensor(data, dtype=torch.float)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        # expect len tensor = 72 + 10
        tensor = tensor.tolist()
        pose = tensor[:72]
        betas = tensor[72:]
        return cls(pose, betas)

class ModelOutput:
    def __init__(self, human_joint_angles, robot_base_pos, robot_base_orient, robot_joint_angles):
        self.human_joint_angles = human_joint_angles  # len 15
        self.robot_base_pos = robot_base_pos  # len 3
        self.robot_base_orient = robot_base_orient  # len 4
        self.robot_joint_angles = robot_joint_angles  # len 10
        assert len(human_joint_angles) == 15, "human_joint_angles should be len 15"
        assert len(robot_base_pos) == 3, "robot_base_pos should be len 3"
        assert len(robot_base_orient) == 4, "robot_base_orient should be len 4"
        assert len(robot_joint_angles) == 10, "robot_joint_angles should be len 10"

    def to_tensor(self):
        data = self.human_joint_angles + self.robot_base_pos + self.robot_base_orient + self.robot_joint_angles
        # data = self.human_joint_angles
        data = torch.tensor(data, dtype=torch.float, device=get_device())
        data[:15] = transform_euler_to_aa(data[:15])
        return data

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        # expect len tensor = 15 + 3 + 4 + 10
        # convert tensor to list
        human_joint_angles = transform_aa_to_euler(tensor[:15])
        robot_base_pos = tensor[15:18]
        robot_base_orient = tensor[18:22]
        robot_joint_angles = tensor[22:]
        return cls(human_joint_angles.to_list(), robot_base_pos.to_list(), robot_base_orient.to_list(), robot_joint_angles.to_list())

    def convert_to_dict(self):
        return {
            'human': {
                'joint_angles': self.human_joint_angles,
            },
            'robot': {
                'base': [self.robot_base_pos, self.robot_base_orient],
                'joint_angles': self.robot_joint_angles
            }
        }


class ModelOutputHumanOnly:
    def __init__(self, human_joint_angles):
        self.human_joint_angles = human_joint_angles  # len 15
        assert len(human_joint_angles) == 15, "human_joint_angles should be len 15"

    # @timing
    def to_tensor(self):
        data = self.human_joint_angles
        # data = self.human_joint_angles
        data = torch.tensor(data, dtype=torch.float, device=get_device())
        data[:15] = transform_euler_to_aa(data[:15])
        return data

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        # expect len tensor = 15 + 3 + 4 + 10
        # convert tensor to list
        human_joint_angles = transform_aa_to_euler(tensor[:15])
        # robot_base_pos = tensor[15:18]
        # robot_base_orient = tensor[18:22]
        # robot_joint_angles = tensor[22:]
        return cls(human_joint_angles.to_list())

    def convert_to_dict(self):
        return {
            'human': {
                'joint_angles': self.human_joint_angles,
            },
        }