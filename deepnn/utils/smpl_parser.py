# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script
import smplx
import torch
import numpy as np
import trimesh
from smplx import SMPL as _SMPL
import pyrender

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

LEFT_HAND_JOINTS = ["L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"]
RIGHT_HAND_JOINTS  = ["R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]

# convert to map
SMPL_BONE_IDX_MAP = {name: i for i, name in enumerate(SMPL_BONE_ORDER_NAMES)}
LEFT_HAND_JOINT_INDICES = [SMPL_BONE_IDX_MAP[name] for name in LEFT_HAND_JOINTS]
RIGHT_HAND_JOINT_INDICES = [SMPL_BONE_IDX_MAP[name] for name in RIGHT_HAND_JOINTS]

SMPL_BONE_KINTREE_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest',
    'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow',
    'R_Wrist', 'R_Hand'
]


def merge_end_effector_joint_angle(pose, joint_angles, end_effector):  # TODO: make as batch
    """
    pose: Bx72
    end_effector: Bx1
    joint_angles: Bx15
    """

    joint_indices = LEFT_HAND_JOINT_INDICES if end_effector == 0 else RIGHT_HAND_JOINT_INDICES
    # print(joint_indices)
    for i in range(len(joint_angles) // 3):
        joint_idx = joint_indices[i]
        # print (joint_idx*3, (joint_idx+1)*3, i*3, (i+1)*3)
        pose[joint_idx * 3:(joint_idx + 1) * 3] = joint_angles[i * 3:(i + 1) * 3]
    return pose
class SMPL_Parser(_SMPL):
    def __init__(self, *args, **kwargs):
        """SMPL model constructor
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None, vis=False): # TOOD: make the inference run on GPU (currently need to move to CPU)
        """
        Pose should be batch_size x 72
        """
        # if pose.shape[-1] != 72:
        #     # pose = pose.reshape(-1, 72)
        # pose = pose.float()
        pose = pose.to(self.device)

        if th_betas is not None:
            # th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]
            th_betas = th_betas.to(self.device)
        if th_trans is not None:
            th_trans = th_trans.to(self.device)

        smpl_output = self.forward(
            betas=th_betas,
            transl=th_trans,
            body_pose=torch.reshape(pose[:, 3:], (-1, 23, 3)),
            global_orient=torch.reshape(pose[:, :3], (-1, 1, 3)),
            gender='neutral'
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:, :24]
        betas = smpl_output.betas
        if vis:
            self.render(vertices)
        return vertices, joints

    def render(self, vertices):
        """
        Render the vertices using pyrender
        """
        if vertices.shape[0] == 1:
            vertices = vertices.squeeze(0)
        mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), self.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)
