import os
import numpy as np
import smplx
import trimesh

from vtk import (
    vtkQuadricDecimation,
    vtkPolyData,
    vtkSTLReader,
    vtkSTLWriter,
)
import torch
from assistive_gym.envs.utils.smpl_parser import SMPL_Parser

# BETAS = torch.Tensor(np.random.uniform(-1, 5, (1, 10))) # random betas
BETAS = torch.Tensor(np.zeros((1, 10))) # random betas
# show human mesh for debugging
def show_human_mesh(model_path):
    model = smplx.create(model_path, model_type='smpl', gender='neutral')
    output = model(betas=BETAS, body_pose=torch.Tensor(np.zeros((1, model.NUM_BODY_JOINTS*3))), return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    out_mesh = trimesh.Trimesh(vertices, model.faces)
    out_mesh.show()

def generate_body_hull(jname, vert, outdir, joint_pos = (0, 0, 0)):
    """
    Generate a convex hull for each joint
    Save the convex hull as an obj file with vert name
    :param jname: joint name
    :param vert: array of vertices representing the joint
    :return:
    """

    p_cloud = trimesh.PointCloud(vert)
    p_hull = p_cloud.convex_hull
    centroid = p_hull.centroid
    # the original hull will render at exactly where the body part should be.
    # we move the body part to origin so that we could put it to the right place with joint position later
    p_hull.vertices = p_hull.vertices - joint_pos

    print (jname, "pos: ", joint_pos, " centroid: ", centroid, " volume: ", p_hull.volume, " area: ", p_hull.area, " inertia: ", p_hull.moment_inertia, )
    # Export the mesh to an OBJ file
    outfile = f"{outdir}/{jname}.obj"
    p_hull.export(outfile)
    return {
        "filename": outfile,
        "hull": p_hull,
    }

def generate_geom(model_path):
    smpl_parser = SMPL_Parser(model_path)
    pose = torch.zeros((1, 72))
    # betas = torch.zeros((1, 10)).float()
    (
        smpl_verts,
        smpl_jts,
        skin_weights,
        joint_names,
        joint_offsets,
        joint_parents,
        joint_axes,
        joint_dofs,
        joint_range,
        contype,
        conaffinity,
    ) = smpl_parser.get_mesh_offsets(pose, betas=BETAS)

    vert_to_joint = skin_weights.argmax(axis=1)
    hull_dict = {}

    # create joint geometries
    geom_dir = "/home/louis/Documents/Projects/assistive-gym/assistive_gym/envs/assets/human/meshes/"
    os.makedirs(geom_dir, exist_ok=True)
    joint_pos_dict = {}

    for jind, jname in enumerate(joint_names):
        vind = np.where(vert_to_joint == jind)[0]
        if len(vind) == 0:
            print(f"{jname} has no vertices!")
            continue
        # vert = (smpl_verts[vind] - smpl_jts[jind]) * scale_dict.get(jname, 1) + smpl_jts[jind]
        vert = (smpl_verts[vind] - smpl_jts[jind]) +  smpl_jts[jind]
        r = generate_body_hull(jname, vert, geom_dir, joint_pos=smpl_jts[jind])
        joint_pos_dict[jname] = smpl_jts[jind]

        hull_dict[jname] = {
            "verts": vert,
            "hull": r["hull"],
            "filename": r["filename"],
        }

    return hull_dict, joint_pos_dict, joint_offsets

# TODO: Clear this out
def quadric_mesh_decimation(fname, reduction_rate, verbose=False):
    reader = vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    inputPoly = reader.GetOutput()

    decimate = vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(reduction_rate)
    decimate.Update()
    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print(
            f"Mesh Decimation: (points, faces) goes from ({inputPoly.GetNumberOfPoints(), inputPoly.GetNumberOfPolys()}) "
            f"to ({decimatedPoly.GetNumberOfPoints(), decimatedPoly.GetNumberOfPolys()})"
        )

    stlWriter = vtkSTLWriter()
    stlWriter.SetFileName(fname)
    stlWriter.SetFileTypeToBinary()
    stlWriter.SetInputData(decimatedPoly)
    stlWriter.Write()

if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "examples/data/SMPL_NEUTRAL.pkl")

    generate_geom(model_path)
    show_human_mesh(model_path)