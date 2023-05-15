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

BETAS = torch.Tensor(np.random.uniform(-1, 5, (1, 10))) # random betas
# BETAS = torch.Tensor(np.zeros((1, 10))) # random betas
# show human mesh for debugging
def show_human_mesh(model_path):
    model = smplx.create(model_path, model_type='smpl', gender='neutral')
    output = model(betas=BETAS, body_pose=torch.Tensor(np.zeros((1, model.NUM_BODY_JOINTS*3))), return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    out_mesh = trimesh.Trimesh(vertices, model.faces)
    out_mesh.show()

def generate_body_hull(jname, vert, outdir):
    """
    Generate a convex hull for each joint
    Save the convex hull as an obj file with vert name
    :param jname: joint name
    :param vert: array of vertices representing the joint
    :return:
    """

    p_cloud = trimesh.PointCloud(vert)
    p_hull = p_cloud.convex_hull

    # p_hull.show()
    # print (jname, " volume: ", p_hull.volume, " area: ", p_hull.area, " inertia: ", p_hull.moment_inertia)
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
    geom_dir = "/home/louis/Downloads/goem"
    os.makedirs(geom_dir, exist_ok=True)
    joint_pos_dict = {}

    for jind, jname in enumerate(joint_names):
        vind = np.where(vert_to_joint == jind)[0]
        if len(vind) == 0:
            print(f"{jname} has no vertices!")
            continue
        # vert = (smpl_verts[vind] - smpl_jts[jind]) * scale_dict.get(jname, 1) + smpl_jts[jind]
        vert = (smpl_verts[vind] - smpl_jts[jind]) +  smpl_jts[jind]
        # hull = ConvexHull(vert)
        #
        # norm_verts = vert - smpl_jts[jind]
        # norm_hull = ConvexHull(norm_verts)
        # hull_dict[jname] = {
        #     "norm_hull": norm_hull,
        #     "norm_verts": norm_verts,
        #     "verts": vert,
        #     "hull": hull,
        # }
        # # print(jname, hull.simplices.shape[0])
        #
        # center = vert[hull.vertices].mean(axis=0)
        # jgeom = mesh.Mesh(np.zeros(hull.simplices.shape[0], dtype=mesh.Mesh.dtype)) # zero volume
        # print (jname, jgeom.get_mass_properties())
        # for i, f in enumerate(hull.simplices):
        #     for j in range(3):
        #         jgeom.vectors[i][j] = vert[f[j], :]
        #     # check if the face's normal is facing outward
        #     normal = np.cross(
        #         jgeom.vectors[i][1] - jgeom.vectors[i][0],
        #         jgeom.vectors[i][2] - jgeom.vectors[i][0],
        #     )
        #     out_vec = jgeom.vectors[i].mean(axis=0) - center
        #     if np.dot(normal, out_vec) < 0:
        #         jgeom.vectors[i] = jgeom.vectors[i][[0, 2, 1]]  # flip the face
        #
        # fname = f"{geom_dir}/{jname}.stl"
        #
        #
        # min_num_vert = 50
        # cur_num_vert = len(hull.vertices)
        # reduction_rate = min(0.9, 1.0 - min_num_vert / cur_num_vert)
        #
        # quadric_mesh_decimation(fname, reduction_rate , verbose=True)
        r = generate_body_hull(jname, vert, geom_dir)
        joint_pos_dict[jname] = smpl_jts[jind]
        hull_dict[jname] = {
            "verts": vert,
            "hull": r["hull"],
            "filename": r["filename"],
        }

    return hull_dict, joint_pos_dict

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

    # def create_smpl_body(self):
    #     model_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    #     # model = smplx.create(model_path, model_type='smpl', gender='neutral')
    #     model = load_model(model_path)
    #
    #     # Generate the mesh
    #     vertices = model.r
    #     faces = model.f
    #
    #     # Get the kinematic tree
    #     kin_tree = model.kintree_table
    #
    #     # Create a mapping from vertex indices to body parts
    #     vertex_to_joint = np.zeros(vertices.shape[0], dtype=int)
    #     face_to_joint = np.zeros(faces.shape[0], dtype=int)
    #     for joint_idx in range(1, kin_tree.shape[1]):
    #         # Get the parent joint
    #         parent_joint = kin_tree[0, joint_idx]
    #
    #         # Get the vertices associated with this joint
    #         joint_vertices = np.where(faces == joint_idx)[0]
    #
    #         vertex_to_joint[joint_vertices] = parent_joint
    #
    #     # Now you can create separate meshes for each body part
    #     body_part_meshes = []
    #     for joint_idx in range(1, kin_tree.shape[1]):
    #         body_part_vertex_indices = np.where(vertex_to_joint == joint_idx)[0]
    #         body_part_vertices = vertices[body_part_vertex_indices]
    #
    #         print ("body_part_vertices: ", body_part_vertices)
    #
    #         # Get the faces associated with this body part
    #         mask = np.isin(faces, body_part_vertex_indices).all(axis=1)
    #         body_part_faces = faces[mask]
    #
    #         index_map = {old_index: new_index for new_index, old_index in enumerate(body_part_vertex_indices)}
    #
    #         # Update the faces to use the new vertex indices
    #         for old_index, new_index in index_map.items():
    #             body_part_faces[body_part_faces == old_index] = new_index
    #
    #         # Create a new mesh for this body part
    #         body_part_mesh = trimesh.Trimesh(vertices=body_part_vertices, faces=body_part_faces)
    #         body_part_meshes.append(body_part_mesh)
    #
    #     for i in range(len(body_part_meshes)):
    #         body_part_meshes[i].show()
    #
    # def create_smpl_body2(self):
    #     model_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    #     model = smplx.create(model_path, model_type='smpl', gender='neutral')
    #     output = model.forward()
    #
    #     # Generate the mesh
    #     vertices = output.vertices
    #     faces = model.faces
    #
    #
    #     vertex_to_joint = lbs.vertices2joints(model.J_regressor, vertices)
    #     # Now you can create separate meshes for each body part
    #     body_part_meshes = []
    #     for joint_idx in range(1, 23):
    #         # Get the vertices associated with this body part
    #         body_part_vertices = vertices[vertex_to_joint == joint_idx]
    #         p_cloud = trimesh.PointCloud(body_part_vertices)
    #         p_hull = p_cloud.convex_hull
    #         #
    #         # # Create a new mesh for this body part
    #         # body_part_mesh = trimesh.Trimesh(vertices=body_part_vertices, faces=faces)
    #         body_part_meshes.append(p_hull)
    #         pass
    #
    #     for i in range(len(body_part_meshes)):
    #         body_part_meshes[i].show()
    #
    #
    # def create_smpl_body3(self):
    #     model_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
    #     # model = smplx.create(model_path, model_type='smpl', gender='neutral')
    #     model = load_model(model_path)
    #
    #     # Generate the mesh
    #     vertices = model.r
    #     faces = model.f
    #
    #     mesh_idx = pickle.load(open(os.path.join(os.getcwd(), "examples/data/segmented_mesh_idx_faces.p"), "rb"), encoding="latin1")
    #     print (mesh_idx)


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "examples/data/SMPL_NEUTRAL.pkl")
    show_human_mesh(model_path)
    generate_geom(model_path)