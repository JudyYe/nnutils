# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (Rotate, Scale, Transform3d, Translate,
                                  euler_angles_to_matrix)
from . import geom_utils, mesh_utils
from pytorch3d.structures import Pointclouds
from .my_pytorch3d import Meshes
from pytorch3d.renderer import TexturesVertex
import pytorch3d.structures.utils as struct_utils


# ### Primitives Utils ###
def create_cube(device, N=1, align='center'):
    """
    :return: verts: (1, 8, 3) faces: (1, 12, 3)
    """
    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    if align == 'center':
        cube_verts -= .5

    # faces corresponding to a unit cube: 12x3
    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )  # 12, 3

    return cube_verts.unsqueeze(0).expand(N, 8, 3), cube_faces.unsqueeze(0).expand(N, 12, 3)


def create_coord(device, N=1, size=1):
    """Meshes of xyz-axis, each is 1unit, in color RGB

    :param device: _description_
    :param N: _description_
    :return: xyz Meshes in batch N. 
    """
    if isinstance(device, int):
        device = torch.device(device)
    scale_size = Scale(torch.tensor([size, size, size], dtype=torch.float32, device=device))
    scale = Scale(torch.tensor(
        [
            [1, 0.05, 0.05],
            [0.05, 1, 0.05],
            [0.05, 0.05, 1],
        ], dtype=torch.float32, device=device
    ))
    translate = Translate(torch.tensor(
        [
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ], dtype=torch.float32, device=device
    ))
    rot = Rotate(euler_angles_to_matrix(torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=torch.float32, device=device
    ), 'XYZ'))
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate)
    transform = transform.compose(scale_size)

    each_verts, each_faces, num_cube = 8, 12, 3
    verts, faces = create_cube(device, num_cube)
    verts = (transform.transform_points(verts)).view(1, num_cube * each_verts, 3)  # (3, 8, 3) -> (1, 32, 3)
    offset = torch.arange(0, num_cube, device=device).unsqueeze(-1).unsqueeze(-1) * each_verts  # faces offset
    faces = (faces + offset).view(1, num_cube * each_faces, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)
    
    textures = torch.zeros_like(verts).reshape(N, num_cube, each_verts, 3)
    textures[:, 0, :, 0] = 1 
    textures[:, 1, :, 1] = 1 
    textures[:, 2, :, 2] = 1 
    textures = textures.reshape(N, num_cube * each_verts, 3)

    meshes = Meshes(verts, faces, TexturesVertex(textures)).to(device)
    return meshes


def create_line(x1, x2, width=None):
    """
    :param x1: Tensor in shape of (N?, 3)
    :param x2: Tensor in shape of (N?, 3)
    :return: padded verts (N, Vcube, 3) and faces (N, Fcube, 3) --> WITHOUT offset
    """
    device = x1.device
    N = len(x1)
    norm = torch.linalg.vector_norm(x2 - x1, dim=-1)  # (N, )
    if width is None:
        thin = norm / 25
    else:
        thin = torch.zeros_like(norm)+ width
    scale = Scale(
        norm, thin, thin, device=device)
    translate = Translate((x2 + x1) / 2, device=device)

    e1 = (x2 - x1) / norm[..., None]

    r = F.normalize(torch.randn_like(e1))
    happy = torch.zeros([N, ], device=device, dtype=torch.bool)
    for i in range(10):
        rand = F.normalize(torch.randn_like(e1))
        r[~happy] = rand
        happy = torch.linalg.vector_norm(torch.cross(r, e1, -1), dim=-1).abs() > 1e-6
        if torch.all(happy):
            break
    if not torch.all(happy):
        print(e1, r, )
        print('!!!! Warning! Cannot find a vector to orthogonize')
    e2 = torch.cross(e1, r)
    e3 = torch.cross(e1, e2)
    rot = Rotate(torch.stack([e1, e2, e3], dim=1), device=device) # seems R is the transposed rot / or row-vector
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate)

    each_verts, each_faces, num_cube = 8, 12, 1
    verts, faces = create_cube(device, num_cube)  # (num_cube=1, 8, 3)
    verts = (transform.transform_points(verts)).view(N, num_cube * each_verts, 3)  # (N, 8, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)

    return verts, faces


# ### Gripper Utils ###
def gripper_mesh(se3=None, mat=None, texture=None, return_mesh=True):
    """
    :param se3: (N, 6)
    :param mat:
    :return:
    """
    if mat is None:
        mat = geom_utils.se3_to_matrix(se3)  # N, 4, 4
    device = mat.device

    verts, faces = create_gripper(mat.device)  # (1, V, 3), (1, V, 3)
    t = Transform3d(matrix=mat.transpose(1, 2), device=device)

    verts = t.transform_points(verts)
    faces = faces.expand(mat.size(0), faces.size(1), 3)

    if texture is not None:
        texture = texture.unsqueeze(1).to(device) + torch.zeros_like(verts)
    else:
        texture = torch.ones_like(verts)

    if return_mesh:
        return Meshes(verts, faces, TexturesVertex(texture))
    else:
        return verts, faces, texture


def create_gripper(device, N=1):
    """
    :param: texture: (N, 3) in scale [-1, 1] or None
    :return: torch.Tensor in shape of (N, V, 3), (N, V, 3)"""
    scale = Scale(torch.tensor(
        [
            [0.005, 0.005, 0.139],
            [0.005, 0.005, 0.07],
            [0.005, 0.005, 0.06],
            [0.005, 0.005, 0.06],
        ], dtype=torch.float32, device=device
    ), device=device)
    translate = Translate(torch.tensor(
        [
            [-0.03, 0, 0, ],
            [-0.065, 0, 0, ],
            [0, 0, 0.065, ],
            [0, 0, -0.065, ],
        ], dtype=torch.float32, device=device
    ), device=device)
    rot = Rotate(euler_angles_to_matrix(torch.tensor(
        [
            [0, 0, 0],
            [0, np.pi / 2, 0],
            [0, np.pi / 2, 0],
            [0, np.pi / 2, 0],
        ], dtype=torch.float32, device=device
    ), 'XYZ'), device=device)
    align = Rotate(euler_angles_to_matrix(torch.tensor(
        [
            [np.pi / 2, 0, 0],
        ], dtype=torch.float32, device=device
    ), 'XYZ'), device=device)
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate, align)

    each_verts, each_faces, num_cube = 8, 12, 4
    verts, faces = create_cube(device, num_cube)
    verts = (transform.transform_points(verts)).view(1, num_cube * each_verts, 3)  # (4, 8, 3) -> (1, 32, 3)
    offset = torch.arange(0, num_cube, device=device).unsqueeze(-1).unsqueeze(-1) * each_verts  # faces offset
    faces = (faces + offset).view(1, num_cube * each_faces, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)

    return verts, faces


# ######## Cameras Utils ########
def create_camera(device, N, size=0.2, focal=1., cam_type='+z'):
    """create N cameras, Meshes of xyz-axis, each is 1unit, in color RGB
    return verts and meshes in shape of (N, Vcam, 3)
    :param focal: focal length in shape of (N, )
    """
    lines = [
        [[0, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [-1, 1, 1]],
        [[0, 0, 0], [1, -1, 1]],
        [[0, 0, 0], [-1, -1, 1]],
        
        [[1, -1, 1], [1, 1, 1]],
        [[1, -1, 1], [-1, -1, 1]],
        [[-1, 1, 1], [-1, -1, 1]],
        [[-1, 1, 1], [1, 1, 1]],
    ]
    L = 8
    each_verts, each_faces, = 8, 12
    lines = torch.FloatTensor(lines).to(device)  # L, 2, 3?
    lines = lines[None].repeat(N, 1, 1, 1)  # (N, L, 2, 3)
    lines[..., -1] *= focal # (N, L, 2, 3)
    lines = lines.reshape(N*L, 2, 3)  # (NL, 2, 3)
    x1, x2 = lines.split([1, 1], dim=-2) # (NL, 1, 3)
    verts, faces = create_line(x1.squeeze(-2), x2.squeeze(-2), ) # (NL, Vcube, 3)
    verts *= size
    verts = verts.reshape(N, L, each_verts, 3)
    faces = faces.reshape(N, L, each_faces, 3)

    offset = torch.arange(0, L, device=device).reshape(1, L, 1, 1) * each_verts 
    faces = (faces + offset).reshape(N, L * each_faces, 3)  # 

    verts = verts.reshape(N, L*each_verts, 3)
    faces = faces.reshape(N, L*each_faces, 3)
    return verts, faces
    

def vis_cam(wTc=None, cTw=None, color='white', cam_type='+z', size=None, focal=1):
    """visualize camera 
    return a List of Meshes, each is a camera mesh in world coordinate
    :param wTc: camera coord to world coord in shape of (4, 4), can have scale, defaults to None
    :param cTw: world coord to camera coord in shape of (4, 4), can have scale, defaults to None
    :param color: ['white', 'red', 'blue', 'yellow']
    :param size: float, camera size
    :param focal: intrinsics, float or tensor in shape of (N, )
    :return: List of Meshes, each is a camera mesh, looks at +z (pytorch3d,opencv,vision convention) or -z (openGL,graphics convention)
    """
    if cTw is not None:
        wTc = geom_utils.inverse_rt(mat=cTw, return_mat=True)
    device = wTc.device
    N = len(wTc)
    dist = mesh_utils.get_camera_dist(wTc=wTc)
    if size is None:
        size = dist.max() * 0.05
        print(size)
    cam_verts, cam_faces = create_camera(device, N, size=size, focal=focal, cam_type=cam_type)  # (N, Vcam, 3)
    wTc = Transform3d(matrix=wTc.transpose(-1, -2))
    wCam_verts = wTc.transform_points(cam_verts)

    mesh_list = []
    for n in range(N):
        m = Meshes([wCam_verts[n]], [cam_faces[n]])
        m.textures = mesh_utils.pad_texture(m, color)
        mesh_list.append(m)
    return mesh_list


# ### Pointcloud to meshes Utils ###
def pc_to_cubic_meshes(xyz: torch.Tensor = None, feature: torch.Tensor = None, pc: Pointclouds = None,
                       align='center', eps=None) -> Meshes:
    if pc is None:
        if feature is None:
            feature = torch.ones_like(xyz)
        pc = Pointclouds(xyz, features=feature)
    device = pc.device
    if pc.isempty():
        N = len(pc)
        zeros = torch.zeros([N, 0, 3], device=device)
        meshes = Meshes(zeros, zeros, textures=TexturesVertex(zeros))
    else:
        N, V, D = pc.features_padded().size()

        norm = torch.sqrt(torch.sum(pc.points_padded() ** 2, dim=-1, keepdim=True))
        std = torch.std(norm, dim=1, keepdim=True)  # (N, V, 3)
        if eps is None:
            eps = (std / 10).clamp(min=5e-3)  # N, 1, 1
        # eps = .1

        cube_verts, cube_faces = create_cube(device, align=align)

        num = pc.num_points_per_cloud()  # (N, )

        faces_list = [cube_faces.expand(num_e, 12, 3) for num_e in num]
        faces_offset = [torch.arange(0, num_e, device=device).unsqueeze(-1).unsqueeze(-1) * 8 for num_e in num]
        faces_list = [(each_f + each_off).view(-1, 3) for each_f, each_off in zip(faces_list, faces_offset)]

        verts = cube_verts.expand(N, 8, 3) + torch.randn([N, 8, 3], device=device) * 0.01  # (1, 8, 3)
        verts = pc.points_padded().unsqueeze(-2) + (verts * eps).unsqueeze(1)  # N, 8, 3  -> N, V, 8, 3
        verts = verts.view(N, V * 8, 3)

        num8_list = (num * 8).tolist()
        verts_list = struct_utils.padded_to_list(verts, num8_list)

        feature = pc.features_padded().unsqueeze(-2).expand(N, V, 8, D).reshape(N, V * 8, D)
        feature_list = struct_utils.padded_to_list(feature, num8_list)
        texture = TexturesVertex(feature_list)

        meshes = Meshes(verts_list, faces_list, texture)

    return meshes


if __name__ == '__main__':
    import os
    import numpy as np
    from jutils import image_utils, mesh_utils

    device = 'cuda:0'
    N = 10
    save_dir = '/home/yufeiy2/scratch/result/vis/'
    os.makedirs(save_dir, exist_ok=True)
    t = 3
    coord = create_coord(device)

    # pytorch3d camera
    rot = geom_utils.random_rotations(N, device=device)
    trans = torch.FloatTensor([[0, 0, t]]).to(device)
    cTw = geom_utils.rt_to_homo(rot, trans.repeat(N, 1))

    cams = vis_cam(cTw=cTw, color='blue', size=0.2, focal=2) # size can be specified

    id_trans = torch.eye(3).to(device).reshape(1, 3, 3) 
    id_cam = vis_cam(cTw=geom_utils.rt_to_homo(id_trans, trans), color='red')
    # size default to 0.05*camera dist

    scene = mesh_utils.join_scene(cams + id_cam + [coord])
    image_utils.save_gif(
        mesh_utils.render_geom_rot(scene, scale_geom=True, out_size=1024), '%s/test' % (save_dir))


