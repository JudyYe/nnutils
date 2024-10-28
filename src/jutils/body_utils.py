import os
import os.path as osp
import pickle

import numpy as np
import pytorch3d.transforms as transforms
import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from . import geom_utils

class BodyWrapper(nn.Module):
    def __init__(
        self,
        smplh_path="/is/cluster/fast/yye/pretrain/body_models/smplh/male/model.npz",
        num_betas=16,
        dmpl_fname=None,
        num_dmpls=None,
        num_expressions=None,
        gender="male",
    ) -> None:
        super().__init__()
        self.num_betas = num_betas
        # surface_model_male_fname = osp.join(smplh_path, gender, "model.npz")
        surface_model_male_fname = osp.join(smplh_path)
        self.model = BodyModel(
            bm_fname=surface_model_male_fname,
            num_betas=num_betas,
            num_expressions=num_expressions,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
    
    def get_shaped_offset(self, axisang, betas=None):
        """
        :param betas: (N, 16)
        :
        """
        if betas is None:
            betas = torch.zeros(axisang.size(0), self.num_betas, device=axisang.device)

        device = axisang.device
        N = axisang.size(0)
        # pelvis
        pose = torch.cat([axisang, torch.zeros(N, 21 * 3, device=device)], 1)
        _, jts, = self(None, pose, betas=betas, )
        t_mano = jts[:, 0]

        rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
        delta = t_mano - torch.matmul(rot_r, t_mano.unsqueeze(-1)).squeeze(-1)
        return delta

    def shaped_i2o(self,trans, axisang, betas=None):
        trans = trans + self.get_shaped_offset(axisang, betas)
        return trans, axisang

    def shaped_o2i(self, trans, axisang, betas=None):
        trans = trans - self.get_shaped_offset(axisang, betas)
        return trans, axisang
    
    @staticmethod
    def get_offset(axisang, betas=None):
        """
        :param axisang: (N, 3)
        :return: trans: (N, 3) = r_r - R_r t_r
        """
        device = axisang.device
        N = axisang.size(0)
        # pelvis
        t_mano = (
            torch.tensor([[-0.0022, -0.2408, 0.0286]], device=device)
            .float()
            .repeat(N, 1)
        )
        rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
        delta = t_mano - torch.matmul(rot_r, t_mano.unsqueeze(-1)).squeeze(-1)
        return delta

    @staticmethod
    def i2o(trans, axisang, betas=None):
        trans = trans + BodyWrapper.get_offset(axisang)
        return trans, axisang

    @staticmethod
    def o2i(trans, axisang, betas=None):
        trans = trans - BodyWrapper.get_offset(axisang)
        return trans, axisang

    @staticmethod
    def get_smpl_parents():
        bm_path = os.path.join(SMPLH_PATH, "male/model.npz")
        npz_data = np.load(bm_path)
        ori_kintree_table = npz_data["kintree_table"]  # 2 X 52
        parents = ori_kintree_table[0, :22]  # 22
        parents[0] = -1  # Assign -1 for the root joint's parent idx.
        return parents

    def get_rest_pose_joints(self):
        zero_root_trans = torch.zeros(1, 3).cuda().float()
        zero_rot_aa_rep = torch.zeros(1, 22, 3).cuda().float()
        betas = torch.zeros(1, 16).cuda().float()
        _, rest_human_jnts, _ = self._forward_layer(
            self.model, zero_rot_aa_rep, zero_root_trans, betas
        )
        # 1 X 1 X J X 3

        parents = BodyWrapper.get_smpl_parents()
        parents[0] = (
            0  # Make root joint's parent itself so that after deduction, the root offsets are 0
        )
        rest_human_offsets = (
            rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]
        )

        return rest_human_offsets  # 1 X J X 3

    def forward(
        self,
        glb_wTp,
        rot_rep: torch.Tensor,
        betas=None,
        trans=None,
        texture="verts",
        return_mesh=True,
        **kwargs,
    ):
        """

        :param glb_trans: (B, 4, 4)
        :param rot_rep: (B, 21(+1), 3) or (B, 21*3)
        :param betas: _description_, defaults to None
        :param gender: _description_, defaults to 'male'

        :return joints: (B, 22, 3)
        :return verts: (B, 6890, 3)
        :return faces: (B, 13776, 3)
        """
        bs = rot_rep.shape[0]
        device = rot_rep.device

        if rot_rep.ndim == 2:
            rot_rep = rot_rep.reshape(bs, -1, 3)
        if rot_rep.shape[1] != 22:
            rot_pelvis = torch.zeros_like(rot_rep[:, 0:1])
            rot_rep = torch.cat([rot_pelvis, rot_rep], 1)

        if trans is None:
            trans = torch.zeros_like(rot_rep[:, 0])
        if betas is None:
            betas = torch.zeros([bs, self.num_betas], device=device)
        verts, joints, faces = self._forward_layer(
            self.model, rot_rep, trans, betas, **kwargs
        )
        if glb_wTp is not None:
            glb_wTp = Transform3d(matrix=glb_wTp.transpose(-1, -2))
            verts = glb_wTp.transform_points(verts)
            joints = glb_wTp.transform_points(joints)

        if texture == "verts":
            textures = torch.ones_like(verts)
            textures = TexturesVertex(textures)
        elif torch.is_tensor(texture):
            textures = TexturesUV(
                texture, self.faces_uv.repeat(bs, 1, 1), self.verts_uv.repeat(bs, 1, 1)
            )
        elif texture == "uv":
            ones = torch.ones([bs, 64, 64, 3], device=device)
            textures = TexturesUV(
                ones, self.faces_uv.repeat(bs, 1, 1), self.verts_uv.repeat(bs, 1, 1)
            )
        else:
            raise NotImplementedError
        if return_mesh:
            return Meshes(verts, faces, textures), joints
        else:
            return verts, faces, textures, joints

    @staticmethod
    def _forward_layer(bm, aa_rot_rep, root_trans, betas, **kwargs):
        """

        :param bm: _description_
        :param aa_rot_rep: (B, 22, 3)
        :param root_trans: (B, 3)
        :param betas: (B, 16)
        :return: verts, joints, faces in shape of (N, V, 3), (N, J, 3), (N, F, 3)
        """
        bs, num_joints, _ = aa_rot_rep.shape
        if num_joints != 52:
            padding_zeros_hand = torch.zeros(bs, 30, 3).to(
                aa_rot_rep.device
            )  # BS X T X 30 X 3
            aa_rot_rep = torch.cat(
                (aa_rot_rep, padding_zeros_hand), dim=-2
            )  # BS X T X 52 X 3

        cur_pred_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
        cur_pred_pose = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
        cur_pred_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90
        cur_pred_trans = root_trans

        pred_body = bm(
            pose_body=cur_pred_pose,
            pose_hand=cur_pred_pose_hand,
            betas=betas,
            root_orient=cur_pred_orient,
            trans=cur_pred_trans,
        )
        joints = pred_body.Jtr[:, :num_joints, :]
        verts = pred_body.v
        faces = pred_body.f[None].repeat(bs, 1, 1)

        return verts, joints, faces


def load_pose_file():
    with open(pose_file, "rb") as f:
        data = pickle.load(f)
    trans, pose, head = data
    trans = trans.to(device).float()
    pose = pose.to(device).float()
    head = head.to(device).float()
    tsl, quat = head[..., :3], head[..., 3:]
    mat = transforms.quaternion_to_matrix(quat)
    wThead = geom_utils.rt_to_homo(mat, tsl)
    return trans, pose, wThead


def test():
    trans, pose, wThead = load_pose_file()  # (T, 3), (T, 22, 3), (T, 4, 4)
    # wThead = geom_utils.inverse_rt(mat=wThead, return_mat=True)
    print("trans", trans.shape, pose.shape, wThead.shape)
    num_betas = 16

    wrapper = BodyWrapper(SMPLH_PATH, num_betas=num_betas).to(device)

    trans = trans[0:1]
    pose = pose[0:1]

    trans_glb, rot_glb = BodyWrapper.i2o(trans, pose[:, 0, :])
    wTp = geom_utils.axis_angle_t_to_matrix(rot_glb, trans_glb)
    wBody, _ = wrapper(wTp, pose[:, 1:])  # outer
    wBody.textures = mesh_utils.pad_texture(wBody, "white")

    iBody, _ = wrapper(None, pose, trans=trans)  # inner
    iBody.textures = mesh_utils.pad_texture(iBody, "red")

    scene = mesh_utils.join_scene([wBody, iBody])
    coord = plot_utils.create_coord(device, 1, 1)
    scene = mesh_utils.join_scene([scene, coord])
    image_list = mesh_utils.render_geom_rot_v2(scene)
    image_utils.save_gif(image_list, osp.join(save_dir, "smplh_testwrapper"))


if __name__ == "__main__":
    from fire import Fire
    from jutils import geom_utils, image_utils, mesh_utils, plot_utils

    save_dir = "tmp/"
    SMPLH_PATH = "/is//cluster/fast/yye/pretrain/body_models/smplh"
    device = "cuda:0"
    pose_file = "/is/cluster/fast/yye/egoego_release/test_data_res/egoego_demo_on_ares/frl_apartment_4-MPI_HDM05_bk_HDM_bk_03-02_02_120_poses_827_frames_30_fps_b649seq0_samp_5_pose.pkl"
    Fire(test)
