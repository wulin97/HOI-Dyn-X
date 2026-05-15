import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as transforms
import numpy as np
import copy

from tools import *



def cat_data_dict_for_pred_gt(data_dict):
    """
    Build data_dict for concatenated [pred, gt] batch.
    If tensor value has shape [B, ...], output is [2B, ...] with [v, v].
    """
    out = {}
    for k, v in data_dict.items():
        if torch.is_tensor(v):
            out[k] = torch.cat([v, v], dim=0)
        elif isinstance(v, list):
            out[k] = v + v
        else:
            out[k] = v
    return out


def masked_mean(x, mask):
    """
    x: [B, T', ...]
    mask: [B, T']
    """
    if mask.sum() == 0:
        return x.sum() * 0.0

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    return x[mask.expand_as(x)].mean()


def get_human_info(x, dataset=None):
    """
    NOTE: inputs are normalized.
    human_joints: [B, T, 24, 3]
    human_rot_mat: [B, T, 22, 3, 3], global rotation
    human_hands_feet_contact: [B, T, 4]
    """
    human_motion = x["motion"]
    human_hands_feet_contact = x["contact_labels"]
    B, T, _ = human_motion.shape

    human_joints = human_motion[:, :, :24 * 3].reshape(B, T, 24, 3)
    global_joint_rot_6d = human_motion[:, :, 24 * 3:].reshape(B, T, 22, 6)
    human_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d)
    return human_joints, human_rot_mat, human_hands_feet_contact


def get_object_info(x, dataset, floor_height=0.02, quick_mode=True):
    """
    obj_bps: [B, 1, 1024, 3]
    obj_motion: [B, T, 12], normalized translation + local rotation
    obj_floor_contact: [B, T, 1]
    obj_pc: [B, T, Nv, 3], world-space object keypoints/geometry
    """
    obj_bps = x["input_obj_bps"]
    obj_motion = x["obj_motion"]

    if quick_mode:
        obj_pc = get_obj_world_keypoints_parallel(obj_motion, x["reference_obj_rot_mat"], x["rest_pose_obj_pts"], dataset)
        min_values = torch.min(obj_pc[:, :, :, -1], dim=2)[0]
        obj_floor_contact = ((min_values < floor_height) * 1.).unsqueeze(-1)
    else:
        obj_pc = get_world_object_geo(x["obj_name"], x["obj_motion"], x["reference_obj_rot_mat"], dataset)
        min_values = torch.stack([torch.min(item[:, :, -1], dim=1)[0] for item in obj_pc])
        obj_floor_contact = ((min_values < floor_height) * 1.).unsqueeze(-1)

    return obj_motion, obj_bps, obj_floor_contact, obj_pc


def det3x3(m):
    """
    m: [N, 3, 3]
    return: [N]
    """
    return (
        m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
        - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
        + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0])
    )


def symmetric_orthogonalization(x):
    """
    Maps 9D input vectors onto SO(3).
    GPU version without torch.det, avoiding MAGMA batched-LU crash.

    x: [N, 9]
    return:
        r: [N, 3, 3]
        det_m: [N]
    """
    m = x.view(-1, 3, 3)

    if not torch.isfinite(m).all():
        raise RuntimeError("[symmetric_orthogonalization] Input contains NaN or Inf.")

    u, s, vh = torch.linalg.svd(m, full_matrices=False)

    r_tmp = torch.bmm(u, vh)
    det = det3x3(r_tmp).sign().view(-1, 1, 1)

    vh_fixed = torch.cat((vh[:, :2, :], vh[:, -1:, :] * det), dim=1)
    r = torch.bmm(u, vh_fixed)
    det_m = torch.prod(s, dim=-1)

    return r, det_m

# def symmetric_orthogonalization(x):
#     """
#     Maps 9D input vectors onto SO(3).
#     x: [N, 9]
#     return:
#         r: [N, 3, 3]
#         det_m: [N]
#     """
#     device = x.device
#     x = x.cpu()
#     m = x.view(-1, 3, 3)
#     u, s, v = torch.svd(m)
#     vt = torch.transpose(v, 1, 2)
#     det = torch.det(torch.matmul(u, vt)).view(-1, 1, 1)
#     vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), dim=1)
#     r = torch.matmul(u, vt)
#     det_m = torch.prod(s, dim=-1)
#     return r.to(device), det_m.to(device)


def get_obj_world_keypoints_parallel(obj_motion, reference_obj_rot_mat, rest_pose_obj_pts, dataset):
    """
    obj_motion: [B, T', 12], normalized trans + local rot
    reference_obj_rot_mat: [B, 3, 3]
    rest_pose_obj_pts: [B, Nv, 3]
    return: [B, T', Nv, 3], world-space object keypoints
    """
    B, T2, _ = obj_motion.shape
    Nv = rest_pose_obj_pts.shape[1]

    transition = obj_motion[:, :, :3]
    rotation = obj_motion[:, :, 3:].reshape(B, T2, 3, 3)
    curr_obj_com_pos = dataset.de_normalize_obj_pos_min_max(transition)

    curr_obj_rot_mat = torch.bmm(rotation.reshape(-1, 3, 3), reference_obj_rot_mat.repeat(1, T2, 1, 1).reshape(-1, 3, 3))
    curr_obj_rot_mat = curr_obj_rot_mat.reshape(B, T2, 3, 3)

    rest_verts = rest_pose_obj_pts[:, None].repeat(1, T2, 1, 1).reshape(B * T2, Nv, 3)
    transformed = torch.bmm(curr_obj_rot_mat.reshape(-1, 3, 3), rest_verts.transpose(1, 2)).transpose(1, 2)
    transformed = transformed.reshape(B, T2, Nv, 3) + curr_obj_com_pos[:, :, None, :]
    return transformed




class MiniTransformer(nn.Module):
    def __init__(self, n=24, joint_dim=12, cond_dim=92, dim=64, heads=4, depth=1):
        super().__init__()
        self.joint_encoder = nn.Sequential(nn.Linear(joint_dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))
        self.cond_encoder = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n, dim))

        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.attn_pool = nn.Linear(dim, 1)

    def forward(self, joints, cond):
        x = self.joint_encoder(joints) + self.cond_encoder(cond).unsqueeze(1) + self.pos_embed
        x = self.transformer(x)
        attn = F.softmax(self.attn_pool(x).squeeze(-1), dim=1)
        feat = (x * attn.unsqueeze(-1)).sum(dim=1)
        return feat, attn


class Dynamics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.history_win = cfg["history_win"]
        self.future_win = cfg["future_win"]
        self.feat_dim = cfg["feat_dim"]
        self.head = cfg["head"]
        self.depth = cfg["depth"]
        self.rot_loss_method = cfg["rot_loss_method"]

        assert self.future_win == 1, "Clean Dynamics only supports future_win=1. Use max_step to control prediction stride."

        self.bps_encoder = nn.Sequential(nn.Linear(1024 * 3, 128), nn.ReLU(), nn.Linear(128, self.feat_dim))

        cond_dim = self.history_win * 12 + self.feat_dim + 4
        self.human_joints_transformer = MiniTransformer(n=24, joint_dim=12, cond_dim=cond_dim, dim=self.feat_dim, heads=self.head, depth=self.depth)
        self.tran_rot_enc = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 2), nn.ReLU(), nn.Linear(self.feat_dim // 2, 12))


    @staticmethod
    def data_prepare(x, dataset):
        obj_motion, obj_bps, obj_floor_contact, obj_pc = get_object_info(x, dataset)
        human_joints, human_rot_mat, human_contact = get_human_info(x, dataset)
        B, T, _ = obj_motion.shape
        hand_idx = [20, 21, 22, 23]
        feet_idx = [7, 8, 10, 11]

        inputs = {
            "obj_motion": obj_motion,
            "human_joints": human_joints.view(B, T, 24 * 3),
            "human_rot_mat": human_rot_mat.view(B, T, 22 * 3 * 3),
            "contact": human_contact.view(B, T, 4),
        }

        others = {
            "obj_pc": obj_pc.reshape(B, T, 100 * 3),
            "obj_bps": obj_bps.view(B, 1, 1024 * 3),
            "obj_floor_contact": obj_floor_contact,
            "human_hands_joints": human_joints[:, :, hand_idx].view(B, T, 4 * 3),
            "human_feet_joints": human_joints[:, :, feet_idx].view(B, T, 4 * 3),
        }

        return inputs, others

    @staticmethod
    def compose_object_history(inputs, history_ts):
        obj_motion, contact = inputs["obj_motion"], inputs["contact"]
        B, H, T2 = history_ts.shape

        obj_hist = []
        for i in range(H):
            obj_hist.append(get_state_by_ts(obj_motion, history_ts[:, i, :], bt_merge=True).unsqueeze(1))

        obj_hist = torch.cat(obj_hist, dim=1)
        obj_hist_pose = obj_hist.reshape(B * T2, H * 12)
        cur_contact = get_state_by_ts(contact, history_ts[:, -1, :], bt_merge=True)
        return obj_hist_pose, cur_contact

    @staticmethod
    def compose_human_drive(inputs, history_ts, future_ts):
        human_joints, human_rot_mat = inputs["human_joints"], inputs["human_rot_mat"]
        B, H, T2 = history_ts.shape
        anchor_ts, target_ts = history_ts[:, -1, :], future_ts[:, 0, :]

        cur_joints = get_state_by_ts(human_joints, anchor_ts, bt_merge=False)
        tgt_joints = get_state_by_ts(human_joints, target_ts, bt_merge=False)
        cur_rot = get_state_by_ts(human_rot_mat, anchor_ts, bt_merge=False)
        tgt_rot = get_state_by_ts(human_rot_mat, target_ts, bt_merge=False)

        delta_joints = tgt_joints.view(B, T2, 24, 3) - cur_joints.view(B, T2, 24, 3)
        delta_rot = torch.matmul(tgt_rot.view(B, T2, 22, 3, 3), cur_rot.view(B, T2, 22, 3, 3).transpose(-1, -2)).reshape(B * T2, 22, 9)
        pad_rot = torch.zeros(B * T2, 2, 9, device=delta_rot.device, dtype=delta_rot.dtype)

        delta_joints = delta_joints.reshape(B * T2, 24, 3)
        delta_rot = torch.cat([delta_rot, pad_rot], dim=1)
        human_drive = torch.cat([delta_joints, delta_rot], dim=-1)
        return human_drive
    
    def predict(self, inputs, others, max_step=1):
        H, F, S = self.history_win, self.future_win, max_step
        B, T, _ = inputs["contact"].shape
        device = inputs["contact"].device

        history_ts, future_ts = get_advanced_timespan(B=B, H=H, F=F, T=T, S=S)
        history_ts, future_ts = history_ts.to(device), future_ts.to(device)
        B, H, T2 = history_ts.shape

        encoded_bps = self.bps_encoder(others["obj_bps"]).repeat(1, T, 1)
        obj_bps_feat = get_state_by_ts(encoded_bps, history_ts[:, -1, :], bt_merge=True)

        obj_hist_pose, cur_contact = self.compose_object_history(inputs, history_ts)

        condition = torch.cat([obj_hist_pose, obj_bps_feat, cur_contact], dim=-1)

        human_drive = self.compose_human_drive(inputs, history_ts, future_ts)
        human_feat, joints_attn = self.human_joints_transformer(human_drive, condition)

        out = self.tran_rot_enc(human_feat).view(B, T2, F, 12)
        trans, raw_rot = out[..., :3], out[..., 3:12]

        if self.rot_loss_method == "fro":
            rot, det_m = raw_rot, None
        elif self.rot_loss_method == "theta":
            rot_flat, det_m = symmetric_orthogonalization(raw_rot.reshape(-1, 9).float())
            rot = rot_flat.view(B, T2, F, 9)
        else:
            raise NotImplementedError

        return {
            "trans": trans,
            "rot": rot,
            "raw_rot": raw_rot,
            "contact": cur_contact,
            "timespan": history_ts,
            "next_timespan": future_ts,
            "method": self.rot_loss_method,
            "joints_attn": joints_attn,
            "det_m": det_m,
        }

    def forward(self, x, dataset, max_step=1):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset = dataset.dataset

        inputs, others = self.data_prepare(x, dataset)
        pred = self.predict(inputs, others, max_step=max_step)
        pred.update({"hands_feet_contact": inputs["contact"], "obj_floor_contact": others["obj_floor_contact"]})
        return pred