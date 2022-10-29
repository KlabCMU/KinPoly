import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import os
import sys
import pickle
import time
import math
import torch
import numpy as np

sys.path.append(os.getcwd())
import glob
import pdb
import os.path as osp
import yaml

sys.path.append(os.getcwd())

import cv2
import torch.utils.data as data
import torch
import joblib

sys.path.append(os.getcwd())
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from kin_poly.utils import *
from kin_poly.utils.compute_loss import DeepMimicLoss, TrajLoss
from kin_poly.models.mlp import MLP
from kin_poly.utils.torch_humanoid import Humanoid
from kin_poly.utils.flags import flags


class StateARDataset(data.Dataset):

    def __init__(self, cfg, data_mode, seed=0, sim=False):
        np.random.seed(seed)
        data_file = osp.join(cfg.data_dir, "features", f"{cfg.data_file}.p")

        print(f"Loading data: {data_file}")
        self.data_traj = joblib.load(data_file)
        self.name = cfg.data_file

        self.cfg = cfg
        self.rotrep = cfg.rotrep
        self.meta_id = meta_id = cfg.meta_id
        self.data_mode = data_mode
        self.fr_num = cfg.fr_num
        self.overlap = cfg.fr_margin * 2
        self.base_folder = cfg.data_dir
        self.of_folder = os.path.join(self.base_folder, "fpv_of")
        self.traj_folder = os.path.join(self.base_folder,
                                        "traj_norm")  # ZL:use traj_norm
        meta_file = osp.join(self.base_folder, "meta", f"{meta_id}.yml")
        self.meta = yaml.load(open(meta_file, "r"), Loader=yaml.FullLoader)
        self.msync = self.meta["video_mocap_sync"]
        self.dt = 1 / self.meta["capture"]["fps"]
        self.pose_only = cfg.pose_only
        self.base_rot = [0.7071, 0.7071, 0.0, 0.0]
        self.counter = 0
        # get take names
        if data_mode == "all" or self.cfg.wild:
            self.takes = self.cfg.takes["train"] + self.cfg.takes["test"]
        else:
            self.takes = self.cfg.takes[data_mode]

        # if self.cfg.action != "all":
        #     self.takes = [t for t in self.takes if t.startswith(self.cfg.action)]
        # self.takes = [k for k in self.takes if k.startswith("sit")] # Debugging......
        self.sim = sim
        self.preprocess_data(data_mode=data_mode)
        self.len = len(self.takes)

        if self.cfg.use_of:
            # data_file = osp.join(cfg.data_dir, "features", "of_feat_smpl_all.p")
            # print(f"Loading of data: {data_file}")
            # of_data = joblib.load(data_file)
            # self.of_data = {}
            # for curr_action in of_data.keys():
            #     action_of_data = of_data[curr_action]
            #     self.of_data.update({f"{curr_action}-{k}": v for k, v in action_of_data.items()})
            data_file = osp.join(cfg.data_dir, "features",
                                 f"{self.cfg.of_file}.p")
            print(f"Loading of data: {data_file}")
            self.of_data = joblib.load(data_file)

    def preprocess_data(self, data_mode="train"):

        data_all = defaultdict(list)
        pbar = tqdm(self.takes)
        for take in pbar:
            pbar.set_description_str(f"Loading data for {self.cfg.data_file}")
            curr_expert = self.data_traj[take]
            gt_qpos = curr_expert["qpos"]
            seq_len = gt_qpos.shape[0]

            if (self.data_mode == "train" and seq_len < self.fr_num
                    and not self.cfg.wild):
                continue

            of_files = curr_expert["of_files"]
            assert len(of_files) == seq_len

            # data that needs pre-processing
            traj_pos = self.get_traj_de_heading(gt_qpos)
            traj_root_vel = self.get_root_vel(gt_qpos)
            traj = np.hstack(
                (traj_pos,
                 traj_root_vel))  # Trajectory and trajectory root velocity
            # target_qvel = np.concatenate((curr_expert['qvel'][1:, :], curr_expert['qvel'][-2:-1, :]))
            target_qvel = curr_expert["qvel"]

            if data_mode == "train" and not self.cfg.wild:
                data_all["wbpos"].append(curr_expert["wbpos"])
                data_all["wbquat"].append(curr_expert["wbquat"])
                data_all["bquat"].append(curr_expert["bquat"])

            data_all["qvel"].append(target_qvel)
            data_all["target"].append(traj)
            data_all["qpos"].append(gt_qpos)
            data_all["head_vels"].append(curr_expert["head_vels"])
            data_all["head_pose"].append(curr_expert["head_pose"])
            data_all["action_one_hot"].append(curr_expert["action_one_hot"])
            data_all["obj_head_relative_poses"].append(
                curr_expert["obj_head_relative_poses"]
                [:, :7])  # Taking in only the first object's pose
            data_all["of_files"].append(of_files)

            # if data_mode == "train"
            data_all["obj_pose"].append(curr_expert["obj_pose"])
            # break

        if data == "train" or data == "all":
            all_traj = np.vstack(data_all["target"])

        self.traj_dim = data_all["target"][0].shape[1]
        self.freq_indices = []
        self.all_indices = []
        for i, traj in enumerate(data_all["target"]):
            self.freq_indices += [
                i for _ in range(
                    np.ceil(traj.shape[0] / self.fr_num).astype(int))
            ]
            self.all_indices.append(i)

        self.freq_indices = np.array(self.freq_indices)

        self.data = data_all

    def get_traj_de_heading(self, orig_traj):
        # Remove trejectory-heading + remove horizontal movements
        # results: 57 (-2 for horizontal movements)
        # Contains deheaded root orientation
        if (
                self.cfg.has_z
        ):  # has z means that we are predicting the z directly from the model
            traj_pos = orig_traj[:, 2:].copy()  # qpos without x, y
            # traj_pos[:, 5:] = np.concatenate(
            # (traj_pos[1:, 5:], traj_pos[-2:-1, 5:])
            # )  # body pose 1 step forward for autoregressive target
            # traj_pos[:, 0] = np.concatenate(
            # (traj_pos[1:, 0], traj_pos[-2:-1, 0])
            # )  # z 1 step forward for autoregressive target
            # ZL: why does rotation not have one step forward???
            # ZL: because we do not use it.

            for i in range(traj_pos.shape[0]):
                # traj_pos[i, 1:5] = self.remove_base_rot(traj_pos[i, 1:5])
                traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
        else:  # does not have z means that we are getting the z from the GT and finite-integrate
            traj_pos = orig_traj[:, 3:].copy()  # qpos without x, y, z
            traj_pos[:, 4:] = np.concatenate(
                (traj_pos[1:, 4:], traj_pos[-2:-1, 4:]
                 ))  # body pose 1 step forward for autoregressive target
            for i in range(traj_pos.shape[0]):
                traj_pos[i, :4] = de_heading(traj_pos[i, :4])

        return traj_pos

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def get_root_vel(self, orig_traj):
        # Get root velocity: 1x6
        traj_root_vel = []
        for i in range(orig_traj.shape[0] - 1):
            # vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], self.dt, 'heading')
            curr_qpos = orig_traj[i, :]
            next_qpos = orig_traj[i + 1, :]
            v = (next_qpos[:3] - curr_qpos[:3]) / self.dt

            # curr_qpos[3:7] = self.remove_base_rot(curr_qpos[3:7])
            v = transform_vec(v, curr_qpos[3:7], "heading")
            qrel = quaternion_multiply(next_qpos[3:7],
                                       quaternion_inverse(curr_qpos[3:7]))
            axis, angle = rotation_from_quaternion(qrel, True)

            if angle > np.pi:  # -180 < angle < 180
                angle -= 2 * np.pi  #
            elif angle < -np.pi:
                angle += 2 * np.pi

            rv = (axis * angle) / self.dt
            rv = transform_vec(rv, curr_qpos[3:7], "root")

            traj_root_vel.append(np.concatenate((v, rv)))

        traj_root_vel.append(
            traj_root_vel[-1].copy()
        )  # copy last one since there will be one less through finite difference
        traj_root_vel = np.vstack(traj_root_vel)
        return traj_root_vel

    def load_of(self, of_files):
        ofs = []
        for of_file in of_files:
            of_i = np.load(of_file)
            if self.cfg.augment and self.data_mode == "train":
                of_i = self.augment_flow(of_i)
            ofs.append(of_i)
        ofs = np.stack(ofs)

        return ofs

    def random_crop(self, image, crop_size=(224, 224)):
        h, w, _ = image.shape
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        image = image[top:bottom, left:right, :]
        return image

    def augment_flow(self, flow):
        from scipy.ndimage.interpolation import rotate
        """Random scaling/cropping"""
        scale_size = np.random.randint(*(230, 384))
        flow = cv2.resize(flow, (scale_size, scale_size))
        flow = self.random_crop(flow)
        """Random gaussian noise"""
        flow += np.random.normal(loc=0.0, scale=1.0,
                                 size=flow.shape).reshape(flow.shape)
        return flow

    def __getitem__(self, index):
        # sample random sequence from data
        take_ind = self.sample_indices[index]
        sample = self.get_sample_from_take_ind(take_ind, fr_start=-1)
        sample["obj_pose"] = sample["obj_pose"][:, :
                                                7]  ## ZL: kinda of a janke fix
        return sample

    def get_seq_len(self, index):
        return self.data["qpos"][index].shape[0]

    def get_seq_key(self, index):
        return self.takes[index]

    def get_sample_from_take_ind(self,
                                 take_ind,
                                 full_sample=False,
                                 fr_start=0):
        self.curr_key = self.takes[take_ind]
        self.curr_take_ind = take_ind
        if full_sample:
            self.fr_start = fr_start = 0
            self.fr_end = fr_end = self.data["qpos"][take_ind].shape[0]
        elif fr_start != -1:
            self.fr_start = fr_start
            self.fr_end = fr_end = fr_start + self.fr_num
        else:
            seq_len = self.get_seq_len(take_ind)
            self.fr_start = fr_start = np.random.randint(
                0, seq_len - self.fr_num)
            self.fr_end = fr_end = fr_start + self.fr_num

        data_return = {}

        if self.cfg.use_of:
            # data_return['of'] = self.load_of(self.data['of_files'][take_ind][fr_start: fr_end])
            data_return["of"] = self.of_data[self.curr_key][fr_start:fr_end]

        for k in self.data.keys():
            if k in ["of_files"]:
                continue
            data_return[k] = self.data[k][take_ind][fr_start:fr_end]

        return data_return

    def sample_seq(
        self,
        num_samples=1,
        batch_size=1,
        use_freq=True,
        freq_dict=None,
        full_sample=False,
        sampling_temp=0.5,
        sampling_freq=0.9,
    ):
        start_idx = 0
        if use_freq:
            if freq_dict is None:
                self.ind = ind = np.random.choice(self.freq_indices)
            else:
                init_probs = np.exp(-np.array([
                    ewma(np.array(freq_dict[k])[:, 0] == 1)
                    if len(freq_dict[k]) > 0 else 0 for k in freq_dict.keys()
                ]) / sampling_temp)
                init_probs = init_probs / init_probs.sum()
                self.ind = ind = (np.random.choice(self.all_indices,
                                                   p=init_probs)
                                  if np.random.binomial(1, sampling_freq) else
                                  np.random.choice(self.all_indices))
                chosen_key = self.takes[ind]
                seq_len = self.get_seq_len(ind)

                ####################
                # perfs = np.array(freq_dict[chosen_key])
                # if len(perfs) > 0 and len(perfs[perfs[:, 0] != 1][:, 1]) > 0 and np.random.binomial(1, sampling_freq) and not full_sample:
                #     perfs = perfs[perfs[:, 0] != 1][:, 1]
                #     chosen_idx = np.random.choice(perfs)
                #     start_idx = np.random.randint(max(chosen_idx- 30, 0), min(chosen_idx + 30, seq_len - self.fr_num))
                #     # print(start_idx, chosen_key)
                # elif not full_sample:
                #     start_idx = np.random.randint(0, seq_len - self.fr_num)
                # else:
                #     start_idx = 0

                ####################
                if full_sample:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, seq_len - self.fr_num)

        else:
            self.ind = ind = np.random.choice(self.all_indices)
        # self.ind = ind = self.takes.index('step-08-30-2020-17-24-00')

        data_dict = self.get_sample_from_take_ind(take_ind=ind,
                                                  fr_start=start_idx,
                                                  full_sample=full_sample)
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}

    def get_seq_by_ind(self, ind, full_sample=False):
        data_dict = self.get_sample_from_take_ind(take_ind=ind,
                                                  full_sample=full_sample)
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}

    def get_len(self):
        return len(self.takes)

    def set_seq_counter(self, idx):
        self.counter = idx

    def iter_seq(self):
        take_ind = self.counter % len(self.takes)
        self.curr_key = curr_key = self.takes[take_ind]
        seq_len = self.data["qpos"][take_ind].shape[
            0]  # not using the fr_num at allllllllll
        data_return = {}
        self.counter += 1
        if self.cfg.use_of:
            # data_return['of'] = self.load_of(self.data['of_files'][take_ind])
            data_return["of"] = self.of_data[self.curr_key]

        for k in self.data.keys():
            if k in ["of_files"]:
                continue
            data_return[k] = self.data[k][take_ind]

        return {k: torch.from_numpy(v)[None, ] for k, v in data_return.items()}

    def sampling_generator(self,
                           batch_size=8,
                           num_samples=5000,
                           num_workers=1,
                           fr_num=80):
        self.fr_num = int(fr_num)
        self.iter_method = "sample"
        self.data_curr = [
            i for i in self.freq_indices
            if self.data["qpos"][i].shape[0] >= fr_num
        ]
        self.sample_indices = np.random.choice(self.data_curr,
                                               num_samples,
                                               replace=True)

        self.data_len = len(self.sample_indices)  # Change sequence length

        loader = torch.utils.data.DataLoader(self,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
        return loader

    def __len__(self):
        return self.data_len

    def iter_data(self):
        data = {}
        for take_ind in range(len(self.takes)):
            self.curr_key = curr_key = self.takes[take_ind]
            seq_len = self.data["qpos"][take_ind].shape[
                0]  # not using the fr_num at allllllllll
            data_return = {}
            if self.cfg.use_of:
                # data_return['of'] = self.load_of(self.data['of_files'][take_ind])
                data_return["of"] = self.of_data[self.curr_key]

            for k in self.data.keys():
                if k in ["of_files"]:
                    continue
                data_return[k] = self.data[k][take_ind]

            data_return["cl"] = np.array([0, seq_len])
            data[curr_key] = {
                k: torch.from_numpy(v)[None, ]
                for k, v in data_return.items()
            }
        return data

    def get_data(self):
        return self.data
