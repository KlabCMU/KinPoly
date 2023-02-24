import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import sys
import pickle
import time
import math
import torch
import numpy as np

sys.path.append(os.getcwd())

import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from kin_poly.utils import *
from kin_poly.models.mlp import MLP
from kin_poly.models.traj_ar_smpl_net import TrajARNet
from kin_poly.data_loaders.statear_smpl_dataset import StateARDataset
from kin_poly.utils.torch_humanoid import Humanoid
from kin_poly.utils.torch_ext import get_scheduler
from kin_poly.utils.statear_smpl_config import Config
# from multiprocessing import Pool
from uhc.utils.tools import CustomUnpickler


def eval_sequences(cur_jobs):
    with torch.no_grad():
        traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)
        results = defaultdict(dict)
        pbar = tqdm(cur_jobs)
        for seq_key, data_dict in pbar:
            data_acc = defaultdict(list)
            # if args.cfg.startswith("wild"):
            #     from scripts.wild_meta import take_names
            #     start = take_names[seq_key]['start_idx']
            #     print(f"Wild start {start}")
            # action = seq_key.split("-")[0]
            # if action != "push":
            #     continue

            data_dict = {k: torch.from_numpy(v).to(device).type(dtype) for k, v in data_dict.items()}
            feature_pred = traj_ar_net.forward(data_dict)

            data_acc['qpos'] = feature_pred['qpos'][0].cpu().numpy()
            data_acc['qpos_gt'] = data_dict['qpos'][0].cpu().numpy()
            data_acc['obj_pose'] = data_dict['obj_pose'][0].cpu().numpy()

            # print(orig_qpos.shape)
            # if args.smooth:
            # from scipy.ndimage import gaussian_filter1d
            # pred_qpos[:, 7:] = gaussian_filter1d(pred_qpos[:, 7:], 1, axis = 0).copy()
            results[seq_key] = data_acc
            # results["pesudo_expert"][seq_key] = get_expert(results["traj_pred"][seq_key], 0, results["traj_pred"][seq_key].shape[0], cfg=cfg, env=env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--wild', action='store_true', default=False)
    args = parser.parse_args()

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'

    cfg = Config(args.action, args.cfg, wild=args.wild, create_dirs=(args.iter == 0), mujoco_path="assets/mujoco_models/%s.xml")
    """setup"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=args.gpu_index) if (torch.cuda.is_available() and args.mode == "train") else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    """Datasets"""
    if args.mode == 'train':
        dataset = StateARDataset(cfg, args.data)
    else:
        dataset = StateARDataset(cfg, args.data, sim=True)
    data_sample = dataset.sample_seq()
    data_sample = {k: v.clone().to(device).type(dtype) for k, v in data_sample.items()}
    """networks"""
    state_dim = dataset.traj_dim
    traj_ar_net = TrajARNet(cfg, data_sample=data_sample, device=device, dtype=dtype, mode=args.mode, as_policy=False)
    if args.iter > 0:
        cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp, meta = CustomUnpickler(open(cp_path, "rb")).load()
        # traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)

    traj_ar_net.to(device)
    curr_lr = cfg.lr
    optimizer = torch.optim.Adam(traj_ar_net.parameters(), lr=curr_lr, weight_decay=cfg.weightdecay)

    scheduler = get_scheduler(
        optimizer,
        policy="lambda",
        nepoch_fix=cfg.num_epoch_fix,
        nepoch=cfg.num_epoch,
    )

    fr_num_start = 80
    fr_num_end = 150
    if args.mode == 'train':
        traj_ar_net.train()
        for i_epoch in range(args.iter, cfg.num_epoch):
            sampling_rate = max((1 - i_epoch / cfg.num_epoch) * 0.3, 0)

            fr_num = fr_num_start + i_epoch / cfg.num_epoch * (fr_num_end - fr_num_start) // 5 * 5
            print(f"sampling_rate: {sampling_rate:.3f}, fr_num: {fr_num}, cfg: {args.cfg}")
            traj_ar_net.set_schedule_sampling(sampling_rate)
            t0 = time.time()
            losses = np.array(0)
            epoch_loss = 0
            generator = dataset.sampling_generator(num_samples=cfg.num_sample, batch_size=cfg.batch_size, num_workers=10, fr_num=fr_num)

            pbar = tqdm(generator)
            optimizer = torch.optim.Adam(traj_ar_net.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
            for data_dict in pbar:
                data_dict
                data_dict = {k: v.clone().to(device).type(dtype) for k, v in data_dict.items()}

                feature_pred = traj_ar_net.forward(data_dict)
                loss, loss_idv = traj_ar_net.compute_loss(feature_pred, data_dict)

                optimizer.zero_grad()
                loss.backward()  # Testing GT
                optimizer.step()  # Testing GT

                pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}]")
                epoch_loss += loss.cpu().item()
                losses = loss_idv + losses

            epoch_loss /= cfg.num_sample
            losses /= cfg.num_sample
            curr_lr = optimizer.param_groups[0]["lr"]
            logger.info(f'epoch {i_epoch:4d}    time {time.time() - t0:.2f}   loss {epoch_loss:.4f} {np.round(losses * 100, 4).tolist()} lr: {curr_lr} ')
            scheduler.step()

            if cfg.save_model_interval > 0 and (i_epoch + 1) % cfg.save_model_interval == 0:
                with to_cpu(traj_ar_net):
                    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_epoch + 1)
                    model_cp = {'stateAR_net_dict': traj_ar_net.state_dict()}
                    meta = {}
                    pickle.dump((model_cp, meta), open(cp_path, 'wb'))

    elif args.mode == 'test':
        start = 0
        counter = 0
        traj_ar_net.eval()
        traj_ar_net.set_schedule_sampling(0)

        jobs = list(dataset.iter_data().items())
        data_res_full = eval_sequences(jobs)
        num_jobs = 5

        res_path = '%s/iter_%04d_%s_%s.p' % (cfg.result_dir, args.iter, args.data, cfg.data_file)
        print(f"results dir: {res_path}")
        pickle.dump(data_res_full, open(res_path, 'wb'))

        if args.wild:
            os.system(f"python -m scripts.eval_pose_all --cfg {args.cfg} --iter {args.iter} --mode stats --wild")
            os.system(f"python -m scripts.eval_pose_all --cfg {args.cfg} --iter {args.iter} --mode vis --wild")
        else:
            stats_cmd = f"python -m scripts.eval_pose_all --cfg {args.cfg} --iter {args.iter} --mode stats"
            vis_cmd = f"python -m scripts.eval_pose_all --cfg {args.cfg} --iter {args.iter} --mode vis"
            print(stats_cmd)
            os.system(stats_cmd)
            print(vis_cmd)
            os.system(vis_cmd)
