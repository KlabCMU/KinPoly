import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
from collections import defaultdict
from tqdm import tqdm
import joblib
import numpy as np
import torch
import copy
import time

from kin_poly.utils.statear_smpl_config import Config
from uhc.khrylib.utils import *
from uhc.khrylib.rl.utils.visualizer import Visualizer
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from kin_poly.utils.flags import flags
from uhc.utils.tools import CustomUnpickler


class MyVisulizer(Visualizer):

    def __init__(self, vis_file):
        super().__init__(vis_file)
        ngeom = 24
        self.env_vis.model.geom_rgba[ngeom + 1:ngeom * 2] = np.array([0.7, 0.0, 0.0, 1])

        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = 45
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        with torch.no_grad():
            while True:
                results = defaultdict(list)
                policy_net.set_mode('test')
                env.set_mode('test')
                if args.ind != -1:
                    context_sample = data_loader.get_seq_by_ind(args.ind, full_sample=True)
                    ar_context = policy_net.init_context(context_sample)
                    env.load_context(ar_context)
                else:
                    context_sample = data_loader.iter_seq()
                    ar_context = policy_net.init_context(context_sample)
                    env.load_context(ar_context)

                state = env.reset()
                if running_state is not None:
                    state = running_state(state, update=False)
                results['fail_safe'] = False
                for t in range(1000000):

                    results['target'].append(env.target['qpos'])
                    # poses['target'].append(env.gt_targets['qpos'][env.cur_t])
                    results['pred'].append(env.get_humanoid_qpos())
                    results['obj_pose'].append(env.get_obj_qpos())
                    state_var = tensor(state, dtype=dtype).unsqueeze(0).to(device)
                    t_s = time.time()
                    mean_action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                    mean_action = int(mean_action) if policy_net.type == 'discrete' else mean_action.astype(np.float64)
                    next_state, reward, done, res = env.step(mean_action)
                    dt = time.time() - t_s
                    results['dt'].append(dt)
                    # env.render()

                    # rwd, rwd_acc = custom_reward(env, state, mean_action, res)
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(rwd, rwd_acc)
                    # results['reward'].append(rwd)

                    if running_state is not None:
                        next_state = running_state(next_state, update=False)
                    if done:
                        print("done", res['percent'], args.fail_safe)
                        if args.fail_safe and res['percent'] != 1:
                            env.ar_fail_safe()
                            results['fail_safe'] = True
                            print(f"Triggering fail safe at timestep {env.cur_t}: {data_loader.curr_key} | {res['percent']:.3f}")
                        else:
                            print(res['percent'], data_loader.curr_key, len(results['target']), np.sum(results['dt']), np.mean(results['dt']))
                            break
                    state = next_state

                # results ={k: np.vstack(v) for k, v in results.items()}

                self.num_fr = len(results['pred'])
                yield results

    def update_pose(self):
        self.env_vis.data.qpos[:76] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[76:152] = self.data['target'][self.fr]
        self.env_vis.data.qpos[76] = 100
        self.env_vis.data.qpos[152:] = self.data['obj_pose'][self.fr]

        # self.env_vis.data.qpos[env.model.nq] += 1.0
        # if args.record_expert:
        # self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        # if args.hide_expert:
        # self.env_vis.data.qpos[env.model.nq + 2] = 100.0
        self.env_vis.sim_forward()

    def display_coverage(self):
        res_dir = osp.join(cfg.result_dir, f"{args.iter:04d}_{cfg.data_file}_coverage_full.pkl")
        print(res_dir)
        data_res = joblib.load(res_dir)
        print(len(data_res))

        def vis_gen():
            keys = sorted(list(data_res.keys()))
            if (args.action != "all"):
                keys = [k for k in keys if k.startswith(args.action)]

            keys = [k for k in keys if data_res[k]['fail_safe'] == True]

            for k in keys:
                v = data_res[k]
                print(f"{k} {v['percent']:.3f} {v['fail_safe']}")
                self.num_fr = len(v['pred'])
                yield v

        self.data_gen = iter(vis_gen())
        self.show_animation()

    def display_fit(self):
        data_res = {}
        for take in data_loader.takes:
            curr_seq_res = osp.join(cfg.result_dir, f"{take}.pkl")
            if osp.exists(curr_seq_res):
                data_res[take] = joblib.load(curr_seq_res)

        def vis_gen():
            keys = sorted(list(data_res.keys()))
            for k in keys:
                v = data_res[k]
                print(f"{k} {v['percent']:.3f}")
                self.num_fr = len(v['pred'])
                yield v

        self.data_gen = iter(vis_gen())
        self.show_animation()

    def record_video(self):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(20):
                self.render()
            if not args.preview:
                t0 = time.time()
                save_screen_shots(self.env_vis.viewer.window, f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['/usr/local/bin/ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0', '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)


def run_seq(data_idx):
    results = {}
    # policy_net.load_state_dict(model_cp['policy_dict'])
    # value_net.load_state_dict(model_cp['value_dict'])
    # pbar = tqdm(data_idx)
    pbar = iter(data_idx)
    for idx in pbar:
        data_loader.set_seq_counter(idx)
        context_sample = data_loader.iter_seq()
        ar_context = policy_net.init_context(context_sample)
        env.load_context(ar_context)
        state = env.reset()
        cur_key = data_loader.curr_key
        seq_result = defaultdict(list)
        seq_result['fail_safe'] = False

        if running_state is not None:
            state = running_state(state, update=False)
        with torch.no_grad():
            for t in range(1000000):
                seq_result['target'].append(env.target['qpos'])
                seq_result['pred'].append(env.get_humanoid_qpos())
                seq_result['obj_pose'].append(env.get_obj_qpos())
                state_var = tensor(state, dtype=dtype).unsqueeze(0).clone()
                mean_action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                # values = value_net(state_var)[0].cpu().numpy()
                # seq_result['values'].append(values)
                next_state, reward, done, res = env.step(mean_action)

                if running_state is not None:
                    next_state = running_state(next_state, update=False)
                if done:
                    seq_result['percent'] = res['percent']

                    if args.fail_safe and res['percent'] != 1:
                        seq_result['fail_safe'] = True
                        env.ar_fail_safe()
                        print(f"Triggering fail safe at timestep {env.cur_t}: {data_loader.curr_key} | {res['percent']:.3f}")
                    else:
                        break

                    # pbar.set_description(f"{res['percent']} | {np.mean(seq_result['values']):.3f} | {cur_key}")
                state = next_state
        results[cur_key] = seq_result
    return results


def test_coverage():
    jobs = list(range(data_loader.get_len()))
    # jobs = list(range(10))
    # run_seq(jobs)

    from torch.multiprocessing import Pool
    coverage = 0
    data_res_full = {}
    data_res_coverage = {}
    num_jobs = args.num_threads
    chunk = np.ceil(len(jobs) / num_jobs).astype(int)
    jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i],) for i in range(len(jobs))]
    print(len(job_args))
    try:
        pool = Pool(num_jobs)  # multi-processing
        job_res = pool.starmap(run_seq, job_args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    [data_res_full.update(j) for j in job_res]
    for k, res in data_res_full.items():
        data_res_coverage[k] = {
            "percent": res['percent'],
            "values": res['values'],
            "fail_safe": res['fail_safe'],
        }
        print(f"{res['percent']} | {np.mean(res['values']):.3f} | fail_safe: {res['fail_safe']} | {k}")
        if res["percent"] == 1 and not res["fail_safe"]:
            coverage += 1

    print(f"Coverage of {coverage} out of {data_loader.get_len()}")
    res_dir = osp.join(cfg.result_dir, f"{args.iter:04d}_{cfg.data_file}_coverage.pkl")
    res_full_dir = osp.join(cfg.result_dir, f"{args.iter:04d}_{cfg.data_file}_coverage_full.pkl")
    print(res_dir)
    joblib.dump(data_res_coverage, res_dir)
    joblib.dump(data_res_full, res_full_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--iter', type=int, default=-1)
    parser.add_argument('--ar_iter', type=int, default=-1)
    parser.add_argument('--cc_cfg', type=str, default="uhc")
    parser.add_argument('--cc_iter', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='vis')
    parser.add_argument('--input', action='store_true', default=False)
    parser.add_argument('--num_threads', type=int, default=20)
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--wild', action='store_true', default=False)
    parser.add_argument('--ind', type=int, default=-1)
    parser.add_argument('--algo', type=str, default='statear')
    parser.add_argument('--fail_safe', action='store_true', default=False)
    parser.add_argument('--test_time', action='store_true', default=False)
    parser.add_argument('--ar_mode', action='store_true', default=False)

    args = parser.parse_args()

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'

    flags.debug = args.debug

    cc_cfg = CC_Config(
        cfg_id=args.cc_cfg,
        create_dirs=False,
        base_dir="",
    )
    if args.wild:
        cc_cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml"
    else:
        cc_cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all_step.xml"

    cfg = Config(args.action, args.cfg, create_dirs=(args.iter == 0), wild=args.wild, mujoco_path="assets/mujoco_models/%s.xml")
    np.random.seed(1)

    # print(cfg.takes)
    """make and seed env"""
    model = load_model_from_path(cc_cfg.mujoco_model_file)
    # if args.mode != "stats" :
    # sim = MjSim(model)
    # viewer = MjViewer(sim)

    from uhc.khrylib.rl.core.critic import Value
    from uhc.khrylib.models.mlp import MLP
    from kin_poly.envs.humanoid_ar_v1 import HumanoidAREnv
    from kin_poly.data_loaders.statear_smpl_dataset import StateARDataset
    from kin_poly.models.policy_ar import PolicyAR
    if args.wild:
        vis_file = "humanoid_smpl_neutral_mesh_all_vis.xml"
    else:
        vis_file = "humanoid_smpl_neutral_mesh_all_vis_step.xml"

    cc_cfg.env_start_first = True
    device = torch.device("cpu")
    from kin_poly.core.reward_function import reward_func
    custom_reward = reward_func[cfg.policy_specs['reward_id']]

    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.manual_seed(cc_cfg.seed)
    torch.set_grad_enabled(False)
    """Datasets"""
    data_loader = StateARDataset(cfg, args.data, sim=True)
    data_sample = data_loader.sample_seq()
    data_sample = {k: v.clone().to(device).type(dtype) for k, v in data_sample.items()}

    state_dim = data_loader.traj_dim
    policy_net = PolicyAR(cfg, data_sample, device=device, dtype=dtype, ar_iter=args.ar_iter)
    with torch.no_grad():
        context_sample = policy_net.init_context(data_sample)

    env = HumanoidAREnv(cfg, wild=args.wild, cc_cfg=cc_cfg, cc_iter=args.cc_iter, init_context=context_sample, mode="test", ar_mode=args.ar_mode)
    env.seed(cfg.seed)
    actuators = env.model.actuator_names
    state_dim = policy_net.state_dim
    action_dim = env.action_space.shape[0]
    running_state = None  # No running state for the ARNet!!!

    value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))

    if args.iter > -1:
        if args.wild:
            if args.test_time:
                cp_path = '%s/iter_wild_test_%04d.p' % (cfg.policy_model_dir, args.iter)
            else:
                cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)
        else:
            if args.test_time:
                cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)
            else:
                cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)

        # cp_path = f'{cfg.policy_model_dir}/iter_wild_test_7210.p'
        # cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)

        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = CustomUnpickler(open(cp_path, "rb")).load()

        policy_net.load_state_dict(model_cp['policy_dict'])
        # policy_net.old_arnet[0].load_state_dict(copy.deepcopy(policy_net.traj_ar_net.state_dict())) # ZL: should use the new old net as well

        value_net.load_state_dict(model_cp['value_dict'])
        running_state = None  # ARNet does not use running state

    to_device(device, policy_net, value_net)

    policy_net.set_mode('test')
    env.set_mode('test')

    if args.mode == "stats":
        test_coverage()
    else:
        # policy_net.load_state_dict(model_cp['policy_dict'])
        # value_net.load_state_dict(model_cp['value_dict'])
        policy_net.to(device)
        value_net.to(device)
        vis = MyVisulizer(vis_file)
        if args.mode == "vis":
            vis.show_animation()
        elif args.mode == "record":
            vis.record_video()
        elif args.mode == "disp_stats":
            vis.display_coverage()
        elif args.mode == "disp_fit":
            vis.display_fit()
