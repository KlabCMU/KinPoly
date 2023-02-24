"""
File: /agent_ar.py
Created Date: Thursday February 18th 2021
Author: Zhengyi Luo
-----
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----

"""
from kin_poly.utils.logger import create_logger
import joblib
import os.path as osp
import pdb
import sys
import glob
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from collections import defaultdict
import multiprocessing
import math
import time
import os
import torch
import wandb

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

from uhc.khrylib.models.mlp import MLP
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from uhc.khrylib.rl.agents import AgentPPO
from uhc.khrylib.rl.core import estimate_advantages
from uhc.khrylib.utils.torch import *
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.rl.core import LoggerRL
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.utils import get_eta_str

from kin_poly.utils.tools import fix_height
from kin_poly.envs.humanoid_ar_v1 import HumanoidAREnv
from kin_poly.models.policy_ar import PolicyAR
from kin_poly.utils.flags import flags
from kin_poly.data_loaders.statear_smpl_dataset import StateARDataset
from kin_poly.core.trajbatch_ego import TrajBatchEgo
from kin_poly.core.reward_function import reward_func
from kin_poly.utils.torch_ext import get_scheduler
from uhc.utils.tools import CustomUnpickler


class AgentAR(AgentPPO):

    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training

        self.setup_vars()
        self.setup_data_loader()
        self.setup_policy()
        self.setup_env()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            running_state=None,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.policy_specs["num_optim_epoch"],
            gamma=cfg.policy_specs["gamma"],
            tau=cfg.policy_specs["tau"],
            clip_epsilon=cfg.policy_specs["clip_epsilon"],
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.policy_specs["end_reward"],
            use_mini_batch=False,
            mini_batch_size=0,
        )
        if self.cfg.joint_controller:
            self.update_modules.append(self.env.cc_policy)
            self.sample_modules.append(self.env.cc_policy)
        self.train_init()

    def setup_vars(self):
        cfg = self.cfg
        self.wild = cfg.wild
        self.epoch = 0
        self.value_net = None
        self.cc_cfg = None
        self.cc_cfg_wild = None
        self.env = None
        self.env_wild = None
        self.freq_dict = None
        self.data_loader = None
        self.test_data_loaders = []
        self.expert_reward = None
        self.running_state = None
        self.optimizer_value = None
        self.optimizer_policy = None
        self.policy_net = None

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger.info("==========================Kin Poly===========================")
        self.logger.info(f"sampling temp: {cfg.policy_specs.get('sampling_temp', 0.5)}")
        self.logger.info(f"sampling freq: {cfg.policy_specs.get('sampling_freq', 0.5)}")
        self.logger.info(f"init_update_iter: {cfg.policy_specs.get('num_init_update', 3)}")
        self.logger.info(f"step_update_iter: {cfg.policy_specs.get('num_step_update', 10)}")
        self.logger.info(f"use_of: {cfg.use_of}")
        self.logger.info(f"use_action: {cfg.use_action}")
        self.logger.info(f"use_vel: {cfg.use_vel}")
        self.logger.info(f"use_context: {cfg.use_context}")
        self.logger.info(f"add_noise: {cfg.add_noise}")
        self.logger.info(f"Data file: {cfg.data_file}")
        self.logger.info(f"Feature Version: {cfg.use_of}")
        self.logger.info("============================================================")

    def setup_policy(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        data_sample = self.data_loader.sample_seq()
        data_sample = {k: v.to(device).clone().type(dtype) for k, v in data_sample.items()}
        self.policy_net = policy_net = PolicyAR(cfg, data_sample, device=device, dtype=dtype, ar_iter=cfg.ar_iter)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        state_dim = self.policy_net.state_dim
        action_dim = self.env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, self.cc_cfg.value_hsize, self.cc_cfg.value_htype))
        to_device(device, self.value_net)

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        """load CC model"""
        self.cc_cfg = cc_cfg = CC_Config(
            cfg_id=cfg.cc_cfg,
            base_dir="",
            create_dirs=False,
        )
        self.cc_cfg_wild = cc_cfg_wild = CC_Config(
            cfg_id=cfg.cc_cfg,
            base_dir="",
            create_dirs=False,
        )

        if cfg.wild:
            cc_cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml"
        else:
            cc_cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all_step.xml"
        cc_cfg_wild.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml"

        with torch.no_grad():
            data_sample = self.data_loader.sample_seq()
            data_sample = {k: v.to(device).clone().type(dtype) for k, v in data_sample.items()}
            context_sample = self.policy_net.init_context(data_sample)
        self.env = HumanoidAREnv(cfg, cc_cfg=cc_cfg, init_context=context_sample, mode="train", wild=cfg.wild)
        self.env.seed(cfg.seed)
        self.env_wild = HumanoidAREnv(
            cfg,
            cc_cfg=cc_cfg_wild,
            init_context=context_sample,
            mode="train",
            wild=True,
        )
        self.env_wild.seed(cfg.seed)

    def setup_optimizer(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if cfg.policy_specs["policy_optimizer"] == "Adam":
            self.optimizer_policy = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=cfg.policy_specs["policy_lr"],
                weight_decay=cfg.policy_specs["policy_weightdecay"],
            )
        else:
            self.optimizer_policy = torch.optim.SGD(
                self.policy_net.parameters(),
                lr=cfg.policy_specs["policy_lr"],
                momentum=cfg.policy_specs["policy_momentum"],
                weight_decay=cfg.policy_specs["policy_weightdecay"],
            )

        if cfg.policy_specs["value_optimizer"] == "Adam":
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(),
                lr=cfg.policy_specs["value_lr"],
                weight_decay=cfg.policy_specs["value_weightdecay"],
            )
        else:
            self.optimizer_value = torch.optim.SGD(
                self.value_net.parameters(),
                lr=cfg.policy_specs["value_lr"],
                momentum=cfg.policy_specs["policy_momentum"],
                weight_decay=cfg.policy_specs["value_weightdecay"],
            )

        self.scheduler_policy = get_scheduler(
            self.optimizer_policy,
            policy="lambda",
            nepoch_fix=self.cfg.num_epoch_fix,
            nepoch=self.cfg.num_epoch,
        )

        self.scheduler_value = get_scheduler(
            self.optimizer_value,
            policy="lambda",
            nepoch_fix=self.cfg.num_epoch_fix,
            nepoch=self.cfg.num_epoch,
        )

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        freq_path = osp.join(cfg.result_dir, "freq_dict.pt")
        try:
            self.freq_dict = ({k: [] for k in self.data_loader.takes} if not osp.exists(freq_path) else joblib.load(freq_path))
        except:
            print("error parsing freq_dict, using empty one")
            self.freq_dict = {k: [] for k in self.data_loader.takes}
        self.logger = create_logger(os.path.join(cfg.log_dir, "log.txt"))

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[cfg.policy_specs["reward_id"]]

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        log = info["log"]

        c_info = log.avg_c_info
        log_str = f"Ep: {self.epoch}\t {cfg.id} \tT_s {info['T_sample']:.2f}\t \
                    T_u { info['T_update']:.2f}\tETA {get_eta_str(self.epoch, cfg.policy_specs['max_iter_num'], info['T_total'])} \
                \texpert_R_avg {log.avg_c_reward:.4f} {np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=',')}\
                 \texpert_R_range ({log.min_c_reward:.4f}, {log.max_c_reward:.4f})\teps_len {log.avg_episode_len:.2f}"

        self.logger.info(log_str)
        wandb.log(
            {
                "rewards": log.avg_c_info,
                "eps_len": log.avg_episode_len,
                "avg_rwd": log.avg_c_reward,
            },
            step=self.epoch,
        )

        if "log_eval" in info:
            [wandb.log(test, step=self.epoch) for test in info["log_eval"]]

    def per_epoch_update(self, epoch):
        self.scheduler_policy.step()
        self.scheduler_value.step()

    def optimize_policy(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.epoch = epoch
        t0 = time.time()
        self.per_epoch_update(epoch)
        batch, log = self.sample(cfg.policy_specs["min_batch_size"])

        if cfg.policy_specs["end_reward"]:
            self.env.end_reward = (log.avg_c_reward * cfg.policy_specs["gamma"] / (1 - cfg.policy_specs["gamma"]))
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            "log": log,
            "T_sample": t1 - t0,
            "T_update": t2 - t1,
            "T_total": t2 - t0,
        }

        if (self.epoch + 1) % cfg.policy_specs["save_model_interval"] == 0:
            log_eval = self.eval_policy("test")
            info["log_eval"] = log_eval

        if not cfg.no_log:
            self.log_train(info)
        joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def setup_data_loader(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.test_data_loaders = []
        self.data_loader = data_loader = StateARDataset(cfg, cfg.data, sim=True)
        self.test_data_loaders.append(StateARDataset(cfg, "test"))

        from kin_poly.utils.statear_smpl_config import Config

        cfg_wild = Config(
            "all",
            cfg.id,
            wild=True,
            create_dirs=False,
            mujoco_path="/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/%s.xml",
        )
        self.test_data_loaders.append(StateARDataset(cfg_wild, "test"))

    def load_checkpoint(self, i_iter):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if i_iter > 0:
            if self.wild:
                cp_path = "%s/iter_wild_%04d.p" % (cfg.policy_model_dir, i_iter)
            else:
                cp_path = "%s/iter_%04d.p" % (cfg.policy_model_dir, i_iter)

            if not osp.exists(cp_path):
                cp_path = "%s/iter_%04d.p" % (cfg.policy_model_dir, i_iter)

            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])

            # policy_net.old_arnet[0].load_state_dict(copy.deepcopy(policy_net.traj_ar_net.state_dict())) # ZL: should use the new old net as well

            self.value_net.load_state_dict(model_cp["value_dict"])
            # if self.cfg.joint_controller:
            #     self.env.cc_policy.load_state_dict(model_cp['cc_dict'])

            self.running_state = model_cp["running_state"]
            self.epoch = i_iter
            to_device(device, self.value_net)

    def save_checkpoint(self, i_iter):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        # self.tb_logger.flush()
        policy_net, value_net, running_state = (
            self.policy_net,
            self.value_net,
            self.running_state,
        )
        with to_cpu(policy_net, value_net):
            if self.wild:
                cp_path = "%s/iter_wild_%04d.p" % (cfg.policy_model_dir, i_iter + 1)
            else:
                cp_path = "%s/iter_%04d.p" % (cfg.policy_model_dir, i_iter + 1)

            model_cp = {
                "policy_dict": policy_net.state_dict(),
                "value_dict": value_net.state_dict(),
                "running_state": running_state,
            }
            if self.cfg.joint_controller:
                to_cpu(self.env.cc_policy)
                model_cp["cc_dict"] = self.env.cc_policy.state_dict()

            pickle.dump(model_cp, open(cp_path, "wb"))

    def train_init(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if self.epoch == 0:
            self.policy_net.update_init_supervised(
                self.cfg,
                self.data_loader,
                device=self.device,
                dtype=self.dtype,
                num_epoch=self.cfg.policy_specs.get("warm_update_init", 500),
            )
            self.policy_net.train_full_supervised(
                cfg,
                self.data_loader,
                device=self.device,
                dtype=self.dtype,
                scheduled_sampling=0.3,
                num_epoch=self.cfg.policy_specs.get("warm_update_full", 50),
            )
            self.policy_net.setup_optimizers()
            self.save_checkpoint(0)

    # def next_fit_seq(self):
    #     self.fit_ind += 1
    #     if self.fit_ind == self.data_loader.len:
    #         exit()
    #     context_sample = self.data_loader.get_seq_by_ind(
    #         self.fit_ind, full_sample=True)

    def eval_policy(self, data_mode="train"):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if data_mode == "train":
            data_loaders = [self.data_loader]
        elif data_mode == "test":
            data_loaders = self.test_data_loaders

        res_dicts = []
        for data_loader in data_loaders:
            coverage = 0
            num_jobs = self.num_threads
            jobs = list(range(data_loader.get_len()))
            np.random.shuffle(jobs)

            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]

            data_res_coverage = {}
            with to_cpu(*self.sample_modules):
                with torch.no_grad():
                    queue = multiprocessing.Queue()
                    for i in range(len(jobs) - 1):
                        worker_args = (jobs[i + 1], data_loader, queue)
                        worker = multiprocessing.Process(target=self.eval_seqs, args=worker_args)
                        worker.start()
                    res = self.eval_seqs(jobs[0], data_loader, None)
                    data_res_coverage.update(res)
                    for i in tqdm(range(len(jobs) - 1)):
                        res = queue.get()
                        data_res_coverage.update(res)

            for k, res in data_res_coverage.items():
                # print(res["percent"], data_loader.takes[k])
                if res["percent"] == 1:
                    coverage += 1
                    if data_mode == "train":
                        [self.freq_dict[data_loader.takes[k]].append([res["percent"], 0]) for _ in range(1)]
                else:
                    if data_mode == "train":
                        [self.freq_dict[data_loader.takes[k]].append([res["percent"], 0]) for _ in range(3)]

            eval_path = osp.join(cfg.result_dir, f"eval_dict_{data_mode}.pt")
            eval_dict = (joblib.load(eval_path) if osp.exists(eval_path) else defaultdict(list))
            eval_dict[self.epoch] = {data_loader.takes[k]: v["percent"] for k, v in data_res_coverage.items()}
            joblib.dump(eval_dict, eval_path)
            self.logger.info(f"Coverage {data_mode} of {coverage} out of {data_loader.get_len()}")

            res_bool = np.array([res["percent"] == 1 for k, res in data_res_coverage.items()])
            res_dicts.append({f"coverage_{data_loader.name}": {
                "mean_coverage": np.mean(res_bool),
                "num_coverage": np.sum(res_bool),
                "all_coverage": len(res_bool),
            }})

        return res_dicts

    def eval_seqs(self, fit_ids, data_loader, queue):
        res = {}
        for cur_id in fit_ids:
            res[cur_id] = self.eval_seq(cur_id, data_loader)

        if queue == None:
            return res
        else:
            queue.put(res)

    def eval_cur_seq(self):
        return self.eval_seq(self.fit_ind, self.data_loader)

    def eval_seq(self, fit_ind, loader):
        curr_env = self.env if not loader.cfg.wild else self.env_wild
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                self.policy_net.set_mode("test")
                curr_env.set_mode("test")

                context_sample = loader.get_seq_by_ind(fit_ind, full_sample=True)
                ar_context = self.policy_net.init_context(context_sample)

                curr_env.load_context(ar_context)
                state = curr_env.reset()

                if self.running_state is not None:
                    state = self.running_state(state)
                for t in range(10000):
                    res["target"].append(curr_env.target["qpos"])
                    res["pred"].append(curr_env.get_humanoid_qpos())
                    res["obj_pose"].append(curr_env.get_obj_qpos())

                    state_var = tensor(state).unsqueeze(0)
                    trans_out = self.trans_policy(state_var)

                    action = self.policy_net.select_action(trans_out, mean_action=True)[0].numpy()
                    action = (int(action) if self.policy_net.type == "discrete" else action.astype(np.float64))
                    next_state, env_reward, done, info = curr_env.step(action)

                    # c_reward, c_info = self.custom_reward(curr_env, state, action, info)
                    # res['reward'].append(c_reward)
                    # curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}
                        # print(info['percent'], ar_context['ar_qpos'].shape[1], loader.curr_key, np.mean(res['reward']))
                        res["percent"] = info["percent"]
                        return res
                    state = next_state

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls()
        self.policy_net.set_mode("test")
        self.env.set_mode("train")
        freq_dict = defaultdict(list)

        while logger.num_steps < min_batch_size:
            context_sample = self.data_loader.sample_seq(
                freq_dict=self.freq_dict,
                sampling_temp=self.cfg.policy_specs.get("sampling_temp", 0.5),
                sampling_freq=self.cfg.policy_specs.get("sampling_freq", 0.9),
            )
            self.data_loader.curr_key

            np.random.random(5)
            # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = self.cfg.policy_specs.get("sampling_temp", 0.5), sampling_freq = self.cfg.policy_specs.get("sampling_freq", 0.9), full_sample = True if self.data_loader.get_seq_len(self.fit_ind) < 1000 else False)
            # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = 0.5)
            # context_sample = self.data_loader.sample_seq()

            # should not try to fix the height during training!!!
            ar_context = self.policy_net.init_context(context_sample, fix_height=False)

            self.env.load_context(ar_context)
            state = self.env.reset()

            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or (self.cfg.policy_specs.get("dagger", False) and self.epoch % 2 == 0)

                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()

                action = (int(action) if self.policy_net.type == "discrete" else action.astype(np.float64))
                #################### ZL: Jank Code.... ####################
                gt_qpos = self.env.ar_context["qpos"][self.env.cur_t + 1]
                curr_qpos = self.env.get_humanoid_qpos()
                #################### ZL: Jank Code.... ####################

                next_state, env_reward, done, info = self.env.step(action)
                res_qpos = self.env.get_humanoid_qpos()

                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                if flags.debug:
                    np.set_printoptions(precision=4, suppress=1)
                    print(c_reward, c_info)

                # add end reward
                if self.end_reward and info.get("end", False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                # exp = 1 - mean_action # ZL: mean_action is used, but still valid
                exp = 1
                self.push_memory(
                    memory,
                    state,
                    action,
                    mask,
                    next_state,
                    reward,
                    exp,
                    gt_qpos,
                    curr_qpos,
                    res_qpos,
                    info["cc_action"],
                    info["cc_state"],
                )

                if pid == 0 and self.render:
                    self.env.render()
                if done:
                    freq_dict[self.data_loader.curr_key].append([info["percent"], self.data_loader.fr_start])
                    # print(self.data_loader.curr_key, info['percent'])
                    break

                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict

    def push_memory(
        self,
        memory,
        state,
        action,
        mask,
        next_state,
        reward,
        exp,
        gt_target_qpos,
        curr_qpos,
        res_qpos,
        cc_action,
        cc_state,
    ):
        v_meta = np.array([
            self.data_loader.curr_take_ind,
            self.data_loader.fr_start,
            self.data_loader.fr_num,
        ])
        memory.push(
            state,
            action,
            mask,
            next_state,
            reward,
            exp,
            v_meta,
            gt_target_qpos,
            curr_qpos,
            res_qpos,
            cc_action,
            cc_state,
        )

    # def push_memory(self, memory, state, action, mask, next_state, reward, exp):
    #     v_meta = np.array([self.data_loader.curr_take_ind, self.data_loader.fr_start, self.data_loader.fr_num])
    #     memory.push(state, action, mask, next_state, reward, exp, v_meta)
    def sample(self, min_batch_size):
        t_start = time.time()
        self.pre_sample()
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads - 1):
                    worker_args = (i + 1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0], freq_dict = self.sample_worker(0, None, thread_batch_size)
                self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger, freq_dict = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger

                    self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                self.freq_dict = {k: v if len(v) < 5000 else v[-5000:] for k, v in self.freq_dict.items()}
                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = TrajBatchEgo(memories)
                logger = self.logger_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = torch.from_numpy(batch.v_metas).to(self.dtype).to(self.device)
        gt_target_qpos = (torch.from_numpy(batch.gt_target_qpos).to(self.dtype).to(self.device))
        curr_qpos = torch.from_numpy(batch.curr_qpos).to(self.dtype).to(self.device)
        res_qpos = torch.from_numpy(batch.res_qpos).to(self.dtype).to(self.device)
        cc_actions = torch.from_numpy(batch.cc_action).to(self.dtype).to(self.device)
        cc_states = torch.from_numpy(batch.cc_state).to(self.dtype).to(self.device)

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states[:, :self.policy_net.state_dim]))

        self.policy_net.set_mode("train")
        self.policy_net.initialize_rnn((masks, v_metas))
        """get advantage estimation from the trajectories"""
        print(f"==========================Epoch: {self.epoch}=======================>")

        if not self.cfg.policy_specs.get("grad_joint", False):
            if self.cfg.policy_specs.get("rl_update", False):
                advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)
                self.update_policy(states, actions, returns, advantages, exps)

            if self.cfg.policy_specs.get("init_update", False):
                self.policy_net.update_init_supervised(
                    self.cfg,
                    self.data_loader,
                    device=self.device,
                    dtype=self.dtype,
                    num_epoch=int(self.cfg.policy_specs.get("num_init_update", 5)),
                )

            if self.cfg.policy_specs.get("step_update", False):
                self.policy_net.update_supervised_step(
                    states,
                    gt_target_qpos,
                    curr_qpos,
                    num_epoch=int(self.cfg.policy_specs.get("num_step_update", 10)),
                )

            if self.cfg.policy_specs.get("step_update_dyna", False):
                self.policy_net.update_supervised_dyna(
                    states,
                    res_qpos,
                    curr_qpos,
                    num_epoch=int(self.cfg.policy_specs.get("num_step_dyna_update", 10)),
                )

            if self.cfg.policy_specs.get("full_update", False):
                self.policy_net.train_full_supervised(
                    self.cfg,
                    self.data_loader,
                    device=self.device,
                    dtype=self.dtype,
                    num_epoch=1,
                    scheduled_sampling=0.3,
                )
        else:
            advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)
            self.update_policy_joint(states, gt_target_qpos, curr_qpos, actions, returns, advantages, exps)

        if self.cfg.joint_controller:
            self.update_controller(cc_states, cc_actions, returns, advantages, exps)

        self.policy_net.step_lr()

        return time.time() - t0

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        pbar = tqdm(range(self.opt_num_epochs))
        for _ in pbar:
            ind = exps.nonzero(as_tuple=False).squeeze(1)
            self.update_value(states, returns)
            dist, action_mean, action_std = self.policy_net.forward(self.trans_policy(states)[ind])
            log_probs = dist.log_prob(actions[ind])
            surr_loss, ratio = self.ppo_loss(log_probs, advantages, fixed_log_probs, ind)
            self.optimizer_policy.zero_grad()
            surr_loss.backward()
            self.clip_policy_grad()
            self.optimizer_policy.step()
            pbar.set_description_str(f"PPO Loss: {surr_loss.cpu().detach().numpy():.3f}| Ration: {ratio.mean().cpu().detach().numpy():.3f}")

    def update_controller(self, states, actions, returns, advantages, exps):
        """update policy"""
        if self.cfg.joint_controller:
            to_device(self.device, self.env.cc_policy)
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.env.cc_policy.get_log_prob(self.trans_policy(states), actions)
        pbar = tqdm(range(self.opt_num_epochs))
        for _ in pbar:
            ind = exps.nonzero(as_tuple=False).squeeze(1)
            # self.update_value(states, returns)
            dist = self.env.cc_policy.forward(self.trans_policy(states)[ind])
            log_probs = dist.log_prob(actions[ind])
            surr_loss, ratio = self.ppo_loss(log_probs, advantages, fixed_log_probs, ind)
            self.optimizer_policy.zero_grad()
            surr_loss.backward()
            self.clip_policy_grad()
            self.optimizer_policy.step()
            pbar.set_description_str(f"PPO Loss-controller: {surr_loss.cpu().detach().numpy():.3f}| Ration: {ratio.mean().cpu().detach().numpy():.3f}")

    def update_policy_joint(self, states, target_qpos, curr_qpos, actions, returns, advantages, exps):
        """update policy"""
        # This is seperate
        if self.cfg.policy_specs.get("init_update", False):
            self.policy_net.update_init_supervised(
                self.cfg,
                self.data_loader,
                device=self.device,
                dtype=self.dtype,
                num_epoch=int(self.cfg.policy_specs.get("num_init_update", 5)),
            )

        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        pbar = tqdm(range(self.opt_num_epochs))
        for _ in pbar:
            ind = exps.nonzero(as_tuple=False).squeeze(1)
            self.update_value(states, returns)
            dist, action_mean, action_std = self.policy_net.forward(self.trans_policy(states)[ind])
            log_probs = dist.log_prob(actions[ind])
            if self.cfg.policy_specs.get("grad_alternate", False):
                if self.epoch % 2 == 1:
                    surr_loss, ratio = self.ppo_loss(log_probs, advantages, fixed_log_probs, ind)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
                    pbar.set_description_str(f"PPO Loss: {surr_loss.cpu().detach().numpy():.3f}| Ration: {ratio.mean().cpu().detach().numpy():.3f}")

                if self.epoch % 2 == 0:
                    self.policy_net.traj_ar_net.set_sim(curr_qpos)
                    next_qpos, _ = self.policy_net.traj_ar_net.step(action_mean)
                    loss, loss_idv = self.policy_net.traj_ar_net.compute_loss_lite(next_qpos, target_qpos)
                    self.policy_net.optimizer.zero_grad()
                    loss.backward()
                    self.policy_net.optimizer.step()
                    pbar.set_description_str(f"Per-step {self.epoch} loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.policy_net.scheduler.get_lr()[0]:.5f}")
            else:
                surr_loss, ratio = self.ppo_loss(log_probs, advantages, fixed_log_probs, ind)
                self.policy_net.traj_ar_net.set_sim(curr_qpos)
                if self.cfg.policy_specs.get("sl_ratio", False):
                    # This ratio is a bit problemtic. Well it's wrong.
                    next_qpos, _ = self.policy_net.traj_ar_net.step(actions[ind])
                    loss_step, loss_idv = self.policy_net.traj_ar_net.compute_loss_lite(next_qpos, target_qpos, return_mean=False)  ## BC loss, directly applied
                    loss_step = (ratio * loss_step).mean()
                else:
                    next_qpos, _ = self.policy_net.traj_ar_net.step(action_mean)
                    loss_step, loss_idv = self.policy_net.traj_ar_net.compute_loss_lite(next_qpos, target_qpos)
                loss = loss_step * 10 + surr_loss

                self.optimizer_policy.zero_grad()
                loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()
                pbar.set_description_str(f"Per-step {self.epoch} loss: {loss_step.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}]" +
                                         f" | PPO Loss: {surr_loss.cpu().detach().numpy():.3f}| Ratio: {ratio.mean().cpu().detach().numpy():.3f} | lr: {self.scheduler_policy.get_lr()[0]:.5f}")

    def ppo_loss(self, log_probs, advantages, fixed_log_probs, ind):
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = (torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages)
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss, ratio

    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            # trans_value = self.trans_value(states[:, :self.policy_net.obs_lim])
            trans_value = self.trans_value(states)

            values_pred = self.value_net(trans_value)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def action_loss(self, actions, gt_actions):
        pass
