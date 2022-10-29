import torch.nn as nn
import torch
import pickle

from tqdm import tqdm

from uhc.khrylib.rl.core.distributions import DiagGaussian
from uhc.khrylib.rl.core.policy import Policy
from uhc.khrylib.utils.math import *
from uhc.khrylib.models.mlp import MLP
from kin_poly.models.rnn import RNN
from kin_poly.models.traj_ar_smpl_net import TrajARNet
from kin_poly.utils.flags import flags
import copy
from scipy.ndimage import gaussian_filter1d
from kin_poly.utils.torch_ext import get_scheduler


class PolicyAR(Policy):
    def __init__(self, cfg, data_sample, device, dtype, ar_iter = -1, mode = "test"):
        super().__init__()
        self.cfg = cfg
        self.policy_v = self.cfg.policy_specs['policy_v']
        self.device = device
        self.dtype = dtype
        self.type = 'gaussian'
        fix_std = cfg.policy_specs['fix_std']
        log_std = cfg.policy_specs['log_std']

        traj_ar_net_old = TrajARNet(cfg, data_sample = data_sample, device = device, dtype = dtype, mode = mode, as_policy=True)
        self.old_arnet = [traj_ar_net_old]

        if self.policy_v == 1:
            self.action_dim = action_dim = 80
            self.state_dim = state_dim = traj_ar_net_old.state_dim
            self.traj_ar_net = TrajARNet(cfg, data_sample = data_sample, device = device, dtype = dtype, mode = mode, as_policy=True)
            self.setup_optimizers()
            
        elif self.policy_v == 2:
            self.action_dim = action_dim = 76
            self.state_dim = state_dim = traj_ar_net_old.state_dim + action_dim
            self.htype = htype = self.cfg.policy_specs.get("mlp_htype", "relu")
            self.rnn_hdim = rnn_hdim = self.cfg.policy_specs.get("rnn_hdim", 512)
            self.mlp_hsize = mlp_hsize = self.cfg.policy_specs.get("mlp_hsize", [512, 256])
            self.rnn_type = rnn_type = self.cfg.policy_specs.get("rnn_type", "gru")

            self.action_rnn = RNN(state_dim , rnn_hdim, rnn_type)
            self.action_rnn.set_mode('step')
            self.action_mlp = MLP(rnn_hdim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.action_dim)

        if ar_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cfg.model_dir, ar_iter)
            print('loading arnet model from checkpoint: %s' % cp_path)
            model_cp, meta = pickle.load(open(cp_path, "rb"))
            traj_ar_net_old.load_state_dict(model_cp['stateAR_net_dict'], strict=True)
            if self.policy_v == 1: self.traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)
        
        self.to(device)
        self.obs_lim = self.old_arnet[0].get_obs(data_sample, 0)[0].shape[1]

        # self.qpos_lim = [101, 177]
        self.mode = mode
        print("--------------------------------")
        print(f"Policy v: {self.policy_v}")
        print(f"Mode: {self.mode}")
        print(f"smooth: {self.cfg.smooth}")
        print("--------------------------------")

    def setup_optimizers(self):
        optim = self.cfg.get("optimizer", "Adam")
        if optim == "Adam":
            print("Using Adam")
            self.optimizer = torch.optim.Adam(self.traj_ar_net.parameters(), lr=self.cfg.lr)
        elif optim == "SGD":
            print("Using SGD")
            self.optimizer = torch.optim.SGD(self.traj_ar_net.parameters(), lr=self.cfg.lr)
        else:
            raise NotImplementedError
        
        self.scheduler = get_scheduler(
                                self.optimizer,
                                policy="lambda",
                                nepoch_fix=self.cfg.num_epoch_fix,
                                nepoch=self.cfg.num_epoch)
    def step_lr(self):
        self.scheduler.step()

    def to(self, device):
        # ZL: annoying, need fix
        self.device = device
        if self.policy_v == 1:
            self.traj_ar_net.to(device)
        elif self.policy_v == 2:
            pass
        [net.to(device) for net in self.old_arnet]

        super().to(device)
        return self
 

    def initialize_rnn(self, data):
        masks, v_metas = data
        device, dtype = masks.device, masks.dtype

        end_indice = np.where(masks.cpu().numpy() == 0)[0]
        v_metas = v_metas[end_indice, :]
        num_episode = len(end_indice)
        end_indice = np.insert(end_indice, 0, -1)
        max_episode_len = int(np.diff(end_indice).max())
        self.num_episode = num_episode
        self.max_episode_len = max_episode_len
        self.indices = np.arange(masks.shape[0])
        for i in range(1, num_episode):
            start_index = end_indice[i] + 1
            end_index = end_indice[i + 1] + 1
            self.indices[start_index:end_index] += i * max_episode_len - start_index
       
        self.s_scatter_indices = torch.LongTensor(np.tile(self.indices[:, None], (1, self.state_dim))).to(device)
        self.s_gather_indices = torch.LongTensor(np.tile(self.indices[:, None], (1, self.action_dim))).to(device)

    def init_context(self, data_dict, fix_height = True):
        with torch.no_grad():
            if self.policy_v == 1:
                ar_context = self.traj_ar_net.init_states(data_dict)
                feature_pred = self.traj_ar_net.forward(data_dict)
            else:
                ar_context = self.old_arnet[0].init_states(data_dict)
                feature_pred = self.old_arnet[0].forward(data_dict)
            # ar_context = self.old_arnet[0].init_states(data_dict)
            # feature_pred = self.old_arnet[0].forward(data_dict)

            ar_context['ar_qpos'] = feature_pred["qpos"].clone()
            ar_context['ar_qvel'] = feature_pred["qvel"].clone()
            begin_feet_offset = 0.01

            if self.cfg.smooth:
                if fix_height: # my brother, training mode can't do this.... it will be intialized in the air!!!
                    # print("fixing the height!!!")
                    ar_init_qpos = ar_context['init_qpos'].clone()
                    fk_res = self.old_arnet[0].fk_model.qpos_fk(ar_init_qpos)
                    wbpos = fk_res['wbpos']
                    begin_feet = min(wbpos[0, 4, 2],  wbpos[0, 8, 2])
                    begin_feet -=  begin_feet_offset# Hypter parameter to tune
                    ar_init_qpos[:, 2] -= begin_feet
                    ar_context['init_qpos'] = ar_init_qpos

                
                ar_qpos = ar_context['ar_qpos']
                ar_qpos[:, 7:] = torch.from_numpy(gaussian_filter1d(ar_qpos[:, 7:].cpu(), 1, axis = 0))
                if fix_height:
                    fk_res = self.old_arnet[0].fk_model.qpos_fk(ar_qpos[0])
                    wbpos = fk_res['wbpos']
                    begin_feet = min(wbpos[0, 4, 2],  wbpos[0, 8, 2])
                    begin_feet -= begin_feet_offset # Hypter parameter to tune
                    ar_qpos[0, :, 2] -= begin_feet

                fk_res = self.old_arnet[0].fk_model.qpos_fk(ar_qpos[0])
                ar_context['ar_qpos'] = ar_qpos
                ar_context['ar_wbpos'] = fk_res["wbpos"][None, ].clone()
                ar_context['ar_wbquat'] = fk_res["wbquat"][None, ].clone()
                ar_context['ar_bquat'] = fk_res["bquat"][None, ].clone()

                if flags.debug:
                    print("smoothing!!!")
                
            else:
                ar_context['ar_wbpos'] = feature_pred["pred_wbpos"].clone()
                ar_context['ar_wbquat'] = feature_pred["pred_wbquat"].clone()
                ar_context['ar_bquat'] = feature_pred["pred_rot"].clone()

            self.ar_context = ar_context
            self.count = 0

            if self.policy_v == 2:
                self.action_rnn.initialize(1)
            elif self.policy_v == 1:
                self.traj_ar_net.action_rnn.initialize(1)

            return ar_context

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, all_state):
        if self.policy_v == 1:
            return self.traj_ar_net.get_action(all_state)
        elif self.policy_v == 2:
            action_ar = all_state[:, -self.action_dim:]
            x = self.action_rnn(all_state)
            x = self.action_mlp(x)
            action_delta = self.action_fc(x)
            if flags.debug:
                action_delta = torch.zeros(action_delta.shape)
                
            return action_delta + action_ar
        

    def forward(self, all_state):
        # ZL: I don't like this...
        # ZL: not handling the objects well
        if self.mode == "test":
            with torch.no_grad():
                # if flags.debug:
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(self.traj_ar_net.sim['qpos'][0, :10].cpu().numpy())
                action_mean = self.get_action(all_state)
                    
            self.count += 1 # Debug

            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        elif self.mode == "train":
            device, dtype  = all_state.device, all_state.dtype
            s_ctx = torch.zeros((self.num_episode * self.max_episode_len, self.state_dim), device=device)
            s_ctx.scatter_(0, self.s_scatter_indices, all_state)
            s_ctx = s_ctx.view(-1, self.max_episode_len, self.state_dim).transpose(0, 1).contiguous()
            s_ctx_states = s_ctx
            batch_size = s_ctx_states.shape[1]
            if self.policy_v == 1:
                self.traj_ar_net.action_rnn.initialize(batch_size)
            elif self.policy_v == 2:
                self.action_rnn.initialize(batch_size)

            action_mean_acc = []
            for i in range(s_ctx_states.shape[0]):
                curr_state = s_ctx_states[i] # ZL: here is two days of debugging.......
                action_ar = self.get_action(curr_state)
                action_mean_acc.append(action_ar)
            action_mean_acc = torch.stack(action_mean_acc).transpose(0, 1).contiguous().view(-1, action_mean_acc[0].shape[-1])
            action_mean = torch.gather(action_mean_acc, 0, self.s_gather_indices)
                

            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return DiagGaussian(action_mean, action_std), action_mean, action_std


    def train_full_supervised(self, cfg, data_loader, device, dtype, num_epoch = 20, scheduled_sampling = 0):
        pbar = tqdm(range(num_epoch))
        self.traj_ar_net.set_schedule_sampling(scheduled_sampling)
        for _ in pbar:
            generator = data_loader.sampling_generator(num_samples= cfg.num_sample, batch_size=cfg.batch_size, num_workers=10, fr_num=self.cfg.fr_num)
            for data_dict in generator:
                data_dict = {k:v.clone().to(device).type(dtype) for k, v in data_dict.items()}
                feature_pred = self.traj_ar_net.forward(data_dict)
                loss, loss_idv = self.traj_ar_net.compute_loss(feature_pred, data_dict)
                
                self.optimizer.zero_grad()
                loss.backward()   
                self.optimizer.step()  
                pbar.set_description_str(f"Full loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_lr()[0]:.5f}")
            self.scheduler.step()
        self.traj_ar_net.set_schedule_sampling(0)


    def update_init_supervised(self, cfg, data_loader, device, dtype, num_epoch = 20):
        pbar = tqdm(range(num_epoch))
        for _ in pbar:
            generator = data_loader.sampling_generator(num_samples= cfg.num_sample, batch_size=cfg.batch_size, num_workers=10, fr_num=self.cfg.fr_num)
            
            for data_dict in generator:
                data_dict = {k:v.clone().to(device).type(dtype) for k, v in data_dict.items()}
                data_dict = self.traj_ar_net.init_states(data_dict)
                pred_qpos, pred_qvel, gt_qpos, gt_qvel = data_dict['init_qpos'], data_dict['init_qvel'], data_dict['qpos'][:, 0, :], data_dict['qvel'][:, 0, :]
                loss, loss_idv = self.traj_ar_net.compute_loss_init(pred_qpos, gt_qpos, pred_qvel, gt_qvel)
                self.optimizer.zero_grad()
                loss.backward()   # Testing GT
                self.optimizer.step()  # Testing GT
                pbar.set_description_str(f"Init loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_lr()[0]:.5f}")
            

    def update_supervised_step(self, all_state, target_qpos, curr_qpos, num_epoch = 20):
        pbar = tqdm(range(num_epoch) )
        for _ in pbar:
            _, action_mean, _ = self.forward(all_state)
            self.traj_ar_net.set_sim(curr_qpos)
            next_qpos, _ = self.traj_ar_net.step(action_mean)
            loss, loss_idv = self.traj_ar_net.compute_loss_lite(next_qpos, target_qpos)
            self.optimizer.zero_grad()
            loss.backward()   # Testing GT
            self.optimizer.step()  # Testing GT
            pbar.set_description_str(f"Per-step loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_lr()[0]:.5f}")

    def update_supervised_dyna(self, all_state, target_qpos, curr_qpos, num_epoch = 20):
        pbar = tqdm(range(num_epoch) )
        for _ in pbar:
            _, action_mean, _ = self.forward(all_state)

            self.traj_ar_net.set_sim(curr_qpos)
            next_qpos, _ = self.traj_ar_net.step(action_mean)
            loss, loss_idv = self.traj_ar_net.compute_loss_lite(next_qpos, target_qpos)
            self.optimizer.zero_grad()
            loss.backward()   # Testing GT
            self.optimizer.step()  # Testing GT
            pbar.set_description_str(f"Per-step dyna loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_lr()[0]:.5f}")
        

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {'std_id': std_id, 'std_index': std_index}

    def select_action(self, x, mean_action=False):
        dist, action_mean, action_std = self.forward(x)
        action = action_mean if mean_action else dist.sample()
        return action


    def get_kl(self, x):
        dist, _, _ = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist, _, _ = self.forward(x)
        return dist.log_prob(action)
