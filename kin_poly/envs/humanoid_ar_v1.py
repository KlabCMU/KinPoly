import os
import sys

sys.path.append(os.getcwd())

from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.core.policy_mcp import PolicyMCP
from uhc.envs.humanoid_im import HumanoidEnv
from kin_poly.utils.numpy_smpl_humanoid import Humanoid

from gym import spaces
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib
from kin_poly.utils.flags import flags
from uhc.utils.tools import CustomUnpickler


class HumanoidAREnv(HumanoidEnv):

    def __init__(self, cfg, cc_cfg, init_context, cc_iter=-1, mode="train", wild=False, ar_mode=False):
        mujoco_env.MujocoEnv.__init__(self, cc_cfg.mujoco_model_file, 15)
        self.cc_cfg = cc_cfg
        self.kin_cfg = cfg
        self.wild = wild
        self.setup_constants(cc_cfg, cc_cfg.data_specs, mode=mode, no_root=False)

        # env specific
        self.num_obj = self.get_obj_qpos().shape[0] // 7
        self.action_index_map = [0, 7, 21, 28]
        self.action_len = [7, 14, 7, 7]
        self.action_names = ["sit", "push", "avoid", "step"]
        self.smpl_humanoid = Humanoid(model_file=cfg.mujoco_model_file)

        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.prev_hpos = None
        self.set_model_params()
        self.load_context(init_context)
        self.policy_v = cfg.policy_specs['policy_v']
        self.scheduled_sampling = cfg.policy_specs.get("scheduled_smpling", 0)
        self.pose_delta = self.kin_cfg.model_specs.get("pose_delta", False)
        self.ar_model_v = self.kin_cfg.model_specs.get("model_v", 1)
        self.ar_mode = ar_mode
        self.body_diff_thresh = cfg.policy_specs.get('body_diff_thresh', 10)
        self.body_diff_gt_thresh = cfg.policy_specs.get('body_diff_gt_thresh', 12)

        print(f"scheduled_sampling: {self.scheduled_sampling}")

        self.set_spaces()
        self.jpos_diffw = np.array(cfg.reward_weights.get("jpos_diffw", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))[:, None]
        ''' Load CC Controller '''
        state_dim = self.get_cc_obs().shape[0]
        action_dim = self.cc_action_dim
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg, action_dim=action_dim, state_dim=state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg, action_dim=action_dim, state_dim=state_dim)

        self.cc_value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        print(cc_cfg.model_dir)
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            cc_iter = np.max([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(cc_cfg.model_dir)])
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        print(('loading model from checkpoint: %s' % cp_path))
        model_cp = CustomUnpickler(open(cp_path, "rb")).load()
        running_state = model_cp['running_state']
        self.cc_running_state = running_state
        if not self.kin_cfg.joint_controller:
            self.cc_policy.load_state_dict(model_cp['policy_dict'])
            self.cc_value_net.load_state_dict(model_cp['value_dict'])

    def load_context(self, data_dict):
        self.ar_context = {k: v[0].detach().cpu().numpy() if v.requires_grad else v[0].cpu().numpy() for k, v in data_dict.items()}

        self.ar_context['len'] = self.ar_context['qpos'].shape[0] - 1
        self.gt_targets = self.smpl_humanoid.qpos_fk_batch(self.ar_context['qpos'])
        self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][0])

    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def set_spaces(self):
        cfg = self.cc_cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = SMPL_BONE_NAMES
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.cc_action_dim = self.ndof + self.vf_dim

        self.action_dim = 75
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        obs = self.get_obs()

        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        if self.cc_cfg.obs_v == 0:
            cc_obs = self.get_full_obs()
        elif self.cc_cfg.obs_v == 1:
            cc_obs = self.get_full_obs_v1()
        elif self.cc_cfg.obs_v == 2:
            cc_obs = self.get_full_obs_v2()
        return cc_obs

    def get_ar_obs_v1(self):
        t = self.cur_t
        curr_action = self.ar_context["action_one_hot"][0]
        obs = []
        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()

        curr_qpos_local = curr_qpos.copy()
        curr_qpos_local[3:7] = de_heading(curr_qpos_local[3:7])

        pred_wbpos, pred_wbquat = self.get_wbody_pos().reshape(-1, 3), self.get_wbody_quat().reshape(-1, 4)

        head_idx = self.get_head_idx()
        pred_hrot = pred_wbquat[head_idx]
        pred_hpos = pred_wbpos[head_idx]

        pred_hrot_heading = pred_hrot

        if self.kin_cfg.use_context or self.kin_cfg.use_of:
            if "context_feat_rnn" in self.ar_context:
                obs.append(self.ar_context['context_feat_rnn'][t, :])
            else:
                obs.append(np.zeros((256)))

        if self.kin_cfg.use_head:
            # get target head
            t_hrot = self.ar_context['head_pose'][t, 3:].copy()
            t_hpos = self.ar_context['head_pose'][t, :3].copy()
            # get target head vel
            t_havel = self.ar_context['head_vels'][t, 3:].copy()
            t_hlvel = self.ar_context['head_vels'][t, :3].copy()
            t_obj_relative_head = self.ar_context["obj_head_relative_poses"][t, :].copy()

            # difference in head, in head's heading coordinates
            diff_hpos = t_hpos - pred_hpos
            diff_hpos = transform_vec(diff_hpos, pred_hrot_heading, "heading")
            diff_hrot = quaternion_multiply(quaternion_inverse(t_hrot), pred_hrot)

        q_heading = get_heading_q(pred_hrot_heading).copy()

        obj_pos = self.get_obj_qpos(action_one_hot=curr_action)[:3]
        obj_rot = self.get_obj_qpos(action_one_hot=curr_action)[3:7]

        diff_obj = obj_pos - pred_hpos
        diff_obj_loc = transform_vec(diff_obj, pred_hrot_heading, "heading")
        obj_rot_local = quaternion_multiply(quaternion_inverse(q_heading), obj_rot)  # Object local coordinate
        pred_obj_relative_head = np.concatenate([diff_obj_loc, obj_rot_local], axis=0)

        # state
        # order of these matters !!!
        obs.append(curr_qpos_local[2:])  # current height + local body orientation + body pose self.qpose_lm  74
        if self.kin_cfg.use_vel:
            obs.append(curr_qvel)  # current velocities 75

        if self.kin_cfg.use_head:
            obs.append(diff_hpos)  # diff head position 3
            obs.append(diff_hrot)  # diff head rotation 4

        if self.kin_cfg.use_obj:
            obs.append(pred_obj_relative_head)  # predicted object relative to head 7

        if self.kin_cfg.use_head:
            obs.append(t_havel)  # target head angular velocity 3
            obs.append(t_hlvel)  # target head linear  velocity 3
            if self.kin_cfg.use_obj:
                obs.append(t_obj_relative_head)  # target object relative to head 7

        if self.kin_cfg.use_action and self.ar_model_v > 0:
            obs.append(curr_action)

        if self.kin_cfg.use_of:
            # Not sure what to do yet......
            obs.append(self.ar_context['of'][t, :])

        # obs.append(curr_qpos)
        # obs.append(self.get_obj_qpos())
        if self.policy_v == 2:
            obs.append(self.ar_context['ar_qpos'][self.cur_t])

        obs = np.concatenate(obs)

        return obs

    def step_ar(self, a, dt=1 / 30):
        qpos_lm = 74
        pose_start = 7
        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()

        curr_pos, curr_rot = curr_qpos[:3], curr_qpos[3:7]
        curr_heading = get_heading_q(curr_rot)

        body_pose = a[(pose_start - 2):qpos_lm]

        if self.pose_delta:
            body_pose += curr_qpos[pose_start:]
            body_pose[body_pose > np.pi] -= 2 * np.pi
            body_pose[body_pose < -np.pi] += 2 * np.pi

        next_qpos = np.concatenate([curr_pos[:2], a[:(pose_start - 2)], body_pose], axis=0)
        root_qvel = a[qpos_lm:]
        linv = quat_mul_vec(curr_heading, root_qvel[:3])
        next_qpos[:2] += linv[:2] * dt

        angv = quat_mul_vec(curr_rot, root_qvel[3:6])
        new_rot = quaternion_multiply(quat_from_expmap(angv * dt), curr_rot)

        next_qpos[3:7] = new_rot
        return next_qpos

    def step(self, a):
        cfg = self.cc_cfg
        # record prev state
        self.prev_qpos = self.get_humanoid_qpos()
        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        self.prev_hpos = self.get_head().copy()

        if self.policy_v == 1:
            next_qpos = self.step_ar(a.copy())
        elif self.policy_v == 2:
            next_qpos = a.copy()

        self.target = self.smpl_humanoid.qpos_fk(next_qpos)  # forming target from arnet

        # if flags.debug:
        # next_qpos = self.step_ar(self.ar_context['target'][self.cur_t]) #
        # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['qpos'][self.cur_t + 1]) # GT
        # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # Debug
        if self.ar_mode:
            self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1])  #

        cc_obs = self.get_cc_obs()
        cc_obs = self.cc_running_state(cc_obs, update=False)
        mean_action = self.mode == "test" or (self.mode == "train" and self.kin_cfg.joint_controller)
        cc_action = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None,], mean_action=mean_action)[0].numpy()  # CC step

        if flags.debug:
            # self.do_simulation(cc_a, self.frame_skip)
            # self.data.qpos[:self.qpos_lim] = self.get_target_qpos() # debug
            # self.data.qpos[:self.qpos_lim] = self.ar_context['qpos'][self.cur_t + 1] # debug
            # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1] # debug
            self.data.qpos[:self.qpos_lim] = self.ar_context['ar_qpos'][self.cur_t + 1]  # ARNet Qpos
            self.sim.forward()  # debug

        else:
            # if self.mode == "train" and self.scheduled_smpling != 0 and np.random.binomial(1, self.scheduled_smpling):
            #     self.data.qpos[:self.qpos_lim] = self.ar_context['qpos'][self.cur_t + 1]
            #     self.data.qvel[:self.qvel_lim] = self.ar_context['qvel'][self.cur_t + 1]

            #     self.sim.forward() # debug
            # else:
            # self.do_simulation(cc_a, self.frame_skip)
            self.do_simulation(cc_action, self.frame_skip)

        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        head_pos = self.get_body_com(['Head'])
        reward = 1.0

        if cfg.env_term_body == 'body':
            # body_diff = self.calc_body_diff()
            # fail = body_diff > 8

            if self.wild:
                body_diff = self.calc_body_diff()
                fail = body_diff > self.body_diff_thresh
            else:
                body_diff = self.calc_body_diff()
                if self.mode == "train":
                    body_gt_diff = self.calc_body_gt_diff()
                    fail = body_diff > self.body_diff_thresh or body_gt_diff > self.body_diff_gt_thresh
                else:
                    fail = body_diff > self.body_diff_thresh

            # if flags.debug:
            # fail =  False
        else:
            raise NotImplemented()

        end = (self.cur_t >= cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.ar_context['len'])
        done = fail or end
        # if done: # ZL: Debug
        #     exit()
        # print("done!!!", self.cur_t, self.ar_context['len'] )

        percent = self.cur_t / self.ar_context['len']
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end, "percent": percent, "cc_action": cc_action, "cc_state": cc_obs}

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.ar_context['ar_qpos'][self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[:self.qvel_lim] = self.ar_context['ar_qvel'][self.cur_t + 1]
        self.sim.forward()

    def reset_model(self):
        cfg = self.cc_cfg
        ind = 0
        self.start_ind = 0

        if self.ar_mode:
            init_pose_exp = self.ar_context['ar_qpos'][0].copy()
            init_vel_exp = self.ar_context['ar_qvel'][0].copy()
        else:
            init_pose_exp = self.ar_context['init_qpos'].copy()
            init_vel_exp = self.ar_context['init_qvel'].copy()

        # init_vel_exp = self.ar_context['qvel'][ind].copy()

        # init_vel_exp = np.zeros(self.ar_context['init_qvel'].shape)

        # if flags.debug:
        #     init_pose_exp = self.ar_context['qpos'][ind].copy()
        #     init_vel_exp = self.ar_context['qvel'][ind].copy()
        # init_pose_exp[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.qpos_lim - 7)

        # if cfg.reactive_v == 0:
        #     # self.set_state(init_pose, init_vel)
        #     pass
        # elif cfg.reactive_v == 1:
        #     if self.mode == "train" and np.random.binomial(1, 1- cfg.reactive_rate):
        #         # self.set_state(init_pose, init_vel)
        #         pass
        #     elif self.mode == "test":
        #         # self.set_state(init_pose, init_vel)
        #         # netural_qpos = self.netural_data['qpos']
        #         # init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
        #         # init_vel_exp = self.netural_data['qvel']
        #         pass
        #     else:
        #         netural_qpos = self.netural_data['init_qpos']
        #         init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
        #         init_vel_exp = self.netural_data['init_qvel']

        #     # self.set_state(init_pose, init_vel)
        #     self.bquat = self.get_body_quat()
        # else:
        #     init_pose = self.get_humanoid_qpos()
        #     init_pose[2] += 1.0
        #     self.set_state(init_pose, self.data.qvel)

        obj_pose = self.convert_obj_qpos(self.ar_context["action_one_hot"][0], self.ar_context['obj_pose'][0])
        init_pose = np.concatenate([init_pose_exp, obj_pose])
        init_vel = np.concatenate([init_vel_exp, np.zeros(self.num_obj * 6)])

        self.set_state(init_pose, init_vel)
        self.target = self.smpl_humanoid.qpos_fk(init_pose_exp)

        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_qpos(self, delta_t=0):
        expert_qpos = self.target['qpos'].copy()
        return expert_qpos

    def get_target_kin_pose(self, delta_t=0):
        return self.get_expert_qpos()[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        wbpos = self.target['wbpos']
        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        body_com = self.target['body_com']
        return body_com

    def get_expert_bquat(self, delta_t=0):
        bquat = self.target['bquat']
        return bquat

    def get_expert_wbquat(self, delta_t=0):
        wbquat = self.target['wbquat']
        return wbquat

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.ar_context['ar_wbpos'][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.gt_targets['wbpos'][self.cur_t]
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_humanoid_qpos(self):
        return self.data.qpos.copy()[:self.qpos_lim]

    def get_humanoid_qvel(self):
        return self.data.qvel.copy()[:self.qvel_lim]

    def get_obj_qpos(self, action_one_hot=None):
        obj_pose_full = self.data.qpos.copy()[self.qpos_lim:]
        if action_one_hot is None:
            return obj_pose_full
        elif np.sum(action_one_hot) == 0:
            return np.array([0, 0, 0, 1, 0, 0, 0])

        action_idx = np.nonzero(action_one_hot)[0][0]
        obj_start = self.action_index_map[action_idx]
        obj_end = obj_start + self.action_len[action_idx]

        return obj_pose_full[obj_start:obj_end][:7]  # ZL: only support handling one obj right now...

    def convert_obj_qpos(self, action_one_hot, obj_pose):
        if np.sum(action_one_hot) == 0:
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            for i in range(self.num_obj):
                obj_qos[(i * 7):(i * 7 + 3)] = [(i + 1) * 100, 100, 0]
            return obj_qos
        else:
            action_idx = np.nonzero(action_one_hot)[0][0]
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            # setting defult location for objects
            for i in range(self.num_obj):
                obj_qos[(i * 7):(i * 7 + 3)] = [(i + 1) * 100, 100, 0]

            obj_start = self.action_index_map[action_idx]
            obj_end = obj_start + self.action_len[action_idx]
            obj_qos[obj_start:obj_end] = obj_pose

            return obj_qos

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.qvel_lim:]

    def get_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            body_names = self.model.body_names[1:self.body_lim]  # ignore plane
        else:
            body_names = selectList

        for body in body_names:
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)

        return np.concatenate(body_pos)


if __name__ == "__main__":
    pass
