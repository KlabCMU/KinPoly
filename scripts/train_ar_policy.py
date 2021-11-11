import argparse
import os
import sys
import pickle
import time
import glob
import os.path as osp
import joblib
import copy
sys.path.append(os.getcwd())
os.environ['OMP_NUM_THREADS'] = "1"

from copycat.khrylib.utils import *
from torch.utils.tensorboard import SummaryWriter
from relive.models.policy_ar import PolicyAR
from copycat.utils.config import Config as CC_Config
from mujoco_py import load_model_from_path, MjSim
from copycat.khrylib.rl.envs.common.mjviewer import MjViewer
from relive.utils.statear_smpl_config import Config
from relive.core.reward_function import reward_func
from relive.utils.flags import flags
from tqdm import tqdm

from copycat.khrylib.rl.core.critic import Value
from copycat.khrylib.models.mlp import MLP
from relive.envs.humanoid_ar_v1 import HumanoidAREnv
from relive.data_loaders.statear_smpl_dataset import StateARDataset
from relive.models.traj_ar_smpl_net import TrajARNet
from relive.core.agent_ar import AgentAR

def main_loop():
    # if args.render:
    #     agent.sample(1e8)
    # else:
    if cfg.policy_specs['reward_id'] == "dynamic_supervision_v3":
        bar = 0.2
    else:
        bar = 0
        # bar = 0.6 
        # bar = 0.9
    
    agent.data_loader.get_seq_by_ind(0, full_sample = True) # ZL hacky...
    for i_iter in range(args.iter, cfg.num_epoch):
        # eval_res = agent.eval_policy("test")
        # logger.info(eval_res)
        # """generate multiple trajectories that reach the minimum batch_size"""

    #     if (i_iter % 10 == 0 or i_iter <= 1) and args.test_time:
    #         while True:
    #             curr_seq_res = osp.join(cfg.result_dir, f"{agent.data_loader.get_seq_key(agent.fit_ind)}.pkl")
    #             while osp.exists(curr_seq_res):
    #                 print(f"fitted: {curr_seq_res}")
    #                 agent.next_fit_seq()
    #                 curr_seq_res = osp.join(cfg.result_dir, f"{agent.data_loader.get_seq_key(agent.fit_ind)}.pkl")
                    
    #             info = agent.eval_cur_seq()
    #             logger.info("========> Evaluating")
    #             logger.info(f"{info['percent']:.3f}, {len(info['reward'])}|{agent.ar_context['ar_qpos'].shape[1]}, {agent.data_loader.curr_key},  {agent.fit_ind}, {np.mean(info['reward']):.3f}")
    #             # if info['percent'] == 1 and np.mean(info['reward']) > 0.25:
    #             if info['percent'] == 1 and np.mean(info['reward']) > bar:
    #                 joblib.dump(info, curr_seq_res)
    #                 agent.next_fit_seq()
    #             else:
    #                 break
        batch, log = agent.sample(cfg.policy_specs['min_batch_size'])
        
        if cfg.policy_specs['end_reward']:
            agent.env.end_reward = log.avg_c_reward * cfg.policy_specs['gamma'] / (1 - cfg.policy_specs['gamma'])

        """update networks"""
        t0 = time.time()
        agent.update_params(batch)
        t1 = time.time()

        """logging"""
        c_info = log.avg_c_info
        logger.info(
            'Ep: {}\t {} Test-time: {} Ind: {}  \tT_s {:.2f}\tT_u {:.2f}\tETA {}\texpert_R_avg {:.4f} {}'
            '\texpert_R_range ({:.4f}, {:.4f})\teps_len {:.2f}'
            .format(i_iter, args.cfg, args.test_time, agent.fit_ind, log.sample_time, t1 - t0, get_eta_str(i_iter, cfg.policy_specs['max_iter_num'], t1 - t0 + log.sample_time), log.avg_c_reward,
                    np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=','),
                    log.min_c_reward, log.max_c_reward, log.avg_episode_len))

        tb_logger.add_scalar('total_reward', log.avg_c_reward, i_iter)
        tb_logger.add_scalar('episode_len', log.avg_episode_reward, i_iter)
        for i in range(c_info.shape[0]):
            tb_logger.add_scalar('reward_%d' % i, c_info[i], i_iter)
            tb_logger.add_scalar('eps_reward_%d' % i, log.avg_episode_c_info[i], i_iter)

        # if args.test_time: cfg.policy_specs['save_model_interval'] = 10 # ZL: more agressive saving schedule for test time
        if args.test_time:
            joblib.dump(agent.freq_dict, osp.join(cfg.result_dir, f"freq_dict_{'wild_' if args.wild else '' }test.pt"))
        else:
            joblib.dump(agent.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))


        if (i_iter + 1) % cfg.policy_specs.get("test_interval", 50) == 0:
            # if not args.test_time:
            #     eval_res = agent.eval_policy("train", i_iter = i_iter)
            #     logger.info(eval_res)
            eval_res = agent.eval_policy("test", i_iter = i_iter)
            logger.info(eval_res)                

            
        if cfg.policy_specs['save_model_interval'] > 0 and (i_iter+1) % cfg.policy_specs['save_model_interval'] == 0:
            tb_logger.flush()
            agent.save_checkpoint(i_iter)
            # if not args.test_time:
            
        
                
        """clean up gpu memory"""
        torch.cuda.empty_cache()

    logger.info('training done!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--cc_cfg',  default="copycat_9_1")
    parser.add_argument('--cc_iter', type=int, default=-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--wild', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_threads', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--ar_iter', type=int, default=-1)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--test_time', action='store_true', default=False)
    parser.add_argument('--show_noise', action='store_true', default=False)
    args = parser.parse_args()

    flags.debug = args.debug

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'

    """load CC model"""
    cc_cfg = CC_Config(args.cc_cfg, "/insert_directory_here//", create_dirs=False)
    cc_cfg_wild = CC_Config(args.cc_cfg, "/insert_directory_here//", create_dirs=False)

    if args.wild:
        cc_cfg.mujoco_model_file = "humanoid_smpl_neutral_mesh_all.xml"
    else:
        cc_cfg.mujoco_model_file = "humanoid_smpl_neutral_mesh_all_step.xml"
    cc_cfg_wild.mujoco_model_file = "humanoid_smpl_neutral_mesh_all.xml"

    cfg = Config(args.action, args.cfg, wild = args.wild, create_dirs=(args.iter == 0), mujoco_path = "%s.xml")

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = SummaryWriter(cfg.tb_test_dir) if not args.render else None
    logger = create_logger(os.path.join(cfg.log_dir, f'log_ar{"_test" if args.test_time else ""}_{"wild_" if args.wild else ""}policy.txt'), file_handle=not args.render)


    if args.render:
        args.num_threads = 1


    """Datasets"""
    data_loader = StateARDataset(cfg, args.data, sim = True)
    data_sample = data_loader.sample_seq()
    data_sample =  {k:v.to(device).clone().type(dtype) for k, v in data_sample.items()}

    state_dim = data_loader.traj_dim
    policy_net = PolicyAR(cfg, data_sample, device= device, dtype = dtype, ar_iter = args.ar_iter)
    with torch.no_grad():
        context_sample = policy_net.init_context(data_sample)

    env = HumanoidAREnv(cfg, cc_cfg = cc_cfg, init_context = context_sample, mode = "train", wild = args.wild)
    env.seed(cfg.seed)
    env_wild = HumanoidAREnv(cfg, cc_cfg = cc_cfg_wild, init_context = context_sample, mode = "train", wild = True)
    env_wild.seed(cfg.seed)


    actuators = env.model.actuator_names
    # state_dim = env.observation_space.shape[0]
    state_dim = policy_net.state_dim
    action_dim = env.action_space.shape[0]
    running_state = None  # No running state for the ARNet!!!
    
    value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))

    if args.iter > 0:
        if args.wild:
            if args.test_time: cp_path = '%s/iter_wild_test_%04d.p' % (cfg.policy_model_dir, args.iter )
            else: cp_path = '%s/iter_wild_%04d.p' % (cfg.policy_model_dir, args.iter)
        else:
            if args.test_time: cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)
            else: cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)

        if not osp.exists(cp_path):
            cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)

        # cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)
        # cp_path = f'{cfg.policy_model_dir}/iter_test_6270.p'

        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        policy_net.load_state_dict(model_cp['policy_dict'])
        
        # policy_net.old_arnet[0].load_state_dict(copy.deepcopy(policy_net.traj_ar_net.state_dict())) # ZL: should use the new old net as well
        
        value_net.load_state_dict(model_cp['value_dict'])
        running_state = model_cp['running_state']
    to_device(device, policy_net, value_net)

    if cfg.policy_specs['policy_optimizer'] == 'Adam':
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_specs['policy_lr'], weight_decay=cfg.policy_specs['policy_weightdecay'])
    else:
        optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=cfg.policy_specs['policy_lr'], momentum=cfg.policy_specs['policy_momentum'], weight_decay=cfg.policy_specs['policy_weightdecay'])
        
    if cfg.policy_specs['value_optimizer'] == 'Adam':
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=cfg.policy_specs['value_lr'], weight_decay=cfg.policy_specs['value_weightdecay'])
    else:
        optimizer_value = torch.optim.SGD(value_net.parameters(), lr=cfg.policy_specs['value_lr'], momentum=cfg.policy_specs['policy_momentum'], weight_decay=cfg.policy_specs['value_weightdecay'])

    # reward functions
    expert_reward = reward_func[cfg.policy_specs['reward_id']]

    """create agent"""
    agent = AgentAR(wild = args.wild, cfg = cfg, test_time = args.test_time, checkpoint_epoch = args.iter , env=env, env_wild = env_wild, dtype=dtype, device=device, running_state=running_state,
                    custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                    render=args.render, num_threads=args.num_threads, data_loader = data_loader,
                    policy_net=policy_net, value_net=value_net,
                    optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, opt_num_epochs=cfg.policy_specs['num_optim_epoch'],
                    gamma=cfg.policy_specs['gamma'], tau=cfg.policy_specs['tau'], clip_epsilon=cfg.policy_specs['clip_epsilon'],
                    policy_grad_clip=[(policy_net.parameters(), 40)], end_reward=cfg.policy_specs['end_reward'],
                    use_mini_batch=cfg.policy_specs['mini_batch_size'] < cfg.policy_specs['min_batch_size'], mini_batch_size=cfg.policy_specs['mini_batch_size'])

    main_loop()