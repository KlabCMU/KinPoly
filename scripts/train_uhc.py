import argparse
import os
import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp
os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from torch.utils.tensorboard import SummaryWriter
from uhc.utils.config import Config


def pre_iter_update(i_iter):
    cfg.update_adaptive_params(i_iter)
    agent.set_noise_rate(cfg.adp_noise_rate)
    set_optimizer_lr(optimizer_policy, cfg.adp_policy_lr)

    # i_iter 
    adative_iter = cfg.data_specs.get("adaptive_iter", -1)
    # if i_iter != 0 and adative_iter != -1 and i_iter % adative_iter == 0 :
        # agent.data_loader.hard_negative_mining(agent.value_net, agent.env, device, dtype, running_state = running_state, sampling_temp = cfg.sampling_temp)

    if cfg.fix_std:
        policy_net.action_log_std.fill_(cfg.adp_log_std)
    return


def main_loop():

    if args.render:
        pre_iter_update(args.iter)
        agent.sample(1e8)
    else:
        for i_iter in range(args.iter, cfg.max_iter_num):

            if (i_iter) % cfg.get("test_interval", 200) == 0:
                eval_res = agent.eval_policy("train", i_iter = i_iter)
                logger.info(eval_res)


            """generate multiple trajectories that reach the minimum batch_size"""
            pre_iter_update(i_iter)
            batch, log = agent.sample(cfg.min_batch_size)
            if cfg.end_reward:
                agent.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)   

            """update networks"""
            t0 = time.time()
            agent.update_params(batch)
            t1 = time.time()

            """logging"""
            c_info = log.avg_c_info
            logger.info(
                '{} {} \tT_s {:.2f}\tT_u {:.2f}\tETA {}\texpert_R_avg {:.4f} {}'
                '\texpert_R_range ({:.4f}, {:.4f})\teps_len {:.2f}'
                .format(args.cfg, i_iter, log.sample_time, t1 - t0, get_eta_str(i_iter, cfg.max_iter_num, t1 - t0 + log.sample_time), log.avg_c_reward,
                        np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=','),
                        log.min_c_reward, log.max_c_reward, log.avg_episode_len))

            tb_logger.add_scalar('total_reward', log.avg_c_reward, i_iter)
            tb_logger.add_scalar('episode_len', log.avg_episode_reward, i_iter)
            for i in range(c_info.shape[0]):
                tb_logger.add_scalar('reward_%d' % i, c_info[i], i_iter)
                tb_logger.add_scalar('eps_reward_%d' % i, log.avg_episode_c_info[i], i_iter)

            joblib.dump(agent.freq_dict, osp.join(cfg.output_dir, "freq_dict.pt"))


            if cfg.save_model_interval > 0 and (i_iter+1) % cfg.save_model_interval == 0:
                tb_logger.flush()
                agent.save_checkpoint(i_iter)
                
            """clean up gpu memory"""
            torch.cuda.empty_cache()

        logger.info('training done!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_threads', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--show_noise', action='store_true', default=False)
    args = parser.parse_args()

    cfg = Config(args.cfg, args.test, create_dirs=not (args.render or args.iter > 0))

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = SummaryWriter(cfg.tb_dir) if not args.render else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), file_handle=not args.render)




    if args.render:
        from mujoco_py import load_model_from_path, MjSim
        from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
        args.num_threads = 1
        model = load_model_from_path(f'assets/mujoco_models/{cfg.mujoco_model_file}')
        sim = MjSim(model)
        viewer = MjViewer(sim)


    from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
    from uhc.khrylib.rl.core.critic import Value
    from uhc.core.agent_uhc import Agentuhc

    from uhc.core.reward_function import reward_func
    from uhc.core.policy_mcp import PolicyMCP
    from uhc.khrylib.models.mlp import MLP
    from uhc.envs.humanoid_im import HumanoidEnv
    from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle


    """environment"""
    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="train")
    random_expert = data_loader.sample_seq()

    env = HumanoidEnv(cfg, init_expert = random_expert, data_specs = cfg.data_specs, mode = "train", no_root=cfg.no_root)
    env.seed(cfg.seed)
    actuators = env.model.actuator_names
    state_dim = env.observation_space.shape[0]
    print(state_dim)
    print(state_dim)
    print(state_dim)
    action_dim = env.action_space.shape[0]
    running_state = ZFilter((state_dim,), clip=5)
    

    """define actor and critic"""
    if cfg.actor_type == "gauss":
        policy_net = PolicyGaussian(cfg, action_dim = action_dim, state_dim = state_dim)
    elif cfg.actor_type == "mcp":
        policy_net = PolicyMCP(cfg, action_dim = action_dim, state_dim = state_dim)


    value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
    if args.iter > 0:
        cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        policy_net.load_state_dict(model_cp['policy_dict'])
        value_net.load_state_dict(model_cp['value_dict'])
        running_state = model_cp['running_state']
    to_device(device, policy_net, value_net)

    if cfg.policy_optimizer == 'Adam':
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
    else:
        optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
    if cfg.value_optimizer == 'Adam':
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
    else:
        optimizer_value = torch.optim.SGD(value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)

    # reward functions
    expert_reward = reward_func[cfg.reward_id]
    """create agent"""
    agent = Agentuhc(cfg = cfg, env=env, dtype=dtype, device=device, running_state=running_state,
                    custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                    render=args.render, num_threads=args.num_threads, data_loader = data_loader,
                    policy_net=policy_net, value_net=value_net,
                    optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                    gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                    policy_grad_clip=[(policy_net.parameters(), 40)], end_reward=cfg.end_reward,
                    use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)

    main_loop()