import os
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter
import pandas as pd
from utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from utils.adj_buffer import AdjBuffer
from utils.util import  get_dim_from_space
from utils.util import DecayThenFlatSchedule
from utils.normalization import RewardScaling

# ===== 修改点 1：新增 run 前缀工具函数 =====
def _get_run_csv_name(run_dir, filename):
    """
    将 run 目录下的 CSV 文件统一加 run 前缀。

    例如：
      run_dir = ".../run1"
      filename = "progress.csv"
      return = "run1_progress.csv"

    若 filename 已经带 run 前缀，则不重复添加。
    """
    filename = str(filename)

    if not filename.endswith(".csv"):
        return filename

    run_name = os.path.basename(str(run_dir).rstrip(os.sep))

    # 防御：如果 run_dir 不是 run1/run2 这种格式，则保持原名
    if not run_name.startswith("run"):
        return filename

    prefix = run_name + "_"
    if filename.startswith(prefix):
        return filename

    return prefix + filename

class RecRunner(object):
    """Base class for training recurrent policies."""

    # ===== 修改点 2：所有 base_runner scalar CSV 自动带 run 前缀 =====
    def _append_scalar_csv(self, filename, row_dict):
        filename = _get_run_csv_name(self.run_dir, filename)
        progress_filename = os.path.join(self.run_dir, filename)

        df = pd.DataFrame([row_dict])
        if not os.path.exists(progress_filename):
            df.to_csv(progress_filename, index=False)
            return

        # Scalar dictionaries can gain fields after warmup (for example,
        # policy losses do not exist before the first optimizer update). Keep
        # the CSV schema aligned instead of appending wider rows under an old
        # header. Rewriting happens only when a new column first appears.
        existing_columns = list(pd.read_csv(progress_filename, nrows=0).columns)
        new_columns = [column for column in df.columns if column not in existing_columns]
        all_columns = existing_columns + new_columns

        if new_columns:
            existing_df = pd.read_csv(progress_filename)
            existing_df.reindex(columns=all_columns).to_csv(progress_filename, index=False)

        df.reindex(columns=all_columns).to_csv(
            progress_filename,
            mode='a',
            header=False,
            index=False,
        )

    # 检查sddfg参数
    def _validate_dynamic_graph_args(self):
        if getattr(self.args, "algorithm_name", None) == "sddfg":
            if self.hidden_size % max(1, int(self.args.gat_heads)) != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} must be divisible by gat_heads={self.args.gat_heads} "
                    "to avoid silent GAT head dimension truncation."
                )
            if bool(getattr(self.args, "require_connected_adj", False)):
                max_agents = int(getattr(self.args, "max_player_num", 1))
                max_order = max(2, int(getattr(self.args, "highest_orders", 2)))
                min_connected_factors = (
                    max_agents - 1 + max_order - 2
                ) // (max_order - 1)
                if int(getattr(self.args, "num_factor", 0)) < min_connected_factors:
                    raise ValueError(
                        "require_connected_adj needs at least {} factors for "
                        "max_player_num={} and highest_orders={}, got {}.".format(
                            min_connected_factors,
                            max_agents,
                            max_order,
                            getattr(self.args, "num_factor", 0),
                        )
                    )

        min_adj_begin_step = int(getattr(self.args, "min_adj_begin_step", 5000))
        adj_begin_step = int(getattr(self.args, "adj_begin_step", 0))
        adj_lr = float(getattr(self.args, "adj_lr", 5e-4))

        if self.use_dyn_graph and adj_begin_step < min_adj_begin_step:
            print(
                f"[WARN] adj_begin_step={adj_begin_step} is very early for dynamic-agent training. "
                f"Recommended >= {min_adj_begin_step}."
            )

        if self.use_dyn_graph and adj_lr > 1e-3:
            print(
                f"[WARN] adj_lr={adj_lr} is high for PPO-style adj/GAT updates. "
                "Recommended 5e-4 ~ 1e-3 for dynamic wolfpack."
            )

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        # 将传入的配置保存到实例变量，方便后续使用
        self.args = config["args"]
        self.device = config["device"]
        self.adj = config["adj"]

        # 一些算法分组（字符串列表），用于后续根据 algorithm_name 决定行为分支
        # q_learning 列表里是基于 Q-learning 的算法（例如 QMix、VDN 等）
        self.q_learning = ["qplex", "qtran", "wqmix", "qmix", "vdn", "ddfg", "sopcg", "casec", "sddfg"]
        # adj_correlation 列表表示需要关联邻接（adjacency）网络的算法
        self.adj_correlation = ["ddfg", "sopcg", "casec", "sddfg"]

        # —— 从args抄出各种常见训练/环境配置，便于后续使用 ——
        self.share_policy = self.args.share_policy  # 是否共享策略（多agent同一套参数）
        self.algorithm_name = self.args.algorithm_name  # 算法名（决定采用何种Policy/Trainer）
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps  # 训练目标的总交互步数
        self.use_wandb = self.args.use_wandb  # 是否用wandb记录日志
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per  # 是否使用优先经验回放
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size  # 轨迹回放缓冲区大小
        self.batch_size = self.args.batch_size
        self.adj_buffer_size = self.args.adj_buffer_size  # 邻接图训练的缓冲区大小
        self.hidden_size = self.args.hidden_size  # RNN/网络隐藏维度
        self.highest_orders = self.args.highest_orders  # 因子图最高阶（如1/2/3阶）
        self.use_soft_update = self.args.use_soft_update  # 目标网络软更新
        self.hard_update_interval_episode = self.args.hard_update_interval_episode  # 硬更新的间隔（按episode）
        self.popart_update_interval_step = self.args.popart_update_interval_step  # PopArt更新频率
        self.actor_train_interval_step = self.args.actor_train_interval_step  # Actor更新步频（对AC类）
        self.train_interval_episode = self.args.train_interval_episode  # 每多少个episode进行一次训练
        self.train_adj_episode = self.args.train_adj_episode  # 每多少个episode训练一次邻接图
        self.drop_temperature_episode = self.args.drop_temperature_episode  # 温度/epsilon下降节奏
        self.train_interval = self.args.train_interval  # （备用）训练间隔
        self.use_dyn_graph = self.args.use_dyn_graph  # 是否使用动态图（学习邻接）
        self.equal_vdn = self.args.equal_vdn  # 是否退化成VDN等价形式（特殊实验用）
        self.use_eval = self.args.use_eval  # 是否开启评估
        self.eval_interval = self.args.eval_interval  # 评估间隔（按总步数）
        self.save_interval = self.args.save_interval  # 模型保存间隔（按总步数）
        self.log_interval = self.args.log_interval  # 日志打印/写入间隔（按总步数）
        self.gae_lambda = self.args.gae_lambda  # GAE参数（若使用）
        self.gamma = self.args.gamma  # 折扣因子
        self.use_linear_lr_decay = self.args.use_linear_lr_decay  # 是否线性衰减学习率
        #self.independent_p_q = self.args.independent_p_q  # 一些算法中p/q网络是否独立
        #self.pair_rnn_hidden_dim = self.args.pair_rnn_hidden_dim  # （特定算法用）pair-RNN隐藏维度
        self.epsilon_anneal_time = self.args.epsilon_anneal_time  # epsilon退火时间（探索）

        # 下面是训练过程中的统计量和状态，用于控制训练周期、记录日志、保存等
        self.total_env_steps = 0  # 训练期间与环境交互的总步数（累加）
        self.num_episodes_collected = 0  # 训练期间收集到的总 episode 数
        self.num_adj_episodes_collected = 0  # 已用于邻接训练的 episode 数
        self.total_train_steps = 0  # 已进行的梯度更新次数
        self.last_train_episode = 0  # 上次执行梯度更新时的 episode 计数
        self.last_train_adj_episode = 0  # 上次训练邻接网络时的 episode
        self.last_drop_t_episode = 0  # 上次 temperature 降低时的 episode
        self.last_eval_T = 0  # 上次做 eval 的 step
        self.last_save_T = 0  # 上次保存模型的 step
        self.last_log_T = 0  # 上次记录日志的 step
        self.last_hard_update_episode = 0  # 上次进行硬参数拷贝（target <- live）的 episode
        self.use_vfunction = self.args.use_vfunction  # 是否使用 state/value function v（针对多阶的 value）
        self.use_save = self.args.use_save  # 是否将模型保存到磁盘
        self.pretrain_adj = self.args.pretrain_adj  # 是否预加载（预训练）邻接网络
        self.num_mini_batch = self.args.num_mini_batch  # 用于邻接训练时的 minibatch 数目
        self.adj_train_epochs = max(
            1,
            int(getattr(self.args, "adj_train_epochs", 10)),
        )
        self.use_adj_linear_lr_decay = bool(
            getattr(self.args, "use_adj_linear_lr_decay", False)
        )
        self._adj_lr_decay_warning_emitted = False
        self.adj_begin_step = self.args.adj_begin_step  # 在多少 step 之后开始训练动态邻接
        self.use_adj_init = self.args.use_adj_init  # 是否在训练开始时使用邻接的初始值
        self._validate_dynamic_graph_args() #检查sddfg参数
        self._lr_decay_warning_emitted = False

        # config 中有些字段不是必须的，使用 contains 检查并设置默认值
        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        # 决定 episode_length 与 data_chunk_length（用于 RNN 训练时切分序列）
        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            # 对于 naive recurrent policy，data_chunk_length 直接等于整个 episode 长度
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                # 否则使用 args 中指定的数据块长度（用于截断 BPTT）
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        #import pdb;pdb.set_trace()
        # policy_info 存放了每个 policy 的 observation/action 空间等结构化信息
        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        # 多 agent/因子设置
        self.num_agents = config["num_agents"]
        self.num_factor = self.args.num_factor  # 在 DDFG 中的 factor 数（用于高阶交互建模）
        self.agent_ids = [i for i in range(self.num_agents)]

        # 环境与并行数
        self.env = config["env"]
        self.eval_env = config["eval_env"]
        # no parallel envs
        self.num_envs = self.env.num_envs  # 并行环境的数量（通常是向量化环境里的一维）
        self.num_eval_envs = self.eval_env.num_envs

        self.action_repr_updating = True  # 某些算法（casec）需要更新 action 表示

        # 模型与日志路径设置（支持 wandb 或本地写入）
        # import pdb;pdb.set_trace()
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            # 当使用 wandb 时，保存目录由 wandb 控制
            self.save_dir = str(wandb.run.dir)
            # import pdb;pdb.set_trace()
            self.run_dir = config["run_dir"]
        else:
            # 本地写入日志和模型
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)  # tensorboardX 写入器
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        # 根据算法名选择对应的 Policy/Adj/TrainAlgo 类
        if self.algorithm_name == "ddfg":
            # 当算法为 ddfg 时，导入对应的邻接生成器（Adj）和策略类（Policy），以及训练算法实现（TrainAlgo）
            from algorithms.ddfg.algorithm.adj_generator_new import Adj_Generator as Adj
            from algorithms.ddfg.algorithm.rDDFGPolicy import R_DDFGPolicy as Policy
            from algorithms.ddfg.r_ddfg import R_DDFG as TrainAlgo
        elif self.algorithm_name == "sddfg":
            from algorithms.sddfg.r_sddfg import R_SDDFG as TrainAlgo
            from algorithms.sddfg.algorithm.rSDDFGPolicy import R_SDDFGPolicy as Policy
            from algorithms.sddfg.algorithm.adj_generator import Adj_Generator as Adj

        elif self.algorithm_name == "vdn":
            from algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from algorithms.vdn.vdn import VDN as TrainAlgo

        elif self.algorithm_name == "qmix":
            from algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from algorithms.qmix.qmix import QMix as TrainAlgo

        elif self.algorithm_name == "qplex":
            from algorithms.qplex.algorithm.QPlexPolicy import QPlexPolicy as Policy
            from algorithms.qplex.qplex import QPlex as TrainAlgo

        else:
            # 目前只实现了 sddfg分支，其他算法未实现时抛异常
            raise NotImplementedError

        # 采集函数指向 collect_rollout（子类实现），训练/保存/恢复函数根据算法类别指派
        self.collecter = self.collect_rollout
        if self.algorithm_name in self.adj_correlation:
            self.saver = self.save_q_mdfg_cent
            self.restorer = self.restore_mdfg_cent
        elif self.algorithm_name in self.q_learning:
            self.saver = self.save_q
            self.restorer = self.restore_q
        else:
            self.saver = self.save
            self.restorer = self.restore

        # —— 根据是否Q-learning系，绑定训练方法（train）为 batch_train_q 或 batch_train ——
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train
        self.train_adj = self.batch_train_adj

        # 根据 policy_info 创建每个 policy 的 Policy 实例（例如 actor/critic/network）
        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}

        # 获取 observation/state/action 的维度（从 gym space 中推断）
        self.obs_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["obs_space"])
        self.state_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["share_obs_space"])
        self.act_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["act_space"])

        # initialize trainer class for updating policies
        if self.algorithm_name in self.adj_correlation:
            # 初始化邻接网络，用于预测 agent 之间的结构/相关性
            self.adj_network = Adj(self.args, self.obs_dim, self.state_dim, self.act_dim, self.device)
            # 初始化训练器（TrainAlgo），将 policies、adj_network 等传入
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.adj_network, self.policy_mapping_fn,
                                     device=self.device, episode_length=self.episode_length)
        else:
            # 不需要 adj 的算法直接初始化训练器（TrainAlgo）
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                     device=self.device, episode_length=self.episode_length)
        # 如果提供了 model_dir，则根据是否预训练 adj 决定加载方式
        if self.model_dir is not None:
            if self.pretrain_adj:
                self.load_adj()
            else:
                self.restorer()
        # map policy id to agent ids controlled by that policy
        self.policy_agents = {
            policy_id: sorted(
                [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id])
            for policy_id in self.policies.keys()
        }

        # 保存每个 policy 的 obs/act/central_obs 维度以便后续使用
        self.policy_obs_dim = {policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {policy_id: self.policies[policy_id].central_obs_dim for policy_id in
                                       self.policy_ids}

        # 估计总共会进行多少 train episode，用于 PER 中 beta 的退火调度
        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        # RewardScaling 用于对 reward 做幂次/尺度缩放以稳定训练（实现细节见 utils/normalization.py）
        self.reward_scaling = RewardScaling(shape=1, gamma=self.gamma)

        # 根据是否打开 PER 来选择不同类型的回放缓冲区（普通或优先级）
        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization,
                                                     seed=int(self.args.seed) + 410000,
                                                     )
        else:
            # RecReplayBuffer 用于存储 recurrent（序列）数据，支持按 policy 存储
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.num_factor,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization,
                                          seed=int(self.args.seed) + 410000,
                                          )

        # 邻接缓冲区，用于存储用于训练动态邻接网络的数据（含 gae/gamma/hidden 等）
        self.adj_buffer = AdjBuffer(self.policy_info,
                                    self.policy_agents,
                                    self.num_factor,
                                    self.adj_buffer_size,
                                    self.episode_length,
                                    self.use_same_share_obs,
                                    self.use_avail_acts,
                                    self.use_reward_normalization,
                                    self.gamma,
                                    self.gae_lambda,
                                    self.hidden_size,
                                    adj_return_adv_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_return_adv_coef",
                                            1.0,
                                        )
                                    ),
                                    adj_factor_adv_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_factor_adv_coef",
                                            0.0,
                                        )
                                    ),
                                    seed=int(self.args.seed) + 420000,
                                    use_adj_delayed_triplet_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_delayed_triplet_credit",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_window=int(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_window",
                                            0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_cap",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_min_reward=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_min_reward",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_positive_only=bool(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_positive_only",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_min_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_min_adv",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_credit_require_future_match=bool(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_credit_require_future_match",
                                            False,
                                        )
                                    ),
                                    use_adj_delayed_triplet_success_gate=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_delayed_triplet_success_gate",
                                            False,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_min_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_min_adv",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_scale=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_scale",
                                            1.0,
                                        )
                                    ),
                                    adj_delayed_triplet_success_gate_floor=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_success_gate_floor",
                                            0.0,
                                        )
                                    ),
                                    adj_delayed_triplet_future_overlap_min_nodes=int(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_future_overlap_min_nodes",
                                            3,
                                        )
                                    ),
                                    adj_delayed_triplet_partial_match_weight=float(
                                        getattr(
                                            self.args,
                                            "adj_delayed_triplet_partial_match_weight",
                                            0.5,
                                        )
                                    ),
                                    use_adj_capture_to_win_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_capture_to_win_credit",
                                            False,
                                        )
                                    ),
                                    adj_capture_to_win_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_capture_to_win_credit_min_outcome_adv=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_min_outcome_adv",
                                            0.5,
                                        )
                                    ),
                                    adj_capture_to_win_credit_scale=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_scale",
                                            0.75,
                                        )
                                    ),
                                    adj_capture_to_win_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_cap",
                                            0.35,
                                        )
                                    ),
                                    adj_capture_to_win_credit_require_future_match=bool(
                                        getattr(
                                            self.args,
                                            "adj_capture_to_win_credit_require_future_match",
                                            False,
                                        )
                                    ),
                                    use_adj_pair_triplet_complementary_credit=bool(
                                        getattr(
                                            self.args,
                                            "use_adj_pair_triplet_complementary_credit",
                                            False,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_coef=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_coef",
                                            0.0,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_window=int(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_window",
                                            20,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_cap=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_cap",
                                            0.20,
                                        )
                                    ),
                                    adj_pair_pursuit_credit_min_reward=float(
                                        getattr(
                                            self.args,
                                            "adj_pair_pursuit_credit_min_reward",
                                            0.0,
                                        )
                                    ),
                                    )

    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""

        # 在收集 rollouts 之前把 trainer 切换到 rollout 模式（例如关闭梯度、把网络置为 eval 等）
        self.trainer.prep_rollout()

        # 执行一次采集（子类需实现 collect_rollout），并把返回的环境信息统计到 env_infos
        env_info = self.collecter(explore=True, training_episode=True, warmup=False)
        for k, v in env_info.items():
            self.env_infos[k].append(v)

        # train：如果满足训练触发条件（收集足够的 episode）则进行一次训练
        if ((self.num_episodes_collected - self.last_train_episode - self.batch_size) / self.train_interval_episode) >= 1:
            # LR scheduling must follow the policy-update clock.  Its former
            # placement inside adjacency updates decayed SDDFG while value
            # baselines silently kept their initial learning rate.
            if self.use_linear_lr_decay:
                lr_decay = getattr(self.trainer, "lr_decay", None)
                if callable(lr_decay):
                    policy_lr_anneal_steps = int(
                        getattr(self.args, "policy_lr_anneal_steps", 0)
                    )
                    if policy_lr_anneal_steps <= 0:
                        policy_lr_anneal_steps = self.num_env_steps
                    lr_decay(
                        self.total_env_steps,
                        policy_lr_anneal_steps,
                    )
                elif not self._lr_decay_warning_emitted:
                    print(
                        "[WARN] --use_linear_lr_decay has no effect for "
                        "trainer {} because it has no lr_decay() method.".format(
                            type(self.trainer).__name__
                        )
                    )
                    self._lr_decay_warning_emitted = True
            self.train()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected - self.batch_size

        # 如果使用动态邻接图且满足训练邻接的条件，则训练邻接网络
        if self.use_dyn_graph and self.total_env_steps >= self.adj_begin_step and (
                (self.num_adj_episodes_collected - self.last_train_adj_episode) / self.train_adj_episode) >= 1:
            if not self.pretrain_adj:
                # Keep graph scheduling independent from policy LR decay.
                # Previously the adjacency flag had no effect unless the
                # unrelated global policy decay flag was also enabled.
                if self.use_adj_linear_lr_decay:
                    adj_lr_decay = getattr(self.trainer, "adj_lr_decay", None)
                    if callable(adj_lr_decay):
                        adj_lr_decay(self.total_env_steps)
                    elif not self._adj_lr_decay_warning_emitted:
                        print(
                            "[WARN] --use_adj_linear_lr_decay has no effect "
                            "for trainer {} because it has no "
                            "adj_lr_decay() method.".format(
                                type(self.trainer).__name__
                            )
                        )
                        self._adj_lr_decay_warning_emitted = True
                # 训练邻接网络的具体实现由 batch_train_adj 完成
                self.train_adj()
                self.log_train_adj(self.train_adj_infos)
            # Keep both sides of the scheduling delta in the same counter
            # domain.  Warmup episodes increment num_episodes_collected but
            # are intentionally excluded from num_adj_episodes_collected.
            # Storing the total counter here made the next adjacency update
            # wait dozens of episodes (and progressively longer) instead of
            # the configured train_adj_episode interval.
            self.last_train_adj_episode = (
                self.num_adj_episodes_collected
            )
        # save：按间隔保存模型
        if self.use_save and (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps
        # log：按间隔记录日志
        if self.total_env_steps > 0 and ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval：按间隔执行评估
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def warmup(self, num_warmup_episodes):
        """
        预热：在正式训练前先收集一些随机（或带探索）的episode，填满回放缓冲区，让训练更稳。
        :param num_warmup_episodes: 需要的预热episode数量
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs)):
            # 预热一般用 explore=True，warmup=True，表示只收集不训练
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_episode_rewards'])

        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """（非Qmix类）做一次参数更新：对每个policy从buffer取样→train→（必要的）软/硬更新。"""
        self.trainer.prep_training()  # 切换到训练模式（train）
        self.train_infos = []
        update_actor = False

        for p_id in self.policy_ids:
            # PER：带重要性采样权重beta；否则普通采样
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            # 如果obs是集中式共享，则调用“共享中心obs”的训练；否则用独立版本
            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs \
                else self.trainer.cent_train_policy_on_batch

            # 训练并拿到信息（loss等）、优先级（PER）、索引
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = True  # 这里保留开关位（某些算法只在满足条件时更新actor）

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        # 目标网络更新：优先软更新；否则到间隔用硬更新
        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Q-learning系（QMIX/VDN/DDFG等）的批训练：从buffer采样→trainer.train_policy_on_batch。"""
        self.trainer.prep_training()
        self.train_infos = []

        if self.algorithm_name == 'casec':
            sample = self.buffer.sample(self.batch_size)
            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample, self.action_repr_updating)
            self.train_infos.append(train_info)

        #elif self.algorithm_name == 'rddfg_cent_rw':
        elif self.algorithm_name in ["ddfg", "sddfg"]:
            # DDFG：每个训练间隔内多次小批次迭代，有助于稳定
            for _ in range(self.train_interval_episode):
                for p_id in self.policy_ids:
                    if self.use_per:
                        beta = self.beta_anneal.eval(self.total_train_steps)
                        sample = self.buffer.sample(self.batch_size, beta, p_id)
                    else:
                        # 均分batch到多次迭代
                        sample = self.buffer.sample(self.batch_size // self.train_interval_episode)
                    train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)
                    if self.use_per:
                        self.buffer.update_priorities(idxes, new_priorities, p_id)
                    self.train_infos.append(train_info)
        else:
            # 其他Q系：每个policy各采一批训练一次
            for p_id in self.policy_ids:
                if self.use_per:
                    beta = self.beta_anneal.eval(self.total_train_steps)
                    sample = self.buffer.sample(self.batch_size, beta, p_id)
                else:
                    sample = self.buffer.sample(self.batch_size)
                train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)
                if self.use_per:
                    self.buffer.update_priorities(idxes, new_priorities, p_id)
                self.train_infos.append(train_info)

        # CASEC：首次更新动作表征后，固定住并做一次硬更新（稳定）
        if self.algorithm_name == 'casec' and self.action_repr_updating:
            self.trainer.update_action_repr()
            self.action_repr_updating = False
            self.trainer.hard_target_updates()
            self.last_hard_update_episode = self.num_episodes_collected

        # 目标网络更新：优先软更新；否则按间隔硬更新
        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_adj(self):
        """邻接图策略训练：从adj_buffer按序列小块采样→多次迭代→PPO剪切更新adj_network。"""
        self.trainer.prep_training()
        # 收集训练过程指标（多次迭代取均值）
        self.train_adj_infos = {}

        data_chunk_length = 10  # 每次从轨迹中取多少步的连续片段来训练邻接（小BPTT段）

        # The 16-episode window overlaps across graph updates.  run35 showed
        # that continuing to replay stale graph decisions after clamp_ratio
        # has already exceeded 0.7 makes the graph PPO target drift instead of
        # improving reward.  Stop the current adj update early when either the
        # graph-level or factor-level PPO clamp fraction is too high.
        graph_clip_stop_ratio = float(
            getattr(self.args, "adj_ppo_clip_stop_ratio", 0.0)
        )
        factor_clip_stop_ratio = float(
            getattr(self.args, "adj_ppo_factor_clip_stop_ratio", 0.0)
        )
        min_ppo_epochs = max(
            1,
            int(getattr(self.args, "adj_ppo_min_epochs", 1)),
        )
        configured_recent_episode_window = max(
            0,
            int(getattr(self.args, "adj_recent_episode_window", 0)),
        )
        recent_episode_window = configured_recent_episode_window
        dynamic_recent_enabled = bool(
            getattr(self.args, "use_adj_dynamic_recent_window", False)
        )
        recent_window_shrunk = 0.0
        recent_window_recovered = 0.0
        recent_window_emergency_shrunk = 0.0
        recent_window_high_stale_count = 0.0
        recent_window_low_stale_count = 0.0
        if dynamic_recent_enabled and configured_recent_episode_window > 0:
            min_recent_window = max(
                1,
                min(
                    configured_recent_episode_window,
                    int(getattr(self.args, "adj_recent_episode_window_min", 1)),
                ),
            )
            graph_stale_threshold = float(
                getattr(self.args, "adj_recent_window_stale_threshold", 0.35)
            )
            factor_stale_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_factor_stale_threshold",
                    graph_stale_threshold,
                )
            )
            recover_graph_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_recover_stale_threshold",
                    -1.0,
                )
            )
            if recover_graph_threshold < 0.0:
                recover_graph_threshold = 0.8 * graph_stale_threshold
            recover_factor_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_recover_factor_stale_threshold",
                    -1.0,
                )
            )
            if recover_factor_threshold < 0.0:
                recover_factor_threshold = 0.8 * factor_stale_threshold
            shrink_patience = max(
                1,
                int(getattr(self.args, "adj_recent_window_shrink_patience", 1)),
            )
            recover_patience = max(
                1,
                int(getattr(self.args, "adj_recent_window_recover_patience", 2)),
            )
            severe_margin = max(
                0.0,
                float(getattr(self.args, "adj_recent_window_severe_margin", 0.15)),
            )
            emergency_recent_window = max(
                1,
                min(
                    configured_recent_episode_window,
                    int(
                        getattr(
                            self.args,
                            "adj_recent_episode_window_emergency",
                            1,
                        )
                    ),
                ),
            )
            emergency_graph_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_stale_threshold",
                    graph_stale_threshold + severe_margin,
                )
            )
            emergency_factor_threshold = float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_factor_stale_threshold",
                    factor_stale_threshold + severe_margin,
                )
            )
            prev_graph_stale = float(
                getattr(self, "_last_adj_graph_stale_ratio", np.nan)
            )
            prev_factor_stale = float(
                getattr(self, "_last_adj_factor_stale_ratio", np.nan)
            )
            adaptive_recent_window = int(
                getattr(
                    self,
                    "_adj_dynamic_recent_episode_window",
                    configured_recent_episode_window,
                )
            )
            adaptive_recent_window = max(
                min_recent_window,
                min(configured_recent_episode_window, adaptive_recent_window),
            )
            graph_too_stale = (
                np.isfinite(prev_graph_stale)
                and prev_graph_stale >= graph_stale_threshold
            )
            factor_too_stale = (
                np.isfinite(prev_factor_stale)
                and prev_factor_stale >= factor_stale_threshold
            )
            graph_fresh_enough = (
                np.isfinite(prev_graph_stale)
                and prev_graph_stale <= recover_graph_threshold
            )
            factor_fresh_enough = (
                np.isfinite(prev_factor_stale)
                and prev_factor_stale <= recover_factor_threshold
            )
            high_count = int(
                getattr(self, "_adj_recent_window_high_stale_count", 0)
            )
            low_count = int(
                getattr(self, "_adj_recent_window_low_stale_count", 0)
            )
            if graph_too_stale or factor_too_stale:
                high_count += 1
                low_count = 0
            elif graph_fresh_enough and factor_fresh_enough:
                low_count += 1
                high_count = 0
            else:
                high_count = 0
                low_count = 0

            emergency_graph_stale = (
                np.isfinite(prev_graph_stale)
                and prev_graph_stale >= emergency_graph_threshold
            )
            emergency_factor_stale = (
                np.isfinite(prev_factor_stale)
                and prev_factor_stale >= emergency_factor_threshold
            )
            if emergency_graph_stale or emergency_factor_stale:
                adaptive_recent_window = emergency_recent_window
                recent_window_emergency_shrunk = 1.0
                high_count = 0
                low_count = 0
            elif high_count >= shrink_patience:
                severe_graph_stale = (
                    np.isfinite(prev_graph_stale)
                    and prev_graph_stale >= graph_stale_threshold + severe_margin
                )
                severe_factor_stale = (
                    np.isfinite(prev_factor_stale)
                    and prev_factor_stale >= factor_stale_threshold + severe_margin
                )
                if severe_graph_stale or severe_factor_stale:
                    adaptive_recent_window = min_recent_window
                else:
                    adaptive_recent_window = max(
                        min_recent_window,
                        adaptive_recent_window - 1,
                    )
                high_count = 0
            elif (
                low_count >= recover_patience
                and adaptive_recent_window < configured_recent_episode_window
            ):
                adaptive_recent_window = min(
                    configured_recent_episode_window,
                    adaptive_recent_window + 1,
                )
                low_count = 0
                recent_window_recovered = 1.0

            recent_episode_window = adaptive_recent_window
            recent_window_shrunk = float(
                recent_episode_window < configured_recent_episode_window
            )
            recent_window_high_stale_count = float(high_count)
            recent_window_low_stale_count = float(low_count)
            self._adj_dynamic_recent_episode_window = adaptive_recent_window
            self._adj_recent_window_high_stale_count = high_count
            self._adj_recent_window_low_stale_count = low_count
        early_stop_triggered = 0.0
        last_epoch_clip_ratio = np.nan
        last_epoch_factor_clip_ratio = np.nan
        epochs_ran = 0
        sample_episode_count = np.nan
        sample_recent_fraction = np.nan
        sample_base_episode_count = np.nan
        sample_outcome_contrast_augmented_count = np.nan
        sample_outcome_positive_available = np.nan
        sample_outcome_negative_available = np.nan
        sample_outcome_positive_episode_count = np.nan
        sample_outcome_negative_episode_count = np.nan
        sample_outcome_class_complete = np.nan
        sample_outcome_support_exhausted = np.nan
        sample_outcome_credit_enabled = np.nan
        sample_outcome_cached_selection_reused = np.nan
        sample_outcome_support_round = np.nan
        sample_outcome_cross_update_reuse_count = np.nan
        sample_outcome_positive_available_count = np.nan
        sample_outcome_negative_available_count = np.nan
        sample_outcome_base_positive_count = np.nan
        sample_outcome_base_negative_count = np.nan
        sample_outcome_augmented_positive_count = np.nan
        sample_outcome_augmented_negative_count = np.nan
        sample_outcome_base_age_mean = np.nan
        sample_outcome_base_age_max = np.nan
        sample_outcome_augmented_age_mean = np.nan
        sample_outcome_augmented_age_max = np.nan
        sample_outcome_positive_support_generation = np.nan
        sample_outcome_negative_support_generation = np.nan
        sample_outcome_positive_support_age = np.nan
        sample_outcome_negative_support_age = np.nan
        sample_outcome_support_used_count = np.nan
        sample_outcome_support_used_fraction = np.nan
        sample_outcome_full_buffer_baseline = np.nan
        sample_outcome_base_cohort_baseline = np.nan
        sample_outcome_trained_cohort_baseline = np.nan
        sample_outcome_full_trained_baseline_gap = np.nan
        sample_outcome_trained_capture_episode_count = np.nan
        sample_outcome_cohort_centered_sum = np.nan
        sample_outcome_cohort_center_error = np.nan
        sample_outcome_cohort_center_valid = np.nan
        sample_outcome_positive_gate_episode_count = np.nan
        sample_outcome_negative_gate_episode_count = np.nan
        sample_outcome_positive_credit_episode_count = np.nan
        sample_outcome_negative_credit_episode_count = np.nan
        sample_outcome_signed_scaling_version = np.nan
        sample_outcome_graph_advantage_source_ready_fraction = np.nan
        sample_outcome_graph_confidence_mean = np.nan
        sample_outcome_graph_confidence_std = np.nan
        sample_outcome_graph_confidence_p50 = np.nan
        sample_outcome_graph_confidence_p95 = np.nan
        sample_outcome_graph_confidence_max = np.nan
        sample_outcome_positive_graph_confidence_mean = np.nan
        sample_outcome_positive_graph_confidence_max = np.nan
        sample_outcome_negative_graph_confidence_mean = np.nan
        sample_outcome_negative_graph_confidence_max = np.nan
        sample_outcome_graph_advantage_positive_fraction = np.nan
        sample_outcome_graph_advantage_negative_fraction = np.nan
        sample_outcome_graph_advantage_zero_fraction = np.nan
        sample_outcome_positive_zero_confidence_fraction = np.nan
        sample_outcome_negative_zero_confidence_fraction = np.nan
        sample_outcome_gate_to_credit_drop_fraction = np.nan
        sample_outcome_preclip_positive_mass = np.nan
        sample_outcome_preclip_negative_mass = np.nan
        sample_outcome_postclip_positive_mass = np.nan
        sample_outcome_postclip_negative_mass = np.nan
        sample_outcome_positive_clip_fraction = np.nan
        sample_outcome_negative_clip_fraction = np.nan
        sample_outcome_generation_update_count = np.nan
        sample_outcome_slot_overwrite_count = np.nan
        sample_outcome_generation_conflict_count = np.nan
        sample_outcome_invalid_used_state_count = np.nan

        self._adj_outcome_support_round = int(
            getattr(self, "_adj_outcome_support_round", 0)
        ) + 1
        outcome_support_round = self._adj_outcome_support_round

        for epoch_idx in range(self.adj_train_epochs):
            epoch_clip_ratios = []
            epoch_factor_clip_ratios = []
            for p_id in self.policy_info.keys():
                # 从该policy对应的邻接缓冲中，按“连续片段+mini-batch”方式迭代采样
                data_generator = self.adj_buffer.policy_buffers[p_id].sample_inds(data_chunk_length,
                                                                                  self.num_mini_batch,
                                                                                  recent_episode_window=recent_episode_window,
                                                                                  outcome_support_round=outcome_support_round)
                for sample in data_generator:
                    # 调用trainer的邻接训练：内部是PPO剪切 + 熵正则
                    train_adj_info, new_priorities, idxes = self.trainer.train_adj_on_batch(sample, self.use_adj_init)
                    # 聚合指标
                    for k, v in train_adj_info.items():
                        self.train_adj_infos.setdefault(k, []).append(v)
                    if "clamp_ratio" in train_adj_info:
                        epoch_clip_ratios.append(train_adj_info["clamp_ratio"])
                    if "factor_clamp_ratio" in train_adj_info:
                        epoch_factor_clip_ratios.append(
                            train_adj_info["factor_clamp_ratio"]
                        )
                policy_buffer = self.adj_buffer.policy_buffers[p_id]
                sample_episode_count = float(
                    getattr(policy_buffer, "last_sample_episode_count", np.nan)
                )
                sample_base_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_base_episode_count",
                    np.nan,
                ))
                sample_outcome_contrast_augmented_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_contrast_augmented_count",
                    np.nan,
                ))
                sample_outcome_positive_available = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_available",
                    np.nan,
                ))
                sample_outcome_negative_available = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_available",
                    np.nan,
                ))
                sample_outcome_positive_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_episode_count",
                    np.nan,
                ))
                sample_outcome_class_complete = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_class_complete",
                    np.nan,
                ))
                sample_outcome_support_exhausted = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_exhausted",
                    np.nan,
                ))
                sample_outcome_credit_enabled = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_credit_enabled",
                    np.nan,
                ))
                sample_outcome_cached_selection_reused = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cached_selection_reused",
                    np.nan,
                ))
                sample_outcome_support_round = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_round",
                    np.nan,
                ))
                sample_outcome_cross_update_reuse_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cross_update_reuse_count",
                    np.nan,
                ))
                sample_outcome_positive_available_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_available_count",
                    np.nan,
                ))
                sample_outcome_negative_available_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_available_count",
                    np.nan,
                ))
                sample_outcome_base_positive_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_positive_count",
                    np.nan,
                ))
                sample_outcome_base_negative_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_negative_count",
                    np.nan,
                ))
                sample_outcome_augmented_positive_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_positive_count",
                    np.nan,
                ))
                sample_outcome_augmented_negative_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_negative_count",
                    np.nan,
                ))
                sample_outcome_base_age_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_age_mean",
                    np.nan,
                ))
                sample_outcome_base_age_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_age_max",
                    np.nan,
                ))
                sample_outcome_augmented_age_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_age_mean",
                    np.nan,
                ))
                sample_outcome_augmented_age_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_augmented_age_max",
                    np.nan,
                ))
                sample_outcome_positive_support_generation = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_support_generation",
                    np.nan,
                ))
                sample_outcome_negative_support_generation = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_support_generation",
                    np.nan,
                ))
                sample_outcome_positive_support_age = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_support_age",
                    np.nan,
                ))
                sample_outcome_negative_support_age = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_support_age",
                    np.nan,
                ))
                sample_outcome_support_used_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_used_count",
                    np.nan,
                ))
                sample_outcome_support_used_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_support_used_fraction",
                    np.nan,
                ))
                sample_outcome_full_buffer_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_full_buffer_baseline",
                    np.nan,
                ))
                sample_outcome_base_cohort_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_base_cohort_baseline",
                    np.nan,
                ))
                sample_outcome_trained_cohort_baseline = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_trained_cohort_baseline",
                    np.nan,
                ))
                sample_outcome_full_trained_baseline_gap = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_full_trained_baseline_gap",
                    np.nan,
                ))
                sample_outcome_trained_capture_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_trained_capture_episode_count",
                    np.nan,
                ))
                sample_outcome_cohort_centered_sum = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_centered_sum",
                    np.nan,
                ))
                sample_outcome_cohort_center_error = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_center_error",
                    np.nan,
                ))
                sample_outcome_cohort_center_valid = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_cohort_center_valid",
                    np.nan,
                ))
                sample_outcome_positive_gate_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_gate_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_gate_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_gate_episode_count",
                    np.nan,
                ))
                sample_outcome_positive_credit_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_credit_episode_count",
                    np.nan,
                ))
                sample_outcome_negative_credit_episode_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_credit_episode_count",
                    np.nan,
                ))
                sample_outcome_signed_scaling_version = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_signed_scaling_version",
                    np.nan,
                ))
                sample_outcome_graph_advantage_source_ready_fraction = float(
                    getattr(
                        policy_buffer,
                        "last_sample_outcome_graph_advantage_source_ready_fraction",
                        np.nan,
                    )
                )
                sample_outcome_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_graph_confidence_std = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_std",
                    np.nan,
                ))
                sample_outcome_graph_confidence_p50 = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_p50",
                    np.nan,
                ))
                sample_outcome_graph_confidence_p95 = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_p95",
                    np.nan,
                ))
                sample_outcome_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_positive_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_positive_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_negative_graph_confidence_mean = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_graph_confidence_mean",
                    np.nan,
                ))
                sample_outcome_negative_graph_confidence_max = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_graph_confidence_max",
                    np.nan,
                ))
                sample_outcome_graph_advantage_positive_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_positive_fraction",
                    np.nan,
                ))
                sample_outcome_graph_advantage_negative_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_negative_fraction",
                    np.nan,
                ))
                sample_outcome_graph_advantage_zero_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_graph_advantage_zero_fraction",
                    np.nan,
                ))
                sample_outcome_positive_zero_confidence_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_zero_confidence_fraction",
                    np.nan,
                ))
                sample_outcome_negative_zero_confidence_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_zero_confidence_fraction",
                    np.nan,
                ))
                sample_outcome_gate_to_credit_drop_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_gate_to_credit_drop_fraction",
                    np.nan,
                ))
                sample_outcome_preclip_positive_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_preclip_positive_mass",
                    np.nan,
                ))
                sample_outcome_preclip_negative_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_preclip_negative_mass",
                    np.nan,
                ))
                sample_outcome_postclip_positive_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_postclip_positive_mass",
                    np.nan,
                ))
                sample_outcome_postclip_negative_mass = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_postclip_negative_mass",
                    np.nan,
                ))
                sample_outcome_positive_clip_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_positive_clip_fraction",
                    np.nan,
                ))
                sample_outcome_negative_clip_fraction = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_negative_clip_fraction",
                    np.nan,
                ))
                sample_outcome_generation_update_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_generation_update_count",
                    np.nan,
                ))
                sample_outcome_slot_overwrite_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_slot_overwrite_count",
                    np.nan,
                ))
                sample_outcome_generation_conflict_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_generation_conflict_count",
                    np.nan,
                ))
                sample_outcome_invalid_used_state_count = float(getattr(
                    policy_buffer,
                    "last_sample_outcome_invalid_used_state_count",
                    np.nan,
                ))
                filled_episode_count = float(
                    max(1, getattr(policy_buffer, "filled_i", 1))
                )
                if np.isfinite(sample_episode_count):
                    sample_recent_fraction = (
                        sample_episode_count / filled_episode_count
                    )
            # 首轮训练之后就不再使用“弱化初始化”分支
            self.use_adj_init = False
            epochs_ran = epoch_idx + 1
            if len(epoch_clip_ratios) > 0:
                last_epoch_clip_ratio = float(np.mean(epoch_clip_ratios))
            if len(epoch_factor_clip_ratios) > 0:
                last_epoch_factor_clip_ratio = float(
                    np.mean(epoch_factor_clip_ratios)
                )
            graph_stop = (
                graph_clip_stop_ratio > 0.0
                and np.isfinite(last_epoch_clip_ratio)
                and last_epoch_clip_ratio >= graph_clip_stop_ratio
            )
            factor_stop = (
                factor_clip_stop_ratio > 0.0
                and np.isfinite(last_epoch_factor_clip_ratio)
                and last_epoch_factor_clip_ratio >= factor_clip_stop_ratio
            )
            if epochs_ran >= min_ppo_epochs and (graph_stop or factor_stop):
                early_stop_triggered = 1.0
                break

        self.train_adj_infos.setdefault("adj_ppo_epochs_ran", []).append(
            float(epochs_ran)
        )
        self.train_adj_infos.setdefault(
            "adj_ppo_early_stop_triggered", []
        ).append(float(early_stop_triggered))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_clip_ratio", []
        ).append(float(last_epoch_clip_ratio))
        self.train_adj_infos.setdefault(
            "adj_ppo_last_epoch_factor_clip_ratio", []
        ).append(float(last_epoch_factor_clip_ratio))
        self.train_adj_infos.setdefault(
            "adj_ppo_clip_stop_ratio", []
        ).append(float(graph_clip_stop_ratio))
        self.train_adj_infos.setdefault(
            "adj_ppo_factor_clip_stop_ratio", []
        ).append(float(factor_clip_stop_ratio))
        self.train_adj_infos.setdefault("adj_ppo_min_epochs", []).append(
            float(min_ppo_epochs)
        )
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window", []
        ).append(float(recent_episode_window))
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window_config", []
        ).append(float(configured_recent_episode_window))
        self.train_adj_infos.setdefault(
            "adj_dynamic_recent_window_enabled", []
        ).append(float(dynamic_recent_enabled))
        self.train_adj_infos.setdefault(
            "adj_recent_window_shrunk", []
        ).append(float(recent_window_shrunk))
        self.train_adj_infos.setdefault(
            "adj_recent_window_recovered", []
        ).append(float(recent_window_recovered))
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_shrunk", []
        ).append(float(recent_window_emergency_shrunk))
        self.train_adj_infos.setdefault(
            "adj_recent_window_high_stale_count", []
        ).append(float(recent_window_high_stale_count))
        self.train_adj_infos.setdefault(
            "adj_recent_window_low_stale_count", []
        ).append(float(recent_window_low_stale_count))
        self.train_adj_infos.setdefault(
            "adj_sample_episode_count", []
        ).append(float(sample_episode_count))
        self.train_adj_infos.setdefault(
            "adj_sample_recent_fraction", []
        ).append(float(sample_recent_fraction))
        for metric_name, metric_value in (
                # v3 keeps v2 generation/round semantics and additionally
                # recenters outcome labels on the final optimizer cohort.
                ("adj_outcome_contrast_replay_support_version", 3.0),
                ("adj_sample_base_episode_count", sample_base_episode_count),
                (
                    "adj_sample_outcome_contrast_augmented_count",
                    sample_outcome_contrast_augmented_count,
                ),
                (
                    "adj_sample_outcome_positive_available",
                    sample_outcome_positive_available,
                ),
                (
                    "adj_sample_outcome_negative_available",
                    sample_outcome_negative_available,
                ),
                (
                    "adj_sample_outcome_positive_episode_count",
                    sample_outcome_positive_episode_count,
                ),
                (
                    "adj_sample_outcome_negative_episode_count",
                    sample_outcome_negative_episode_count,
                ),
                (
                    "adj_sample_outcome_class_complete",
                    sample_outcome_class_complete,
                ),
                ("adj_sample_outcome_support_exhausted", sample_outcome_support_exhausted),
                ("adj_sample_outcome_credit_enabled", sample_outcome_credit_enabled),
                ("adj_sample_outcome_cached_selection_reused", sample_outcome_cached_selection_reused),
                ("adj_sample_outcome_support_round", sample_outcome_support_round),
                ("adj_sample_outcome_cross_update_reuse_count", sample_outcome_cross_update_reuse_count),
                ("adj_sample_outcome_positive_available_count", sample_outcome_positive_available_count),
                ("adj_sample_outcome_negative_available_count", sample_outcome_negative_available_count),
                ("adj_sample_outcome_base_positive_count", sample_outcome_base_positive_count),
                ("adj_sample_outcome_base_negative_count", sample_outcome_base_negative_count),
                ("adj_sample_outcome_augmented_positive_count", sample_outcome_augmented_positive_count),
                ("adj_sample_outcome_augmented_negative_count", sample_outcome_augmented_negative_count),
                ("adj_sample_outcome_base_age_mean", sample_outcome_base_age_mean),
                ("adj_sample_outcome_base_age_max", sample_outcome_base_age_max),
                ("adj_sample_outcome_augmented_age_mean", sample_outcome_augmented_age_mean),
                ("adj_sample_outcome_augmented_age_max", sample_outcome_augmented_age_max),
                ("adj_sample_outcome_positive_support_generation", sample_outcome_positive_support_generation),
                ("adj_sample_outcome_negative_support_generation", sample_outcome_negative_support_generation),
                ("adj_sample_outcome_positive_support_age", sample_outcome_positive_support_age),
                ("adj_sample_outcome_negative_support_age", sample_outcome_negative_support_age),
                ("adj_sample_outcome_support_used_count", sample_outcome_support_used_count),
                ("adj_sample_outcome_support_used_fraction", sample_outcome_support_used_fraction),
                ("adj_sample_outcome_full_buffer_baseline", sample_outcome_full_buffer_baseline),
                ("adj_sample_outcome_base_cohort_baseline", sample_outcome_base_cohort_baseline),
                ("adj_sample_outcome_trained_cohort_baseline", sample_outcome_trained_cohort_baseline),
                ("adj_sample_outcome_full_trained_baseline_gap", sample_outcome_full_trained_baseline_gap),
                ("adj_sample_outcome_trained_capture_episode_count", sample_outcome_trained_capture_episode_count),
                ("adj_sample_outcome_cohort_centered_sum", sample_outcome_cohort_centered_sum),
                ("adj_sample_outcome_cohort_center_error", sample_outcome_cohort_center_error),
                ("adj_sample_outcome_cohort_center_valid", sample_outcome_cohort_center_valid),
                ("adj_sample_outcome_positive_gate_episode_count", sample_outcome_positive_gate_episode_count),
                ("adj_sample_outcome_negative_gate_episode_count", sample_outcome_negative_gate_episode_count),
                ("adj_sample_outcome_positive_credit_episode_count", sample_outcome_positive_credit_episode_count),
                ("adj_sample_outcome_negative_credit_episode_count", sample_outcome_negative_credit_episode_count),
                ("adj_sample_outcome_signed_scaling_version", sample_outcome_signed_scaling_version),
                (
                    "adj_sample_outcome_graph_advantage_source_ready_fraction",
                    sample_outcome_graph_advantage_source_ready_fraction,
                ),
                ("adj_sample_outcome_graph_confidence_mean", sample_outcome_graph_confidence_mean),
                ("adj_sample_outcome_graph_confidence_std", sample_outcome_graph_confidence_std),
                ("adj_sample_outcome_graph_confidence_p50", sample_outcome_graph_confidence_p50),
                ("adj_sample_outcome_graph_confidence_p95", sample_outcome_graph_confidence_p95),
                ("adj_sample_outcome_graph_confidence_max", sample_outcome_graph_confidence_max),
                ("adj_sample_outcome_positive_graph_confidence_mean", sample_outcome_positive_graph_confidence_mean),
                ("adj_sample_outcome_positive_graph_confidence_max", sample_outcome_positive_graph_confidence_max),
                ("adj_sample_outcome_negative_graph_confidence_mean", sample_outcome_negative_graph_confidence_mean),
                ("adj_sample_outcome_negative_graph_confidence_max", sample_outcome_negative_graph_confidence_max),
                ("adj_sample_outcome_graph_advantage_positive_fraction", sample_outcome_graph_advantage_positive_fraction),
                ("adj_sample_outcome_graph_advantage_negative_fraction", sample_outcome_graph_advantage_negative_fraction),
                ("adj_sample_outcome_graph_advantage_zero_fraction", sample_outcome_graph_advantage_zero_fraction),
                ("adj_sample_outcome_positive_zero_confidence_fraction", sample_outcome_positive_zero_confidence_fraction),
                ("adj_sample_outcome_negative_zero_confidence_fraction", sample_outcome_negative_zero_confidence_fraction),
                ("adj_sample_outcome_gate_to_credit_drop_fraction", sample_outcome_gate_to_credit_drop_fraction),
                ("adj_sample_outcome_preclip_positive_mass", sample_outcome_preclip_positive_mass),
                ("adj_sample_outcome_preclip_negative_mass", sample_outcome_preclip_negative_mass),
                ("adj_sample_outcome_postclip_positive_mass", sample_outcome_postclip_positive_mass),
                ("adj_sample_outcome_postclip_negative_mass", sample_outcome_postclip_negative_mass),
                ("adj_sample_outcome_positive_clip_fraction", sample_outcome_positive_clip_fraction),
                ("adj_sample_outcome_negative_clip_fraction", sample_outcome_negative_clip_fraction),
                ("adj_sample_outcome_generation_update_count", sample_outcome_generation_update_count),
                ("adj_sample_outcome_slot_overwrite_count", sample_outcome_slot_overwrite_count),
                ("adj_sample_outcome_generation_conflict_count", sample_outcome_generation_conflict_count),
                ("adj_sample_outcome_invalid_used_state_count", sample_outcome_invalid_used_state_count)):
            self.train_adj_infos.setdefault(metric_name, []).append(
                float(metric_value)
            )
        self.train_adj_infos.setdefault(
            "adj_recent_episode_window_emergency", []
        ).append(float(getattr(self.args, "adj_recent_episode_window_emergency", 1)))
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_stale_threshold", []
        ).append(
            float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_stale_threshold",
                    np.nan,
                )
            )
        )
        self.train_adj_infos.setdefault(
            "adj_recent_window_emergency_factor_stale_threshold", []
        ).append(
            float(
                getattr(
                    self.args,
                    "adj_recent_window_emergency_factor_stale_threshold",
                    np.nan,
                )
            )
        )
        graph_stale_values = self.train_adj_infos.get(
            "adj_graph_stale_ratio",
            [],
        )
        factor_stale_values = self.train_adj_infos.get(
            "adj_factor_stale_ratio",
            [],
        )
        if len(graph_stale_values) > 0:
            self._last_adj_graph_stale_ratio = float(
                np.nanmean(graph_stale_values)
            )
        if len(factor_stale_values) > 0:
            self._last_adj_factor_stale_ratio = float(
                np.nanmean(factor_stale_values)
            )

    """（非Q类）保存所有策略的actor/critic权重到 save_dir。"""

    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(), critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(), actor_save_path + '/actor.pt')

    """（QMix/VDN等）保存各policy的Q网络，以及混合器/额外网络。"""
    def save_q(self):
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 算法特定部件的保存
        if self.algorithm_name == "qtran":
            torch.save(self.trainer.eval_joint_q.state_dict(), self.save_dir + '/eval_joint_q.pt')
            torch.save(self.trainer.v.state_dict(), self.save_dir + '/v.pt')
        elif self.algorithm_name == "wqmix":
            for pid in self.policy_ids:
                policy_Q = self.central_policies[pid].q_network
                p_save_path = self.save_dir + '/' + str(pid)
                if not os.path.exists(p_save_path):
                    os.makedirs(p_save_path)
                torch.save(policy_Q.state_dict(), p_save_path + '/central_q_network.pt')
            torch.save(self.trainer.mixer.state_dict(), self.save_dir + '/mixer.pt')
            torch.save(self.trainer.central_mixer.state_dict(), self.save_dir + '/central_mixer.pt')
        else:
            torch.save(self.trainer.mixer.state_dict(), self.save_dir + '/mixer.pt')

    """（带图+中心化）保存：adj_network、各policy的rnn/q/v等多组件。"""
    def save_q_mdfg_cent(self):
        save_path = self.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 邻接网络
        adj = self.adj_network
        torch.save(adj.state_dict(), save_path + '/adj_network.pt')

        # 每个policy的rnn/q/v（以及casec专有的动作表征等）
        for pid in self.policy_ids:
            p_save_path = save_path + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            rnn_Q = self.policies[pid].rnn_network
            torch.save(rnn_Q.state_dict(), p_save_path + '/rnn_network.pt')

            if self.algorithm_name == 'casec':
                if self.independent_p_q:
                    p_rnn = self.policies[pid].p_rnn_network
                    torch.save(p_rnn.state_dict(), p_save_path + '/p_rnn_network.pt')
                action_encoder = self.policies[pid].action_encoder
                torch.save(action_encoder.state_dict(), p_save_path + '/action_encoder.pt')
                p_action_repr = self.policies[pid].p_action_repr
                torch.save(p_action_repr, p_save_path + '/p_action_repr.pt')
                action_repr = self.policies[pid].action_repr
                torch.save(action_repr, p_save_path + '/action_repr.pt')
            else:
                rnn_v = self.policies[pid].rnn_critic_network
                torch.save(rnn_v.state_dict(), p_save_path + '/rnn_critic_network.pt')

            if self.use_vfunction:
                policy_vtot = self.policies[pid].vtot_network
                torch.save(policy_vtot.state_dict(), p_save_path + '/vtot_network.pt')

            for num_orders in range(1, self.highest_orders + 1):
                policy_Q = self.policies[pid].q_network[num_orders]
                torch.save(policy_Q.state_dict(), p_save_path + f'/q_network_{num_orders}.pt')
                if self.use_vfunction:
                    policy_V = self.policies[pid].v_network[num_orders]
                    torch.save(policy_V.state_dict(), p_save_path + f'/v_network_{num_orders}.pt')

    """（非Q类）从 model_dir 恢复：actor/critic 权重到当前policy。"""
    def restore(self):
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')
            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    # ==================== 修改点 A：新增通用指定目录保存函数 ====================
    def save_to_dir(self, save_dir):
        """
        将当前模型保存到指定目录。
        对 ddfg/sddfg，会复用 self.saver，也就是 save_q_mdfg_cent()。
        """
        old_save_dir = self.save_dir
        self.save_dir = str(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver()

        self.save_dir = old_save_dir

    # ==================== 修改点 B：新增 best checkpoint 保存函数 ====================
    def _safe_float(self, x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _latest_train_metric(self, key, default=np.nan):
        """
        Read a scalar from the latest adjacency or policy training result.

        Adjacency metrics such as clamp_ratio are stored as lists in
        train_adj_infos, whereas policy losses are stored as dictionaries in
        train_infos. Prefer the adjacency source because it is the authoritative
        source for graph-policy tie-break metrics.
        """
        adj_info = getattr(self, "train_adj_infos", None)
        if isinstance(adj_info, dict) and key in adj_info:
            try:
                values = np.asarray(adj_info[key], dtype=np.float64).reshape(-1)
                if values.size > 0 and np.all(np.isfinite(values)):
                    return float(values.mean())
            except Exception:
                pass

        infos = getattr(self, "train_infos", None)
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and key in info:
                    return self._safe_float(info.get(key), default)
        return default

    def _is_better_best_win(self, eval_win, eval_reward, clamp_ratio, step):
        """
        best_win 比较优先级：
        1) eval_win_rate；
        2) eval_average_episode_rewards；
        3) clamp_ratio；
        4) later step。
        """
        win_eps = 1e-12
        reward_eps = float(getattr(self.args, "best_reward_eps", 1e-3))
        clamp_eps = float(getattr(self.args, "best_clamp_eps", 1e-4))

        if not hasattr(self, "best_eval_win_rate"):
            self.best_eval_win_rate = -np.inf
        if not hasattr(self, "best_eval_win_reward"):
            self.best_eval_win_reward = -np.inf
        if not hasattr(self, "best_eval_win_clamp_ratio"):
            self.best_eval_win_clamp_ratio = np.inf
        if not hasattr(self, "best_eval_win_step"):
            self.best_eval_win_step = -1

        # 1. win_rate 更高
        if eval_win > self.best_eval_win_rate + win_eps:
            return True

        # win_rate 更低
        if eval_win < self.best_eval_win_rate - win_eps:
            return False

        # 2. win_rate 相同，reward 更高
        if eval_reward > self.best_eval_win_reward + reward_eps:
            return True

        if eval_reward < self.best_eval_win_reward - reward_eps:
            return False

        # 3. reward 接近，比较 clamp_ratio
        clamp_valid = np.isfinite(clamp_ratio)
        best_clamp_valid = np.isfinite(self.best_eval_win_clamp_ratio)

        if clamp_valid and best_clamp_valid:
            if clamp_ratio < self.best_eval_win_clamp_ratio - clamp_eps:
                return True
            if clamp_ratio > self.best_eval_win_clamp_ratio + clamp_eps:
                return False
        elif clamp_valid and not best_clamp_valid:
            return True

        # 4. 最后比较更晚 step
        return int(step) > int(self.best_eval_win_step)

    def maybe_save_best_eval(self, eval_info):
        """
        根据 eval 指标保存 best checkpoint。
        best_reward: 只按 eval_average_episode_rewards。
        best_win: eval_win_rate -> eval_average_episode_rewards -> clamp_ratio -> later step。
        """
        if eval_info is None:
            return

        eval_reward = float(eval_info.get("eval_average_episode_rewards", -np.inf))
        eval_win = float(eval_info.get("eval_win_rate", -np.inf))
        step = int(self.total_env_steps)

        if not hasattr(self, "best_eval_reward"):
            self.best_eval_reward = -np.inf

        best_root = os.path.join(str(self.run_dir), "best_models")
        if not os.path.exists(best_root):
            os.makedirs(best_root)

        # ---------- best_reward ----------
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            reward_dir = os.path.join(best_root, "best_reward")
            self.save_to_dir(reward_dir)

            pd.DataFrame([{
                "step": step,
                "best_eval_reward": float(eval_reward),
                "eval_win_rate": float(eval_win),
            }]).to_csv(
                os.path.join(best_root, "best_reward_info.csv"),
                index=False
            )

            print(
                f"[BEST] Save best_reward checkpoint at step={step}, "
                f"eval_reward={eval_reward:.6f}, eval_win_rate={eval_win:.6f}"
            )

        # ---------- best_win ----------
        clamp_ratio = self._latest_train_metric("clamp_ratio", np.nan)

        if self._is_better_best_win(
                eval_win=eval_win,
                eval_reward=eval_reward,
                clamp_ratio=clamp_ratio,
                step=step,
        ):
            self.best_eval_win_rate = eval_win
            self.best_eval_win_reward = eval_reward
            self.best_eval_win_clamp_ratio = clamp_ratio
            self.best_eval_win_step = step

            win_dir = os.path.join(best_root, "best_win")
            self.save_to_dir(win_dir)

            pd.DataFrame([{
                "step": step,
                "best_eval_win_rate": float(eval_win),
                "eval_average_episode_rewards": float(eval_reward),
                "clamp_ratio": float(clamp_ratio) if np.isfinite(clamp_ratio) else np.nan,
            }]).to_csv(
                os.path.join(best_root, "best_win_info.csv"),
                index=False
            )

            clamp_text = f"{clamp_ratio:.6f}" if np.isfinite(clamp_ratio) else "N/A"
            print(
                f"[BEST] Save best_win checkpoint at step={step}, "
                f"eval_win_rate={eval_win:.6f}, "
                f"eval_reward={eval_reward:.6f}, "
                f"clamp_ratio={clamp_text}"
            )

    """（QMix/VDN等）从 model_dir 恢复：各policy的Q网络 & mixer。"""

    def restore_q(self):
        import os
        base_dir = str(self.model_dir) if self.model_dir is not None else ""
        if base_dir and not base_dir.endswith(os.sep):
            base_dir += os.sep

        for pid in self.policy_ids:
            path = os.path.join(base_dir, str(pid))
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(os.path.join(path, "q_network.pt"), map_location=self.device)
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)

        mixer_path = os.path.join(base_dir, "mixer.pt")
        if hasattr(self.trainer, "mixer") and self.trainer.mixer is not None and os.path.exists(mixer_path):
            policy_mixer_state_dict = torch.load(mixer_path, map_location=self.device)
            self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    """只恢复邻接网络（当pretrain_adj=True时用）。"""

    def load_adj(self):
        import os
        path_adj = str(self.model_dir) if self.model_dir is not None else ""
        if path_adj != "" and (not path_adj.endswith(os.sep)):
            path_adj = path_adj + os.sep
        adj_path = os.path.join(path_adj, "adj_network.pt")
        adj_state_dict = torch.load(adj_path, map_location=self.device)
        self.adj_network.load_state_dict(adj_state_dict)

    """（带图+中心化）从 model_dir 恢复 adj/rnn/q/v 等多组件。"""

    def restore_mdfg_cent(self):
        import os

        base_dir = str(self.model_dir) if self.model_dir is not None else ""
        if base_dir != "" and (not base_dir.endswith(os.sep)):
            base_dir = base_dir + os.sep

        adj_path = os.path.join(base_dir, "adj_network.pt")
        adj_state_dict = torch.load(adj_path, map_location=self.device)
        self.adj_network.load_state_dict(adj_state_dict)

        for pid in self.policy_ids:
            policy_dir = os.path.join(base_dir, str(pid))
            print("load the pretrained model from {}".format(policy_dir))

            rnn_state_dict = torch.load(os.path.join(policy_dir, "rnn_network.pt"), map_location=self.device)
            self.policies[pid].rnn_network.load_state_dict(rnn_state_dict)

            rnn_critic_state_dict = torch.load(os.path.join(policy_dir, "rnn_critic_network.pt"),
                                               map_location=self.device)
            self.policies[pid].rnn_critic_network.load_state_dict(rnn_critic_state_dict)

            if self.use_vfunction:
                vtot_dict = torch.load(os.path.join(policy_dir, "vtot_network.pt"), map_location=self.device)
                self.policies[pid].vtot_network.load_state_dict(vtot_dict)

            for num_orders in range(1, self.highest_orders + 1):
                q_path = os.path.join(policy_dir, f"q_network_{num_orders}.pt")
                policy_q_state_dict = torch.load(q_path, map_location=self.device)
                self.policies[pid].q_network[num_orders].load_state_dict(policy_q_state_dict)

                if self.use_vfunction:
                    v_path = os.path.join(policy_dir, f"v_network_{num_orders}.pt")
                    policy_v_state_dict = torch.load(v_path, map_location=self.device)
                    self.policies[pid].v_network[num_orders].load_state_dict(policy_v_state_dict)

    def log(self):
        """（抽象）写训练与采样相关日志。子类中具体实现。"""
        raise NotImplementedError

    def log_clear(self):
        """（抽象）清理日志缓存。子类中具体实现。"""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        打印/写入“环境相关信息”（如平均奖励、成功率等），并保存到CSV中。
        :param env_info: dict，键是指标名，值是一个list（累积的值）
        :param suffix: 在键名后附加的后缀（如 'eval_'）
        """
        row = {"step": self.total_env_steps}

        for k, v in env_info.items():
            if len(v) > 0:
                mean_v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k
                row[suffix_k] = mean_v

                print(suffix_k + " is " + str(mean_v))
                if self.use_wandb:
                    wandb.log({suffix_k: mean_v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(suffix_k, mean_v, self.total_env_steps)

        filename = 'progress_eval.csv' if suffix == "eval_" else 'progress.csv'
        self._append_scalar_csv(filename, row)

    def log_train(self, policy_id, train_info):
        """
        写“训练指标”日志（如loss、grad_norm等），每个policy单独一列。
        :param policy_id: 哪个policy
        :param train_info: dict，训练过程的标量
        """
        row = {"step": self.total_env_steps, "policy_id": str(policy_id)}
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            row[k] = v
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalar(policy_k, v, self.total_env_steps)

        self._append_scalar_csv('progress_train.csv', row)

    def log_train_adj(self, train_adj_info):
        """
        写“邻接图训练”的指标（如rl_loss/entropy_loss/grad_norm等）。
        train_adj_info: dict，里面每个key是列表（多次迭代的值），这里取均值再写。
        """
        row = {"step": self.total_env_steps}
        for k, v in train_adj_info.items():
            if len(v) > 0:
                v = float(np.mean(v))
            else:
                v = 0.0
            row[k] = v
            # Missing diagnostics deliberately use NaN so CSV readers can
            # distinguish "not observed" from a real zero. TensorBoard rejects
            # non-finite scalars and emitted thousands of misleading warnings
            # in run62; keep the CSV sentinel but do not write it as an event.
            if np.isfinite(v):
                if self.use_wandb:
                    wandb.log({"adj/" + k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(
                        "adj/" + k, v, self.total_env_steps
                    )

        self._append_scalar_csv('progress_train_adj.csv', row)

    def collect_rollout(self):
        """（抽象）与环境交互、采样一整段episode，并写入buffer。子类需实现。"""
        raise NotImplementedError
