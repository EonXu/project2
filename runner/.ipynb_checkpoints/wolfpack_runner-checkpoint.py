import numpy as np
import torch
import time

from runner.base_runner import RecRunner

class WolfpackRunner(RecRunner):
    """
    在 reset/step 时额外处理 exist_mask，并传入 adj_network.sample。
    直接从 obs 的最后一维抽 exist_flag 构造 exist_mask（>0 表示该 slot 存在）
    avail_acts: 存在的 agent 全 1；不存在的 agent 全 0
    """

    def __init__(self, config):
        super(WolfpackRunner, self).__init__(config)

        # [修复] 初始化 train_infos，防止在第一次训练前 log 报错
        self.train_infos = []

        # 预热：随机策略采样若干回合填充经验池
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.start = time.time()
        self.warmup(num_warmup_episodes)
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        self.log_clear()

    def eval(self):
        """评估若干回合并做日志。"""
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collect_rollout(explore=False, training_episode=False, warmup=False)

            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        收集一个完整 episode 并入缓冲，所有智能体共享同一个策略。
        新增 exist_mask 的获取与传递。
        :param explore: (bool) 收集此回合时是否使用探索策略。
        :param training_episode: (bool) 此回合用于评估还是训练。
        :param warmup: (bool) 此回合是否在预热阶段收集。
        :return env_info: (dict) 包含有关展开数据的信息（总奖励等）。
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        """ ---------- reset：环境只给 obs，这里补齐 share_obs / avail_acts / exist_mask----------"""
        obs, share_obs, avail_acts = env.reset()

        self.act_dim = policy.output_dim

        last_acts_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.act_dim), dtype=np.float32)
        rnn_states_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.hidden_size), dtype=np.float32)

        # ------- episode 级缓存（shape 与 SMACRunner/PREYRunner 完全一致） -------
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_adj = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.num_factor), dtype=np.int64) for p_id in self.policy_ids}
        episode_prob_adj = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.num_factor), dtype=np.float32) for p_id in self.policy_ids}
        episode_qtot = {p_id: np.zeros((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        # f_v/f_q 在 PREY/SMAC 里使用了 (num_factor + num_agents) 的槽位；保持一致便于 Trainer 复用
        episode_f_v = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_factor + self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_f_q = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_factor + self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_rnn_states = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32) for p_id in self.policy_ids}

        # dones 初始化（与 SMAC/PREY 保持相同形状）
        dones = np.zeros((self.num_envs, self.num_agents, 1), dtype=np.bool_)

        # -------- 采集整回合 --------
        t = 0
        while t < self.episode_length:
            obs_batch = np.concatenate(obs)                    # [num_envs*num_agents, obs_dim]
            states_batch = np.concatenate(share_obs)           # [num_envs*num_agents, state_dim] 或拼接后的中心态
            avail_acts_batch = np.concatenate(avail_acts)      # [num_envs*num_agents, act_dim]

            # ------ 与 SMACRunner/PREYRunner 同：先更新 RNN hidden ------
            if self.algorithm_name in self.adj_correlation: # get actions for all agents to step the env
                _, rnn_states_batch, _ = policy.get_hidden_states(obs_batch, last_acts_batch, rnn_states_batch)

                if self.use_dyn_graph:
                    prob_adj, adj, _ = self.adj_network.sample(
                        obs_batch[None, :], rnn_states_batch.unsqueeze(0),
                        self.use_adj_init, dones, explore, self.total_env_steps
                    )
                    adj_all = torch.cat(
                        [adj.cpu().detach(), torch.eye(self.num_agents, dtype=torch.int64).unsqueeze(0)], dim=2)
                else:
                    prob_adj = torch.zeros((1, self.num_agents, self.num_factor), dtype=torch.float32)
                    adj = self.adj
                    adj_all = adj.unsqueeze(0)

                # 动作选择
                if warmup:
                    acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                else:
                    if self.algorithm_name == 'casec':
                        acts_batch, qtot, _, f_q = policy.get_actions(rnn_states_batch.unsqueeze(0),
                                                                      torch.tensor(avail_acts_batch),
                                                                      t_env=self.total_env_steps,
                                                                      explore=explore)
                    else:
                        acts_batch, qtot, _, f_q = policy.get_actions(
                            obs_batch[None, :],
                            rnn_states_batch.unsqueeze(0),
                            torch.tensor(avail_acts_batch).to(self.device),
                            t_env=self.total_env_steps,
                            explore=explore,
                            adj_input=adj_all.to(self.device),
                            no_sequence=False,
                            dones=torch.tensor(dones).to(self.device)
                        )
                    if self.use_vfunction:
                        f_v = policy.get_v_values(
                            rnn_states_batch.unsqueeze(0), states_batch[None, :],
                            adj_all.to(self.device), no_sequence=False,
                            dones=torch.tensor(dones).to(self.device)
                        )
            # else:
            #     # 非 DDFG 算法（基本用不到），保持与 MPERunner/SMACRunner 一致
            #     if warmup:
            #         acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
            #         _, rnn_states_batch, _ = policy.get_actions(obs_batch, last_acts_batch, rnn_states_batch, t_env=None, explore=True)
            #     else:
            #         acts_batch, rnn_states_batch, _ = policy.get_actions(
            #             obs_batch, last_acts_batch, rnn_states_batch, avail_acts_batch,
            #             t_env=self.total_env_steps, explore=explore)
            #     prob_adj = torch.zeros((self.num_agents, self.num_factor), dtype=torch.float32)
            #     adj = torch.zeros((self.num_agents, self.num_factor), dtype=torch.int64)

            # numpy 化 & 缓存
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            """ ------- 关键差异：Wolfpack 的 step 返回 4 元组，这里打成 DDFG 期望的 6 元组 -------"""
            env_acts_ind = [np.argmax(a, axis=-1) for a in env_acts]
            next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(env_acts_ind)

            # =================== [修复] ===================
            # 强制扩维：将 (Batch, N) -> (Batch, N, 1) 以匹配 Buffer 形状
            if len(rewards.shape) == 2:
                rewards = np.expand_dims(rewards, axis=-1)

            if len(dones.shape) == 2:
                dones = np.expand_dims(dones, axis=-1)
            # ===================================================

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)

            '''for k in range(self.num_envs):
                 reward_norm = self.reward_scaling(rewards[k,0])
                 rewards[k] = reward_norm'''

            # --------- 写入 episode 缓存（与 SMAC/PREY 完全一致） ---------
            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = env_acts
            episode_rewards[p_id][t] = rewards
            episode_rnn_states[p_id][t] = rnn_states_batch

            # here dones store agent done flag of the next step
            if self.algorithm_name in self.adj_correlation:
                env_adj = adj.cpu().detach().numpy()[0] if self.algorithm_name in ["ddfg", "rddfg_low"] else adj.cpu().detach().numpy()
                env_prob_adj = prob_adj.cpu().detach().numpy()[0] if self.algorithm_name in ["ddfg", "rddfg_low"] else prob_adj.cpu().detach().numpy()
                episode_adj[p_id][t] = env_adj
                episode_prob_adj[p_id][t] = env_prob_adj
                if self.algorithm_name in ["ddfg", "rddfg_low"] and not warmup:
                    episode_qtot[p_id][t] = qtot.cpu().detach().numpy()
                    episode_f_q[p_id][t] = f_q.cpu().detach().numpy()
                    if self.use_vfunction:
                        episode_f_v[p_id][t] = f_v.cpu().detach().numpy()

            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            episode_avail_acts[p_id][t] = avail_acts

            # 时间推进
            t += 1
            obs = next_obs
            share_obs = next_share_obs
            avail_acts = next_avail_acts

        # 末帧写入（与 SMAC/PREY 一致）
        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = share_obs
        episode_avail_acts[p_id][t] = avail_acts

        if self.algorithm_name in self.adj_correlation:
            obs_batch = np.concatenate(obs)
            _, rnn_states_batch, _ = policy.get_hidden_states(obs_batch, last_acts_batch, rnn_states_batch)
            if self.use_dyn_graph:
                prob_adj, adj, _ = self.adj_network.sample(
                    obs_batch[None, :], rnn_states_batch.unsqueeze(0),
                    self.use_adj_init, dones, explore, self.total_env_steps
                )
            else:
                prob_adj = torch.zeros((self.num_agents, self.num_factor), dtype=torch.float32)
                adj = self.adj
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch,
                                                              np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            env_adj = adj.cpu().detach().numpy()[0] if self.algorithm_name in ["ddfg","rddfg_low"] else adj.cpu().detach().numpy()
            env_prob_adj = prob_adj.cpu().detach().numpy()[0] if self.algorithm_name in ["ddfg","rddfg_low"] else prob_adj.cpu().detach().numpy()
            episode_adj[p_id][t] = env_adj
            episode_prob_adj[p_id][t] = env_prob_adj
            episode_rnn_states[p_id][t] = rnn_states_batch

        # 入缓冲区 & 统计
        if explore:
            self.num_episodes_collected += self.num_envs
            ind = self.buffer.insert(self.num_envs,# push all episodes collected in this rollout step to the buffer
                                     episode_obs,
                                     episode_share_obs,
                                     episode_acts,
                                     episode_rewards,
                                     episode_dones,
                                     episode_dones_env,
                                     episode_avail_acts,
                                     episode_adj,
                                     episode_prob_adj)
            if self.algorithm_name in ["ddfg","rddfg_low"] and self.total_env_steps >= self.adj_begin_step and (not warmup):
                self.num_adj_episodes_collected += self.num_envs
                rewards = self.buffer.norm_reward(ind)
                idx = self.adj_buffer.insert(self.num_envs,
                                             episode_obs,
                                             episode_share_obs,
                                             episode_acts,
                                             rewards,
                                             episode_dones,
                                             episode_dones_env,
                                             episode_avail_acts,
                                             episode_adj,
                                             episode_prob_adj,
                                             episode_qtot,
                                             episode_f_v,
                                             episode_f_q,
                                             episode_rnn_states)

                self.adj_buffer.compute_advantage(idx)

        env_info['average_episode_rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])
        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)
        '''if self.use_dyn_graph:
            self.log_train_adj(p_id, self.train_adj_infos[0])'''

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
