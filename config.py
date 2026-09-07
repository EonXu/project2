import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="DDFG", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="ddfg", choices=[
        "qtran", "qplex", "wqmix", "qmix", "vdn", "ddfg", "sopcg", "casec", "sddfg"])
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True)
    parser.add_argument('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument('--n_eval_rollout_threads', type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument('--num_env_steps', type=int,
                        default=2000000, help="Number of env steps to train for")
    parser.add_argument('--use_wandb', action='store_false', default=True,
                        help="Whether to use weights&biases, if not, use tensorboardX instead")
    parser.add_argument('--user_name', type=str, default="zoeyuchao")

    # env parameters
    parser.add_argument('--env_name', type=str, default="wolfpack")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument('--episode_length', type=int,
                        default=200, help="Max length for any episode")
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--adj_buffer_size', type=int, default=32,
                        help="Max # of transitions that adj replay buffer can contain")
    parser.add_argument(
        "--adj_recent_episode_window",
        type=int,
        default=0,
        help=(
            "If positive, train the adjacency policy only from the most recent "
            "N episodes in the adjacency replay buffer while still using the "
            "full buffer for return baselines. This keeps graph PPO near "
            "on-policy when topology distributions drift quickly."
        ),
    )
    parser.add_argument('--use_reward_normalization', action='store_true',
                        default=False, help="Whether to normalize rewards in replay buffer")
    parser.add_argument('--use_popart', action='store_true', default=False,
                        help="Whether to use popart to normalize the target loss")
    parser.add_argument('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")

    # prioritized experience replay
    parser.add_argument('--use_per', action='store_true', default=False,
                        help="Whether to use prioritized experience replay")
    parser.add_argument('--per_nu', type=float, default=0.9,
                        help="Weight of max TD error in formation of PER weights")
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help="Alpha term for prioritized experience replay")
    parser.add_argument('--per_eps', type=float, default=1e-6,
                        help="Eps term for prioritized experience replay")
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help="Starting beta term for prioritized experience replay")

    # network parameters
    parser.add_argument("--use_centralized_Q", action='store_false',
                        default=True, help="Whether to use centralized Q function")
    parser.add_argument('--share_policy', action='store_false',
                        default=True, help="Whether agents share the same policy")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument('--use_feature_normalization', action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument('--use_orthogonal', action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_cell", action='store_true',
                        default=False, help="Whether to use GRUCell")

    # recurrent parameters
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    parser.add_argument("--use_rnn_layer", action='store_false',
                        default=True, help='Whether to use a recurrent policy')
    parser.add_argument("--use_naive_recurrent_policy", action='store_false',
                        default=True, help='Whether to use a naive recurrent policy')
    # TODO now only 1 is support
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument('--data_chunk_length', type=int, default=200,
                        help="Time length of chunks used to train via BPTT")
    parser.add_argument('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")

    # attn parameters
    parser.add_argument("--attn", action='store_true', default=False)
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True)
    parser.add_argument("--use_cat_self", action='store_false', default=True)

    # optimizer parameters
    parser.add_argument('--adj_lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    parser.add_argument('--critic_lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--opti_alpha", type=float, default=0.99,
                        help='RMSProp alpha')
    parser.add_argument("--weight_decay", type=float, default=0)

    # algo common parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for env")
    parser.add_argument(
        '--q_n_step',
        type=int,
        default=1,
        help=(
            "Terminal-gated finite-horizon TD return used by SDDFG recurrent "
            "Q learning. Values above 1 apply the full return only to "
            "transitions whose reachable window contains a real terminal-win "
            "marker; all other transitions retain exact one-step TD. The "
            "default 1 preserves historical one-step behavior."
        ),
    )
    parser.add_argument(
        '--q_terminal_replay_lane',
        action='store_true',
        default=False,
        help=(
            "For SDDFG terminal-gated Q learning, preserve every uniform "
            "batch and, once per train interval, append one terminal-win "
            "episode only when the uniform draw missed all such episodes. "
            "Only its terminal-gated transitions contribute to the loss."
        ),
    )
    parser.add_argument(
        '--q_terminal_replay_loss_weight',
        type=float,
        default=0.10,
        help=(
            "Per-transition optimizer weight for the appended terminal-"
            "credit replay lane. Uniform replay, including natural terminal "
            "samples, remains weight 1.0. The formal 0.10 default is the "
            "run159 regression fix for repeated high-error auxiliary MSE."
        ),
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--use_huber_loss', action='store_true',
                        default=False, help="Whether to use Huber loss for critic update")
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # soft update parameters
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    parser.add_argument('--hard_update_interval_episode', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")

    # sddfg parameters
    parser.add_argument("--gat_heads", type=int, default=4, help="Number of multi-head attention for GAT")
    parser.add_argument("--gat_negative_slope", type=float, default=0.2, help="LeakyReLU negative slope in GAT")
    parser.add_argument("--gat_hyperedge_hidden", type=int, default=32,
                        help="Hidden dimension of pair-to-hyperedge scorer for 3-order SDDFG")
    parser.add_argument(
        "--adj_order3_bonus",
        type=float,
        default=1.0,
        help=(
            "Multiplicative prior for third-order SDDFG hyperedge scores. "
            "Values above one counteract pair-score argmax collapse in "
            "tasks where three-agent coordination is important."
        ),
    )
    parser.add_argument(
        "--adj_order3_bonus_start",
        type=float,
        default=-1.0,
        help=(
            "Initial third-order hyperedge score multiplier. A negative "
            "value keeps --adj_order3_bonus constant from the beginning."
        ),
    )
    parser.add_argument(
        "--adj_order3_bonus_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps used to anneal adj_order3_bonus_start to "
            "--adj_order3_bonus. Zero disables this annealing."
        ),
    )
    parser.add_argument(
        "--adj_sampling_temperature_start",
        type=float,
        default=1.0,
        help=(
            "Initial temperature for stochastic SDDFG graph sampling during "
            "training rollouts. One preserves the raw constrained policy."
        ),
    )
    parser.add_argument(
        "--adj_sampling_temperature_final",
        type=float,
        default=1.0,
        help=(
            "Final temperature for stochastic SDDFG graph sampling. Values "
            "below one anneal training graphs toward eval-time argmax graphs."
        ),
    )
    parser.add_argument(
        "--adj_sampling_temperature_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps used to anneal graph sampling temperature. "
            "Zero keeps the initial temperature."
        ),
    )
    parser.add_argument(
        "--use_train_consistent_eval_graph",
        action="store_true",
        default=False,
        help=(
            "Sample eval-time SDDFG graphs from the current training graph "
            "behavior distribution while keeping policy actions greedy. "
            "Evaluation uses an isolated graph RNG."
        ),
    )
    parser.add_argument(
        "--use_adj_topology_persistence",
        action="store_true",
        default=False,
        help=(
            "Use the existing greedy-mixture mass to retain the previous "
            "factor at the same slot when it remains eligible. The exact "
            "Markov behavior probability is stored for adjacency PPO."
        ),
    )
    parser.add_argument(
        "--adj_min_order3_ratio_start",
        type=float,
        default=0.0,
        help=(
            "Initial minimum fraction of SDDFG factor slots reserved for "
            "third-order candidates. Zero disables the quota mechanism."
        ),
    )
    parser.add_argument(
        "--adj_min_order3_ratio_final",
        type=float,
        default=0.0,
        help=(
            "Final minimum fraction of factor slots reserved for third-order "
            "candidates. The quota is applied only when at least three agents "
            "are active and feasible triplets exist."
        ),
    )
    parser.add_argument(
        "--adj_min_order3_ratio_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps used to anneal the third-order reservation "
            "ratio. Zero keeps the final value."
        ),
    )
    parser.add_argument(
        "--adj_order3_quota_score_floor",
        type=float,
        default=0.0,
        help=(
            "Relative score floor for forced third-order quota choices. "
            "The reference is the best currently eligible pair when one "
            "exists, otherwise the best eligible candidate. This prevents "
            "the order3 quota from preserving very weak triplets just to "
            "meet a count target."
        ),
    )
    parser.add_argument(
        "--adj_order3_quota_mode",
        type=str,
        default="hard",
        help=(
            "How the lower order3 quota is applied in SDDFG graph sampling. "
            "'hard' keeps the legacy masking behavior. 'soft' converts the "
            "quota deficit into a probability bonus for high-quality triplets "
            "instead of forcing triplets when their reward-driven credit is "
            "negative."
        ),
    )
    parser.add_argument(
        "--adj_order3_soft_quota_coef",
        type=float,
        default=0.0,
        help=(
            "Strength of the soft order3 quota probability bonus. Used only "
            "when --adj_order3_quota_mode=soft."
        ),
    )
    parser.add_argument(
        "--adj_triplet_feature_mode",
        type=str,
        default="pair",
        help=(
            "Triplet scorer feature set. 'pair' uses the three internal pair "
            "scores. 'synergy' uses mean/min/max/std/balance features so the "
            "hyperedge scorer can learn triplet quality beyond pair mean."
        ),
    )
    parser.add_argument(
        "--adj_triplet_balance_coef",
        type=float,
        default=0.0,
        help=(
            "Down-weight unbalanced triplets whose weakest internal pair is "
            "much weaker than the triplet mean. This is a graph-quality prior, "
            "not an environment reward signal."
        ),
    )
    parser.add_argument(
        "--use_adj_advantage_triplet_scorer",
        action="store_true",
        default=False,
        help=(
            "Use graph-conditioned raw factor credit to maintain an EMA "
            "quality score for each pair/triplet candidate, and feed the "
            "triplet-vs-pair marginal credit back into SDDFG triplet scores. "
            "Disabled by default to preserve legacy graph behavior."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_ema_alpha",
        type=float,
        default=0.05,
        help=(
            "EMA rate for advantage-aware pair/triplet candidate quality "
            "updates."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_score_coef",
        type=float,
        default=0.50,
        help=(
            "Strength of the advantage-aware marginal triplet credit "
            "multiplier applied to third-order candidate scores."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_score_scale",
        type=float,
        default=0.05,
        help=(
            "Raw-credit scale used before tanh when converting triplet "
            "marginal credit into a bounded score multiplier."
        ),
    )
    parser.add_argument(
        "--use_adj_triplet_credit_direct_rank",
        action="store_true",
        default=False,
        help=(
            "Use the triplet-vs-pair marginal credit as a direct ranking bias "
            "on third-order candidates instead of the weaker legacy EMA "
            "multiplier. This is only active with "
            "--use_adj_advantage_triplet_scorer."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_rank_coef",
        type=float,
        default=0.0,
        help=(
            "Direct ranking-bias strength for advantage-aware triplet "
            "marginal credit. Values above zero make positive marginal "
            "triplets compete directly against pairs and suppress negative "
            "marginal triplets."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_min_multiplier",
        type=float,
        default=0.25,
        help=(
            "Lower bound for the direct triplet-credit ranking multiplier. "
            "Only used when --use_adj_triplet_credit_direct_rank is enabled."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_max_multiplier",
        type=float,
        default=3.0,
        help=(
            "Upper bound for the direct triplet-credit ranking multiplier. "
            "Only used when --use_adj_triplet_credit_direct_rank is enabled."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_negative_rank_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied only to negative marginal-credit direct-rank logits. "
            "Values below 1.0 keep direct rank from eliminating triplet "
            "coverage based on short-horizon negative credit."
        ),
    )
    parser.add_argument(
        "--adj_triplet_credit_min_positive_fraction",
        type=float,
        default=0.0,
        help=(
            "If positive, negative direct-rank suppression is further damped "
            "when the current triplet candidate catalog has fewer than this "
            "fraction of positive marginal-credit candidates."
        ),
    )
    parser.add_argument(
        "--adj_triplet_negative_graph_penalty",
        type=float,
        default=0.50,
        help=(
            "Penalty applied to locally positive factors selected inside "
            "negative-advantage graphs when updating advantage-aware triplet "
            "quality. This prevents low-return topologies from promoting "
            "their locally best but globally bad triplets."
        ),
    )
    parser.add_argument(
        "--adj_min_pair_ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of selected SDDFG factor slots reserved for "
            "pair factors. This is a complementary guard for Wolfpack: pair "
            "pursuit edges should not be squeezed out by triplet quotas."
        ),
    )
    parser.add_argument(
        "--adj_max_order3_ratio_start",
        type=float,
        default=1.0,
        help=(
            "Initial maximum fraction of selected SDDFG factor slots allowed "
            "to be third-order. One disables the upper band constraint."
        ),
    )
    parser.add_argument(
        "--adj_max_order3_ratio_final",
        type=float,
        default=1.0,
        help=(
            "Final maximum fraction of selected factor slots allowed to be "
            "third-order. Use this with the min ratio to keep graphs in a "
            "healthy pair/triplet band instead of all-triplet collapse."
        ),
    )
    parser.add_argument(
        "--adj_max_order3_ratio_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps used to anneal the third-order upper-band "
            "ratio. Zero keeps the final value."
        ),
    )
    parser.add_argument(
        "--adj_greedy_sample_prob_start",
        type=float,
        default=0.0,
        help=(
            "Initial probability of taking the constrained argmax graph "
            "decision during training rollouts. Zero preserves categorical "
            "sampling."
        ),
    )
    parser.add_argument(
        "--adj_greedy_sample_prob_final",
        type=float,
        default=0.0,
        help=(
            "Final probability of using argmax graph decisions during "
            "training rollouts. Annealing this upward reduces stochastic "
            "rollout versus argmax-eval graph mismatch."
        ),
    )
    parser.add_argument(
        "--adj_greedy_sample_prob_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps used to anneal the argmax graph-decision "
            "mixture probability. Zero keeps the final value."
        ),
    )
    parser.add_argument(
        "--adj_greedy_sample_prob_cap",
        type=float,
        default=1.0,
        help=(
            "Upper bound for the effective greedy graph-decision mixture. "
            "This prevents late training from locking into low-return graph "
            "topologies even when train/eval order gaps are already small."
        ),
    )
    parser.add_argument(
        "--use_adj_order3_credit_gate",
        action="store_true",
        default=False,
        help=(
            "Adaptively relax order3 quota and greedy graph hardening when "
            "recent third-order factor PPO credit is negative. Disabled by "
            "default to preserve legacy SDDFG behavior."
        ),
    )
    parser.add_argument(
        "--adj_order3_credit_gate_loss_scale",
        type=float,
        default=0.005,
        help=(
            "Positive order3_factor_rl_loss value that maps to a full "
            "credit-gate intervention. Larger values make the gate less "
            "sensitive."
        ),
    )
    parser.add_argument(
        "--adj_order3_credit_gate_min_scale",
        type=float,
        default=0.55,
        help=(
            "Smallest multiplier applied to order3 quota and greedy graph "
            "sampling when the credit gate is fully active."
        ),
    )
    parser.add_argument(
        "--adj_order3_credit_gate_ema_alpha",
        type=float,
        default=0.1,
        help=(
            "EMA rate for the positive order3_factor_rl_loss signal used by "
            "the adaptive order3 credit gate."
        ),
    )
    parser.add_argument(
        "--adj_order3_credit_gate_max_delta",
        type=float,
        default=1.0,
        help=(
            "Maximum per-update change of the order3 credit gate. Values "
            "below one make graph hardening react smoothly to noisy relative "
            "credit estimates."
        ),
    )
    parser.add_argument(
        "--use_adj_order3_relative_credit_gate",
        action="store_true",
        default=False,
        help=(
            "Drive the order3 credit gate from the excess triplet loss over "
            "pair loss instead of the absolute triplet loss. This avoids "
            "suppressing useful triplets merely because all factor-local "
            "advantages are temporarily noisy."
        ),
    )
    parser.add_argument(
        "--adj_order3_credit_gate_margin",
        type=float,
        default=0.0,
        help=(
            "Margin subtracted from order3_factor_rl_loss - "
            "order2_factor_rl_loss before the relative credit gate reacts."
        ),
    )
    parser.add_argument(
        "--adj_return_adv_coef",
        type=float,
        default=1.0,
        help=(
            "Weight of trajectory return advantage for SDDFG adjacency "
            "learning."
        ),
    )
    parser.add_argument(
        "--adj_factor_adv_coef",
        type=float,
        default=0.0,
        help=(
            "Weight of centered factor Q-V advantage used as auxiliary "
            "credit for SDDFG adjacency learning."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_coef",
        type=float,
        default=0.0,
        help=(
            "Order-aware multiplier for SDDFG factor-local adjacency credit. "
            "Triplet factors receive 1 + coef relative weight while pair "
            "factors keep weight 1.0. Zero preserves the legacy loss."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_positive_only",
        action="store_true",
        default=False,
        help=(
            "Apply the full order-aware multiplier only to positive "
            "factor-local advantages. Negative triplet residuals use "
            "--adj_order_adv_negative_coef instead. This lets SDDFG promote "
            "reward-positive triplets without making the credit gate react "
            "mostly to artificially amplified negative triplet noise."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_negative_coef",
        type=float,
        default=0.0,
        help=(
            "Order-aware multiplier coefficient for negative factor-local "
            "advantages when --adj_order_adv_positive_only is enabled. Zero "
            "keeps negative triplet suppression at the pair scale."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_require_positive_graph_adv",
        action="store_true",
        default=False,
        help=(
            "When positive-only order credit is enabled, promote a triplet "
            "only if its local residual is positive and the whole sampled "
            "graph has positive advantage. This prevents locally good "
            "triplets inside low-return graphs from being reinforced."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_graph_gate_mode",
        type=str,
        default="binary",
        help=(
            "Graph-advantage gate used with "
            "--adj_order_adv_require_positive_graph_adv. 'binary' uses a "
            "hard graph_advantage > 0 filter. 'soft' uses a continuous "
            "positive-graph-advantage weight, which reduces noisy promotion "
            "jumps without reinforcing triplets from negative-return graphs."
        ),
    )
    parser.add_argument(
        "--adj_order_adv_graph_gate_scale",
        type=float,
        default=1.0,
        help=(
            "Scale multiplier for the soft graph-advantage promotion gate. "
            "The denominator is the batch mean absolute graph advantage times "
            "this value."
        ),
    )
    parser.add_argument(
        "--use_adj_triplet_graph_return_credit",
        action="store_true",
        default=False,
        help=(
            "Route a bounded portion of positive graph-level return "
            "advantage back to selected triplet factors before order-aware "
            "weighting and triplet credit EMA updates. This keeps delayed "
            "capture/win payoff from being erased when the local triplet "
            "residual is negative."
        ),
    )
    parser.add_argument(
        "--adj_triplet_graph_return_credit_coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient applied to positive graph advantage when "
            "--use_adj_triplet_graph_return_credit is enabled."
        ),
    )
    parser.add_argument(
        "--adj_triplet_graph_return_credit_cap",
        type=float,
        default=0.0,
        help=(
            "Maximum supplemental triplet credit as a multiple of the batch "
            "mean absolute graph advantage. Zero disables the cap."
        ),
    )
    parser.add_argument(
        "--adj_triplet_graph_return_credit_min_graph_adv",
        type=float,
        default=0.0,
        help=(
            "Minimum graph advantage that must be exceeded before routing "
            "graph-return credit to selected triplet factors."
        ),
    )
    parser.add_argument(
        "--adj_triplet_graph_return_credit_raw_gate_scale",
        type=float,
        default=0.0,
        help=(
            "If positive, down-weight graph-return triplet credit for factors "
            "whose raw local residual is negative. The gate reaches zero at "
            "-scale times the batch mean absolute graph advantage and one at "
            "zero residual."
        ),
    )
    parser.add_argument(
        "--adj_triplet_graph_return_credit_require_delayed_gate",
        action="store_true",
        default=False,
        help=(
            "Require delayed success-window evidence before graph-return "
            "credit can promote a selected triplet. This prevents positive "
            "graph returns from broadly rewarding triplets that did not "
            "participate in the later success window."
        ),
    )
    parser.add_argument(
        "--use_adj_delayed_triplet_credit",
        action="store_true",
        default=False,
        help=(
            "Add a triplet-only adjacency credit component from positive "
            "training rewards observed in a short future window. This spreads "
            "capture/success payoff to preceding triplet topology decisions "
            "without using evaluation-only information."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for the delayed positive-reward triplet credit "
            "component."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_window",
        type=int,
        default=0,
        help=(
            "Number of future environment steps used to spread positive "
            "reward back to selected triplet factors."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_cap",
        type=float,
        default=0.0,
        help=(
            "Absolute cap applied after scaling delayed triplet credit. Zero "
            "leaves it uncapped."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_min_reward",
        type=float,
        default=0.0,
        help=(
            "Per-step team reward threshold subtracted before positive "
            "future rewards are spread to triplet factors."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_positive_only",
        action="store_true",
        default=False,
        help=(
            "Keep only delayed triplet credit from future windows whose "
            "baseline-normalized delayed reward is positive. This prevents "
            "near-ubiquitous negative/zero-mean delayed noise from dominating "
            "the triplet scorer."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_min_adv",
        type=float,
        default=0.0,
        help=(
            "Minimum standardized delayed reward advantage required before "
            "a future reward window contributes triplet credit."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_credit_require_future_match",
        action="store_true",
        default=False,
        help=(
            "Credit a selected triplet from delayed positive rewards only "
            "when the same triplet node set appears again inside the future "
            "credit window. This makes delayed triplet credit graph-conditioned "
            "instead of broadcasting future reward to every selected triplet."
        ),
    )
    parser.add_argument(
        "--use_adj_delayed_triplet_success_gate",
        action="store_true",
        default=False,
        help=(
            "Gate delayed triplet credit by a graph-level future reward "
            "window advantage. This keeps triplet delayed credit focused on "
            "success-like high-return windows while using only training "
            "rollout rewards."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_success_gate_min_adv",
        type=float,
        default=0.0,
        help=(
            "Minimum standardized graph-level future-window advantage before "
            "delayed triplet credit passes the success gate."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_success_gate_scale",
        type=float,
        default=1.0,
        help=(
            "Soft ramp width for the delayed triplet success gate. The gate "
            "is clipped to [0, 1] after subtracting the minimum advantage."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_success_gate_floor",
        type=float,
        default=0.0,
        help=(
            "Minimum nonzero success-gate weight for valid graph transitions "
            "when success gating is enabled. This makes success-window "
            "credit a soft weighting instead of a hard filter."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_future_overlap_min_nodes",
        type=int,
        default=3,
        help=(
            "Minimum node overlap between a current triplet and a future "
            "triplet before future-window reward can be spread back. Use 3 "
            "for exact future match, or 2 for connected pursuit-group credit."
        ),
    )
    parser.add_argument(
        "--adj_delayed_triplet_partial_match_weight",
        type=float,
        default=0.5,
        help=(
            "Credit multiplier used when delayed triplet future matching "
            "passes by partial overlap rather than exact triplet identity."
        ),
    )
    parser.add_argument(
        "--use_adj_capture_to_win_credit",
        action="store_true",
        default=False,
        help=(
            "Add a conservative triplet credit term only when the whole "
            "training episode has above-baseline return. This ties delayed "
            "triplet promotion to capture-to-win quality rather than any "
            "future positive reward in isolation."
        ),
    )
    parser.add_argument(
        "--adj_capture_to_win_credit_coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for capture-to-win triplet credit. Zero disables "
            "the term even if --use_adj_capture_to_win_credit is present."
        ),
    )
    parser.add_argument(
        "--adj_capture_to_win_credit_min_outcome_adv",
        type=float,
        default=0.5,
        help=(
            "Minimum standardized episode return advantage before a triplet "
            "can receive capture-to-win credit."
        ),
    )
    parser.add_argument(
        "--adj_capture_to_win_credit_scale",
        type=float,
        default=0.75,
        help=(
            "Soft ramp width for the episode-outcome quality gate used by "
            "capture-to-win triplet credit."
        ),
    )
    parser.add_argument(
        "--adj_capture_to_win_credit_cap",
        type=float,
        default=0.35,
        help=(
            "Maximum capture-to-win credit as a multiple of batch mean "
            "absolute graph advantage. Zero leaves the term uncapped."
        ),
    )
    parser.add_argument(
        "--adj_capture_to_win_credit_require_future_match",
        action="store_true",
        default=False,
        help=(
            "Require the current triplet to reappear or overlap in the "
            "future success window before capture-to-win credit is applied."
        ),
    )
    parser.add_argument(
        "--use_adj_pair_triplet_complementary_credit",
        action="store_true",
        default=False,
        help=(
            "Add capture-anchored credit only when a pair-only backbone becomes "
            "part of a strictly later triplet on a real capture_count event."
        ),
    )
    parser.add_argument(
        "--adj_pair_pursuit_credit_coef",
        type=float,
        default=0.0,
        help="Coefficient for strict-future pair-to-capture-triplet credit.",
    )
    parser.add_argument(
        "--adj_pair_pursuit_credit_window",
        type=int,
        default=20,
        help="Strictly-future window for pair-to-capture-triplet matching.",
    )
    parser.add_argument(
        "--adj_pair_pursuit_credit_cap",
        type=float,
        default=0.20,
        help="Maximum pair pursuit credit as a multiple of graph-advantage scale.",
    )
    parser.add_argument(
        "--adj_pair_pursuit_credit_min_reward",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. Capture-anchored pair credit "
            "uses the environment capture_count event and never team reward."
        ),
    )
    parser.add_argument(
        "--adj_exploration_mix",
        type=float,
        default=0.0,
        help=(
            "Fixed uniform-mixture probability for constrained SDDFG graph "
            "sampling. Zero keeps behavior and PPO target distributions "
            "identical while categorical sampling still explores."
        ),
    )
    parser.add_argument(
        "--adj_train_epochs",
        type=int,
        default=10,
        help=(
            "Number of full adjacency-buffer PPO epochs per graph update. "
            "Use a smaller value when train_adj_episode is small and replay "
            "windows overlap heavily."
        ),
    )
    parser.add_argument(
        "--adj_ppo_clip_stop_ratio",
        type=float,
        default=0.35,
        help=(
            "If positive, stop the current adjacency PPO update early once "
            "the mean clamp_ratio exceeds this value after at least one full "
            "epoch. This prevents graph-policy drift when behavior and target "
            "topologies diverge. The default is enabled because SDDFG's "
            "overlapping adjacency replay windows otherwise silently produced "
            "run35/run36 clamp ratios above 0.7."
        ),
    )
    parser.add_argument(
        "--adj_ppo_factor_clip_stop_ratio",
        type=float,
        default=0.35,
        help=(
            "If positive, stop the current adjacency PPO update early once "
            "the mean factor_clamp_ratio exceeds this value. This catches "
            "factor-local credit drift that can remain high even when the "
            "graph-level ratio looks acceptable. The default mirrors "
            "adj_ppo_clip_stop_ratio so direct train_wolfpack.py invocations "
            "do not disable the guard by omission."
        ),
    )
    parser.add_argument(
        "--adj_ppo_min_epochs",
        type=int,
        default=1,
        help=(
            "Minimum adjacency PPO epochs before high-clamp early stopping "
            "can trigger."
        ),
    )
    parser.add_argument(
        "--use_adj_dynamic_recent_window",
        action="store_true",
        default=False,
        help=(
            "Adapt the adjacency PPO recent replay window downward when the "
            "previous update still has high graph/factor stale ratios. This "
            "keeps graph PPO closer to on-policy than a fixed recent window."
        ),
    )
    parser.add_argument(
        "--adj_recent_episode_window_min",
        type=int,
        default=1,
        help=(
            "Minimum recent episode window used by dynamic adjacency replay. "
            "Ignored unless --use_adj_dynamic_recent_window is enabled."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_stale_threshold",
        type=float,
        default=0.35,
        help=(
            "Graph stale-ratio threshold that shrinks the adaptive recent "
            "episode window on the next adjacency PPO update."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_factor_stale_threshold",
        type=float,
        default=0.30,
        help=(
            "Factor stale-ratio threshold that shrinks the adaptive recent "
            "episode window on the next adjacency PPO update."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_shrink_patience",
        type=int,
        default=1,
        help=(
            "Number of consecutive high-stale adjacency PPO updates required "
            "before shrinking the adaptive recent episode window."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_recover_patience",
        type=int,
        default=2,
        help=(
            "Number of consecutive low-stale adjacency PPO updates required "
            "before recovering the adaptive recent episode window toward the "
            "configured value."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_recover_stale_threshold",
        type=float,
        default=-1.0,
        help=(
            "Graph stale-ratio threshold used to recover the adaptive recent "
            "episode window. Negative values use 80 percent of the shrink "
            "threshold."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_recover_factor_stale_threshold",
        type=float,
        default=-1.0,
        help=(
            "Factor stale-ratio threshold used to recover the adaptive recent "
            "episode window. Negative values use 80 percent of the factor "
            "shrink threshold."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_severe_margin",
        type=float,
        default=0.15,
        help=(
            "Extra stale-ratio margin above the shrink threshold that sends "
            "the adaptive recent episode window directly to its minimum."
        ),
    )
    parser.add_argument(
        "--adj_recent_episode_window_emergency",
        type=int,
        default=1,
        help=(
            "Near-on-policy recent episode window used immediately when "
            "graph or factor stale ratios exceed the emergency thresholds. "
            "Ignored unless --use_adj_dynamic_recent_window is enabled."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_emergency_stale_threshold",
        type=float,
        default=0.45,
        help=(
            "Graph stale-ratio threshold that immediately switches adjacency "
            "PPO sampling to adj_recent_episode_window_emergency."
        ),
    )
    parser.add_argument(
        "--adj_recent_window_emergency_factor_stale_threshold",
        type=float,
        default=0.35,
        help=(
            "Factor stale-ratio threshold that immediately switches adjacency "
            "PPO sampling to adj_recent_episode_window_emergency."
        ),
    )
    parser.add_argument(
        "--use_adj_ppo_stale_trust",
        action="store_true",
        help=(
            "Down-weight adjacency PPO graph/factor samples whose behavior "
            "and target probabilities are already far outside the trust region "
            "on the first epoch. This addresses stale topology replay that "
            "high-clamp early stopping alone cannot fix."
        ),
    )
    parser.add_argument(
        "--adj_ppo_stale_trust_clip",
        type=float,
        default=0.2,
        help=(
            "Absolute importance-ratio deviation from 1.0 that starts stale "
            "sample down-weighting when use_adj_ppo_stale_trust is enabled."
        ),
    )
    parser.add_argument(
        "--adj_ppo_stale_trust_scale",
        type=float,
        default=0.25,
        help=(
            "Exponential decay scale for stale adjacency PPO sample weights "
            "after adj_ppo_stale_trust_clip is exceeded."
        ),
    )
    parser.add_argument(
        "--adj_ppo_stale_trust_min_weight",
        type=float,
        default=0.25,
        help=(
            "Minimum graph/factor trust weight for stale adjacency PPO samples. "
            "Keeps learning from rare positive outcomes while preventing "
            "out-of-distribution topology replay from dominating the loss."
        ),
    )
    parser.add_argument(
        "--require_connected_adj",
        action="store_true",
        default=False,
        help=(
            "Require every sampled SDDFG factor graph to be connected over "
            "the currently active agents. Coverage alone permits isolated "
            "coordination components."
        ),
    )
    parser.add_argument("--min_adj_begin_step", type=int, default=5000,
                        help="Recommended lower bound for dynamic adj warmup steps")

    # ddfg parameters
    #     policy network parameters
    parser.add_argument('--use_double_q', action='store_false',
                        default=True, help="Whether to use double q learning")
    parser.add_argument("--use_dyn_graph", action='store_true',
                        default=False, help="Whether to use Generative graph network")
    parser.add_argument("--num_rank", type=int, default=3, help="tensor are decomposed with this rank")
    parser.add_argument("--equal_vdn", action='store_true',
                        default=False, help="Whether to make the algorithm equal to vdn")
    parser.add_argument("--msg_anytime", action='store_false',
                        default=True, help="Anytime extension of greedy action selection (Kok and Vlassis, 2006)")
    parser.add_argument("--msg_normalized", action='store_false',
                        default=True,
                        help="Message normalization during greedy action selection (Kok and Vlassis, 2006)")
    parser.add_argument("--lamda", type=float, default=0,
                        help="Damping factor for messaging")
    parser.add_argument("--msg_iterations", type=int, default=8,
                        help="Number of cycles of factor graph message passing algorithm")
    #     adj network parameters
    parser.add_argument('--adj_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layers for adj networks")
    parser.add_argument('--adj_output_dim', type=int, default=2,
                        help="Dimension of output layers for adj networks")
    parser.add_argument('--adj_alpha', type=float, default=0.1,
                        help="alpha")
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help="entropy term coefficient (default: 0.2)")
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument(
        "--policy_lr_anneal_steps",
        type=int,
        default=0,
        help=(
            "Explicit environment-step horizon for policy/critic LR decay. "
            "Zero preserves the legacy behavior of using num_env_steps."
        ),
    )
    parser.add_argument("--entropy_coef", type=float, default=0.001,
                        help='entropy term coefficient (default: 0.001)')
    parser.add_argument(
        "--adj_entropy_coef",
        type=float,
        default=-1.0,
        help=(
            "Initial SDDFG graph entropy coefficient. A negative value "
            "falls back to --entropy_coef for backward compatibility."
        ),
    )
    parser.add_argument(
        "--adj_entropy_coef_final",
        type=float,
        default=-1.0,
        help=(
            "Final SDDFG graph entropy coefficient. A negative value keeps "
            "the legacy constant --entropy_coef behavior."
        ),
    )
    parser.add_argument(
        "--adj_entropy_anneal_steps",
        type=int,
        default=0,
        help=(
            "Environment steps over which the SDDFG graph entropy "
            "coefficient is annealed. Zero disables annealing."
        ),
    )
    parser.add_argument("--use_valuenorm", action='store_true', default=False,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_vfunction", action='store_true', default=False)
    parser.add_argument("--use_epsilon_greedy", action='store_true', default=False)
    parser.add_argument("--pretrain_adj", action='store_true', default=False)
    parser.add_argument("--adj_max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument('--num_mini_batch', type=int, default=4)
    parser.add_argument('--adj_begin_step', type=int, default=0)
    parser.add_argument("--use_adj_init", action='store_false', default=True)

    # -----------------------------
    # qmix parameters
    # -----------------------------
    # hypernet 的层数（1 或 2），默认 2
    parser.add_argument('--hypernet_layers', type=int, default=2,
                        help="Number of layers for hypernetworks. Must be either 1 or 2")
    # mixing network 隐藏层维度，默认 32
    parser.add_argument('--mixer_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layer of mixing network")
    # hypernet 隐藏层维度（当 hypernet_layers == 2 时），默认 64
    parser.add_argument('--hypernet_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")
    parser.add_argument(
        '--qmix_normalize_mixer_state',
        action='store_true',
        default=False,
        help=(
            "Apply non-parametric LayerNorm to the centralized state before "
            "QMIX hypernetworks; intended for raw-coordinate environments."
        ),
    )

    # -----------------------------
    # qplex parameters
    # -----------------------------
    # qplex 的 heads 数量，默认 4
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads in QPLEX.")
    # num_kernel 默认 10
    parser.add_argument("--num_kernel", type=int, default=10, help="Number of kernels in QPLEX attention mixer.")
    # adv_hypernet_embed 默认 64
    parser.add_argument("--adv_hypernet_embed", type=int, default=64,
                        help="Embedding dimension of QPLEX advantage hypernetwork.")
    # weighted_head 默认启用（True）
    parser.add_argument("--weighted_head", action="store_true", default=False,
                        help="Whether to use weighted head in QPLEX.")
    # is_minus_one 默认 True
    parser.add_argument("--is_minus_one", action="store_true", default=False,
                        help="Whether to use minus-one trick in QPLEX original implementation.")
    # adv_hypernet_layers 默认 3
    parser.add_argument("--adv_hypernet_layers", type=int, default=3,
                        help="Number of layers of QPLEX advantage hypernetwork.")
    # qplex_qatten 的正则系数，默认 0.001
    parser.add_argument("--attend_reg_coef", type=float, default=0.001,
                        help="Attention regularization coefficient for QPLEX.")
    # state_bias 默认启用（True）
    parser.add_argument("--state_bias", action="store_true", default=False,
                        help="Whether to use state bias in QPLEX mixer.")
    # mask_dead 默认不启用（False）
    parser.add_argument("--mask_dead", action="store_true", default=False,
                        help="Whether to mask dead agents in QPLEX original implementation.")
    # nonlinear 默认 False（不启用非线性）
    parser.add_argument("--nonlinear", action="store_true", default=False,
                        help="Whether to use nonlinear transformation in QPLEX.")

    # ==================== 修改点：学习率衰减解耦参数 ====================
    parser.add_argument("--use_adj_linear_lr_decay", action="store_true", default=False,
                        help="Whether to apply linear lr decay to adj/GAT optimizer. "
                             "If False, adj lr will keep a floor value or stay constant.")
    parser.add_argument("--adj_lr_decay_floor", type=float, default=2e-5,
                        help="Minimum lr for adj/GAT optimizer when linear lr decay is used.")
    parser.add_argument(
        "--adj_lr_anneal_steps",
        type=int,
        default=500000,
        help=(
            "Environment steps used to anneal only the adj/GAT learning "
            "rate. Set the same value across budgets for prefix comparisons, "
            "or set it to the run budget for full-run annealing."
        ),
    )
    parser.add_argument("--policy_lr_decay_floor", type=float, default=1e-5,
                        help="Minimum lr for policy optimizer when linear lr decay is used.")
    parser.add_argument("--critic_lr_decay_floor", type=float, default=1e-5,
                        help="Minimum lr for critic optimizers when linear lr decay is used.")

    # exploration parameters
    parser.add_argument('--num_random_episodes', type=int, default=5,
                        help="Number of episodes to add to buffer with purely random actions")
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_anneal_time', type=int, default=1000000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument(
        '--use_joint_epsilon_exploration',
        action='store_true',
        default=False,
        help=(
            "Apply epsilon once per coordinated joint decision instead of "
            "independently replacing each agent after the joint solver."
        ),
    )
    parser.add_argument(
        '--post_capture_joint_greedy_floor',
        type=float,
        default=0.0,
        help=(
            "Minimum full-greedy joint-branch probability while at least one "
            "prey is frozen and another prey remains active. Zero disables "
            "the post-capture override."
        ),
    )
    parser.add_argument(
        '--post_capture_explore_max_random_agents',
        type=int,
        default=0,
        help=(
            "Maximum alive-agent random replacements on a post-capture joint "
            "explore branch. Zero preserves legacy all-alive replacement."
        ),
    )
    parser.add_argument(
        '--pre_capture_visible_prey_quorum_guard',
        action='store_true',
        default=False,
        help=(
            "On a pre-capture joint-explore branch, keep the greedy actions "
            "of agents whose current local observations form an exact "
            "two-wolf prey quorum. Other alive slots retain their sampled "
            "random actions."
        ),
    )
    parser.add_argument(
        '--pre_capture_visible_prey_quorum_greedy_frontier_guard',
        action='store_true',
        default=False,
        help=(
            "Before the first capture, rerank only locally observed "
            "exact-quorum frontier slots over actions that maximize robust "
            "one-step prey visibility. The rerank uses existing max-sum "
            "utilities and consumes no RNG."
        ),
    )
    parser.add_argument('--adj_anneal_time', type=int, default=500000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--disount_step', type=int, default=500000,
                        help="Discount factor of computational adj networks")
    parser.add_argument('--act_noise_std', type=float,
                        default=0.1, help="Action noise")

    # train parameters
    parser.add_argument('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    parser.add_argument('--train_interval_episode', type=int, default=1,
                        help="Number of env steps between updates to actor/critic")
    parser.add_argument('--train_adj_episode', type=int, default=4,
                        help="Number of env steps between train adj network")
    parser.add_argument('--drop_temperature_episode', type=int, default=10,
                        help="Number of env steps between drop_temperature")
    parser.add_argument('--train_interval', type=int, default=100,
                        help="Number of episodes between updates to actor/critic")
    parser.add_argument("--use_value_active_masks",
                        action='store_true', default=False)

    # eval parameters
    parser.add_argument('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help="After how many episodes the policy should be evaled")
    parser.add_argument('--num_eval_episodes', type=int, default=1,
                        help="How many episodes to collect for each eval")

    # save parameters
    parser.add_argument('--save_interval', type=int, default=100000,
                        help="After how many episodes of training the policy model should be saved")
    parser.add_argument('--use_save', action='store_false',
                        default=True, help="Whether to save the model")

    # log parameters
    parser.add_argument('--log_interval', type=int, default=1000,
                        help="After how many episodes of training the policy model should be saved")

    # pretained parameters
    parser.add_argument("--model_dir", type=str, default=None)

    # aloha scenario
    parser.add_argument("--max_list_length", type=int, default=5)

    # hallway scenario
    parser.add_argument("--n_groups", type=int, default=5)
    parser.add_argument("--reward_win", type=int, default=1)

    # sensor scenario
    parser.add_argument("--array_height", type=int, default=3)
    parser.add_argument("--array_width", type=int, default=5)
    parser.add_argument("--n_preys", type=int, default=3)
    parser.add_argument("--catch_reward", type=int, default=3)
    parser.add_argument("--scan_cost", type=int, default=1)

    # gather scenario
    parser.add_argument("--map_height", type=int, default=3)
    parser.add_argument("--map_width", type=int, default=5)
    parser.add_argument("--catch_fail_reward", type=int, default=-5)
    parser.add_argument("--target_reward", type=float, default=0.000)
    parser.add_argument("--other_reward", type=int, default=5)

    # disperse scenario
    parser.add_argument("--n_hospitals", type=int, default=4)

    return parser
