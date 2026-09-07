# 项目训练与代码改进历史

## 0. 文档用途与维护规则

- 本文档是 SDDFG / Wolfpack 项目的唯一长期历史背景文档。新任务框开始工作前必须完整阅读本文档。
- 当前仓库中的权威文件实际位于根目录`PROJECT_TRAINING_HISTORY.md`；`docs/PROJECT_TRAINING_HISTORY.md`不存在。为避免形成两份冲突历史，只维护本文件。
- 本文档只在用户明确要求“阶段总结、更新项目历史文档、整理历史run或生成交接总结”时更新；普通日志分析、代码修改、测试和训练命令生成不自动写入长期文档。
- 新结论必须有训练控制台日志、CSV、TensorBoard、保存配置、源码检查或明确的用户记录作为依据。无法核实的内容明确标为“未确认”。
- 本文档只记录已经发生的事实、已验证经验和信息缺口，不包含未验证的未来方案、参数建议或实验计划。
- 对每个机制严格区分四个层次：代码中存在、训练时实际启用、训练时实际触发、对性能产生正面作用。前三级不能推出第四级。
- 本次初始整理来源：本次历史对话整理、历史控制台指标摘录、当前工作区源码与训练脚本检查（2026-07-16）。

## 1. 项目目标与任务背景

- 项目名称：SDDFG / Wolfpack 多智能体强化学习项目。
- 核心问题：在 **单个 episode 内** 处理智能体实时退出、加入和恢复的数量变化；该问题不能替换为不同 episode 使用不同固定人数的“回合间可扩展”。
- 环境：`wolfpack`，环境标识为 `wolfpack-v0`；环境实现位于 `envs/Wolfpack/wolfpack_penalty_open.py`。
- 算法框架：类 PyMARL 训练流水线。比较对象包含 SDDFG、DDFG、VDN、QMIX、QPLEX；SDDFG 采用图结构、动态消息传递、邻接因子选择和掩码适配。
- 动态实验的已确认公共环境设置：`episode_length=200`、`num_agents=4`、`max_player_num=6`、`max_food_num=2`、`shock_steps=50,120`、`shock_remove_num=2`、`shock_join_delay=10`、`shock_join_num=1`、`shock_recover_delay=30`、`dynamic_min_agents=2`、`intra_episode_dynamic=enabled`、`continue_after_success=enabled`、`add_rate=0.0`、`del_rate=0.0`。
- 理论活跃人数轨迹：0--49 步为 4；50--59 步为 2；60--79 步为 3；80--119 步为 5；120--129 步为 3；130--149 步为 4；150--200 步为 6。
- 环境与算法基本关系：环境输出固定最大槽位张量，并以 active / availability / mask 表示当前可参与智能体；图网络与价值分解网络应在固定张量维度内忽略失效槽位，避免因 episode 内人数变化造成维度改变。
- 奖励与终止条件的精确源码语义未在本次整理中逐行复核；历史日志使用 `average_episode_rewards`、`capture_events`、`win_rate`、`first_success_step` 等结果指标。
- 当前主要评价指标：`eval_reward`、`last5 reward`、`best reward`、`eval_win_rate`、`capture_events`、`first_success_step`，以及图阶数、图合法性、train/eval 图分布差、PPO clamp、factor credit 等诊断指标。
- 历史实验使用过20k、200k和2M等长度，但它们不是固定训练阶段。每次训练步数应由目标机制首次触发时间、触发频率、生命周期、历史退化区间、性能判断需求和计算成本共同决定。
- 当前对照基线：VDN、QMIX、QPLEX，均使用同一 episode 内动态环境配置；其中 QMIX 曾修复 PyTorch 兼容性问题。
- 已知对照结果：VDN 200k final reward 约 -3.269、last5 约 -2.970、win rate 约 0.4；QPLEX 200k final reward 约 -1.487、last5 约 -3.231、win rate 约 0.1；QMIX 200k final reward 1.524、last5 -0.913、best 1.608、win rate 0.1；QMIX 2M final reward 6.947、last5 约 6.175、**实际 last10 约 5.559**、win rate 0.8。历史上曾把 6.175 同时写作 QMIX 的 last5/last10；现已按CSV口径说明修正。6.175仍可作为历史严格比较值，但不能称为QMIX实际last10。
- 仓库中还存在动态配置和事件记录完整的 VDN 2M、QPLEX 2M 结果。已知 QPLEX 2M 汇总为 final 8.932、last5 6.661、last10 6.400、win 0.9；历史交接曾将其判为“不可靠”，但当前文档没有对应排除原因、代码版本或错误证据。因此该结果必须保留为“存在但有效性有争议”，不得用于已确认的算法优势声明，也不得在缺证据时删除。
- 已知 SDDFG 强结果：旧 SDDFG `run18` 的 200k 前缀 final reward 为 2.713；`run49` 的完整 2M final reward 为 6.296、last5 为 5.751、last10 为 5.230、best 为 7.815@1.94M、final win rate 为 0.9、capture events 为 4.5；`run68` 的完整 2M final reward 为 6.718、last5 为 6.610、last10 为 6.527、best 为 7.724@1.86M、final win rate 为 0.6、capture events 为 4.6。run68 是当前已核验的 SDDFG 长程最高终值和最高滑动回报记录，但 final 仍低于 QMIX 2M 的 6.947，win 也低于 0.8。不同 run 的源码版本和修改集合不同，不能将差异归因为单一机制。
- 200k 历史状态：`run70` 达到 final 3.555、last5 1.592、last10 0.628、win 0.8、capture 3.4，是当前已核验的后期机制系列最强200k结果；但随后同代码演进链的run71--run75未保持该水平。最新已审计训练版本run75为final 1.788、last5 1.003、win 0.5、capture 2.8，高于run74但低于run70，并在120k最佳点后发生明显退化。
- 训练长度在历史上均按机制首次触发、覆盖率和退化区间决定；200k、400k、600k与2M都是已经使用过的预算，而非固定前置阶段。run75已验证lifecycle v4真实进入训练并消除了run74的实际位移违反与rollback，但其性能和后期稳定性不足。
- 截至第8次整理，最新完整机制实验为fresh seed=1、80k的`run162`。当前production组合为：35维local observation/59维central state中的prey freeze countdown、228k joint epsilon、post-capture greedy floor=0.25、terminal-gated 24-step Q target、terminal replay lane、forced auxiliary weight=0.10，以及post-capture explore最多随机1个alive slot。run162完成80k且无Traceback/NaN/Inf/transaction rollback；相对同seed同长度run161，formal capture为204对169、strict later-distinct为16对3、training win为23对3，80k eval reward/win/capture为1.588/0.3/2.7对0.584/0.1/1.6。该结果首次给出bounded post-capture exploration的强正向行为证据，但仍是单seed 80k，尚未建立相对长程基线的最终论文优势。run108和run126等旧机制节点继续作为各自历史阶段事实保留。来源：run162 summary/config/CSV/manifest/checkpoint、run160/run161直接对照、production trajectory与源码检查；来源：本次历史对话整理。

## 2. 项目目录与关键文件

| 路径 | 已确认作用 | 历史重要修改或状态 |
|---|---|---|
| `envs/Wolfpack/wolfpack_penalty_open.py` | Wolfpack 环境、episode 内 roster/lifecycle、capture 判定与观测逻辑 | 加入 `intra_episode_dynamic`、shock、join、recover、动态 active 状态与掩码适配；后续 capture identity 版本又输出真实 capture event、participant slots、prey/target、event ID 等字段，供 runner 做精确因子归因。 |
| `scripts/train/train_wolfpack.py` | 训练入口、环境参数解析和环境实例化 | 已解析动态环境参数和多项 SDDFG 图训练参数；应避免向 VDN/QMIX/QPLEX 传递仅 SDDFG 支持的参数。 |
| `runner/base_runner.py` | 通用训练循环、Q 类训练和 adj-PPO 更新 | 除 early-stop、recent/adaptive/emergency episode window 外，后续接入 outcome support round、同 update 多 PPO epoch cohort 复用、candidate/lifecycle round 与训练诊断；run86后负责消费selected trusted control，run95后记录pair transaction人口。run100后写入per-epoch optimizer transaction CSV；run102后记录per-objective gradient；run108后接入pair-only pending外层原子事务、完整回滚和cohort日志。run108审计后已将pair-only事务与通用graph/base PPO early-stop隔离，要求完成全部配置epoch后才logical commit；普通adjacency early-stop不变。 |
| `runner/wolfpack_runner.py` | Wolfpack 专属采集、评估、事件到图身份映射、TensorBoard/console 指标汇总 | 已扩展动态图、credit、outcome support、capture identity、candidate-only/active match、local delta、optimizer guard、lifecycle、保持率和 topology 指标；负责把 `graph_t/action_t/capture_event_t` 对齐后写入 `AdjBuffer`。 |
| `algorithms/sddfg/algorithm/adj_generator.py` | 图/邻接因子生成、阶数控制、采样与候选打分 | 已包含阶数 bonus/温度 schedule、min/max band、soft quota、pair reserve、triplet balance/synergy 特征、greedy mixture/cap、credit gate、triplet credit 打分等代码。 |
| `algorithms/sddfg/algorithm/rSDDFGPolicy.py` | SDDFG recurrent factor-Q前向、greedy joint action与epsilon exploration执行 | run149后实际接收35维local observation；run152起接收post-capture greedy floor；run162起保持一次joint Bernoulli不变，仅在eligible explore branch调用bounded replacement helper并记录greedy/final/slot诊断。 |
| `algorithms/sddfg/r_sddfg.py` | SDDFG learner、graph/factor PPO loss、candidate identity loss、梯度/optimizer/lifecycle 保护与诊断 | 历史上依次接入 outcome local delta、factor loss normalization v2、candidate loss v1--v4、target-transition normalization v2、欧氏梯度冲突投影、Adam actual-update guard/state sync、candidate lifecycle。run102后修复graph PPO误用混合`f_advts`的问题，graph source v2只消费独立replay graph-return advantage，factor-local残差为`f_advt-αG`；run100--run102加入per-epoch/per-objective gradient重构；run108 production pending路径增加严格pair-only objective mode和exact pair-target control诊断。 |
| `algorithms/ddfg/r_ddfg.py` | DDFG learner | 为扩展的 adj batch 字段增加兼容解包，避免 SDDFG 新字段导致 DDFG 训练路径错位。性能影响未单独验证。 |
| `utils/adj_buffer.py` | 图策略缓冲区、episode/factor advantage、outcome cohort 与 replay support | 历史上修复 active mask、padding、roster/timestep baseline；后续加入真实 episode ID、slot generation、support round、final optimizer cohort centering、stored graph advantage ready 标志、identity/candidate metadata、event/episode 质量守恒字段。support v6对class-complete pair采用一个完整selected population和一个Adam transaction；run104--run107后又负责pair-evidence funnel、episode reject reason、candidate provenance和production immutable pending snapshot/恢复。 |
| `utils/adj_training_control.py` | adjacency stale-trust control ratio选择与人口总量聚合 | run89后被纳入manifest，负责以raw/trusted graph/factor总numerator、denominator选择实际control ratio；不得用mini-batch ratio均值代替总量比。 |
| `utils/pair_credit.py` | strict-future pair-to-capture 与 capture identity/outcome 权重计算 | 普通正 shaping reward 不再激活 pair credit；offset=0 被排除；两人 capture 精确匹配 order2 pair，三人匹配 exact order3，更多参与者仅用真实 order3 子结构；实现 event 内和 episode 内质量守恒及 canonical identity 校验。run97起的support v6在class-complete时返回单一full-population partition，使pair与graph/base factor使用同一完整训练人口；one-sided/no-evidence仍走普通balanced partition且pair loss为0。 |
| `utils/pair_pending.py` | pair-specific bounded pending状态、immutable payload、single-use与checkpoint序列化 | run107时序重放后新增。保存严格pair evidence的detached production payload，状态区分available/pending/prepared/committed/expired，TTL以adjacency update计数；与generic outcome support used-state分离。run108以TTL=4真实启用但事务回滚；后续run已有成功commit，run117以后True/4、single-use与原子事务链持续进入真实训练。 |
| `utils/rec_buffer.py` | recurrent replay episode存储、uniform sample及terminal-credit auxiliary episode附加 | run159起保留原uniform RNG/sample不变；当本次train interval首个uniform update未命中win且buffer已有成功episode时，以round-robin附加一个terminal episode并携带lane mask。 |
| `utils/terminal_replay.py` | terminal replay lane的确定性成功episode选择 | 实现成功episode索引与round-robin选择；不改变uniform replay population。run159真实提高completion-credit利用量，但weight=1.0导致objective population失衡；run160后仍保留采样频率，改由0.10 loss weight控制aux population。 |
| `utils/joint_exploration.py` | joint epsilon branch、合法随机动作与bounded post-capture slot replacement | `epsilon_random_mask`仍只抽一次joint branch；`bound_joint_random_replacements`在eligible explore时从alive slots均匀无放回选择至多1个slot，0保持legacy全alive replacement。run162生产验证explore事件率不变、replacement最大1、动作合法且多样性未坍缩。 |
| `config.py` | 项目通用配置 | 历史说明称新增图训练参数有默认值；本次仅确认该文件存在，未逐项复核所有参数定义。 |
| `scripts/train_wolfpack_sddfg_intra_episode_dynamic.sh` | SDDFG 动态实验主脚本 | 明确传入环境动态设置、图 schedule、gate、PPO guard、stale trust、recent window、delayed/future-match/success gate、graph-return、capture-to-win、pair/triplet complementary 参数并打印关键配置。各run的实际值仍须以保存config/console为准。 |
| `scripts/train_wolfpack_value_baseline_intra_episode_dynamic.sh` | VDN/QMIX/QPLEX 动态共同脚本 | 复用同一动态环境设置。 |
| `scripts/train_wolfpack_vdn_intra_episode_dynamic.sh`、`scripts/train_wolfpack_qmix_intra_episode_dynamic.sh`、`scripts/train_wolfpack_qplex_intra_episode_dynamic.sh` | 三个值分解基线的启动包装脚本 | 统一转发至 baseline 动态脚本，保持环境参数对齐。 |
| `scripts/validate_sddfg_dynamic_graph.py` | SDDFG 动态图完整预检/断言 | 已随机制演进加入 pair credit、outcome contrast、capture identity、support、event/episode 质量守恒、candidate loss、optimizer guard/state sync、topology persistence 和扩展 batch 检查；多次暴露旧 PyTorch API、维度、测试口径和状态同步错误。 |
| `scripts/analyze_wolfpack_dynamic_200k.py` | 200k 动态训练结果分析工具 | 当前工作区存在，尚未在本次整理中执行或验证输出。 |
| `scripts/debug_candidate_score_to_rank_counterfactual.py` | candidate score→same-population rank/boundary只读反事实 | run106后用于读取per-target transaction CSV；后续v3支持以provenance CSV做first-consumption-only独立generation累积，拒绝重复generation/event、identity/sign/order/participant冲突。工具不导入训练路径，不改变loss/optimizer/RNG。 |
| `scripts/debug_pair_evidence_cohort_overlap.py`及其horizon sweep输出 | strict pair正负证据时窗重叠与support-v6结构重放 | run107确认现有recent replay内正负重叠为0；v2 sweep得到首个结构性class-complete的最小extra pending为4、第二个为7，并明确当时缺immutable payload、policy age、hard资格、checkpoint与signed mass，因而结构可达不等于fully trainable。 |
| `scripts/debug_pair_pending_foundation_synthetic.py`、`scripts/debug_pair_evidence_cohort_overlap_synthetic.py`、`scripts/debug_pair_pending_production_integration.py` | bounded pending基础、结构重放与真实trainer集成测试 | 覆盖immutable payload、ring-buffer覆盖、TTL、stale重算、signed mass、状态机、两epoch原子提交、checkpoint、pair-only objective、回滚/default-off。历史上多次因fixture/schema或旧测试脚本不同步失败；必须以当前脚本版本与实际服务器输出为准。 |
| `scripts/debug_*_synthetic.py` | 候选身份、capture、outcome、pair credit、support、topology、factor loss 等合成诊断 | 包含`debug_candidate_identity_supervision_synthetic.py`、`debug_outcome_factor_loss_synthetic.py`、`debug_outcome_cohort_centering_synthetic.py`等；各阶段服务器 preflight 使用的测试数量和名称随版本增加。“测试通过”仅证明被覆盖的数学/接线条件，不证明训练性能。 |
| `scripts/results/wolfpack/<algo>/<experiment>/run*/` | 训练输出目录 | 路径模式已确认。当前文档记录至run162；run96--run162主要正式产物位于`scripts/results/wolfpack/sddfg/sddfg_intra_ep_4to6_r2_j1_rec30_seed1/`，但run111无目录，run130/131/135/139/140及若干失败attempt无完整summary。run101为版本不同步中断；run118以后增加boundary CSV，run149以后增加freeze/post-capture数据，run162增加bounded replacement诊断。 |

## 3. 当前代码状态与机制版本

### 3.1 episode 内动态 roster 与掩码

- 当前版本：环境支持同一 episode 内失效退出、延迟加入和恢复；张量仍以最大 player 数固定，使用状态/观测/action availability/active mask 表示有效槽位。
- 解决的问题：避免动态人数直接改变网络输入维度，保证加入、退出、恢复可贯穿训练与评估。
- 修改文件：`envs/Wolfpack/wolfpack_penalty_open.py`、`scripts/train/train_wolfpack.py`、`runner/wolfpack_runner.py` 及相关网络/缓冲区路径。
- 进入训练证据：20k 控制台曾记录 `num_players_mean=4.46`、最小 2、最大 6、`join_events=2`、`leave_events=4`、`recover_events=4`、`recovery_completion_rate=1.0`，且训练和 eval 均有相同动态事件统计。
- 性能结论：接线和生命周期触发已证实；不能仅据此认定 SDDFG 性能优势已证实。

### 3.2 动态图合法性、阶数和稀疏消息传递

- 当前版本：`adj_generator.py` 中存在有效 active factor、空 factor、order1/2/3、无效 factor、平均阶数、连通性/覆盖检查；动态图根据 active 状态更新候选与邻接。
- 进入训练证据：历史 console 记录 `adj_valid_factor_ratio`、`adj_empty_factor_ratio`、`adj_order2_ratio`、`adj_order3_ratio`、`adj_invalid_factor_ratio`、`adj_mean_order`；早期动态 20k 中 invalid factor ratio 为 0。
- 性能结论：图合法性指标可用且早期无效因子为零；结构合法不等于结构对捕获有效。

### 3.3 阶数 schedule、band、quota 与 pair/triplet 结构控制

- 当前版本：代码中已有 `adj_order3_bonus` 退火、`adj_sampling_temperature` 退火、min/max order3 ratio、soft quota、min pair ratio、triplet balance、synergy triplet feature、greedy sample probability/cap。
- 修改文件：`adj_generator.py`、训练脚本、预检、runner 日志。
- 已启用/触发证据：run23 以后日志要求并记录 bonus、temperature、max band、soft quota 等；run26 显示 quota 和 order-aware credit 使三阶比例偏高；run27 的 band 将结构拉回目标区间附近的意图被检验。
- 性能结论：硬 quota、band、soft quota 能改变阶数比例；多次结果显示“结构比例好看”未能稳定转化为 reward/capture/win，因此未证实其整体性能正效应。

### 3.4 order-aware credit、relative gate 与 triplet scorer

- 当前版本：learner 和 generator 中有 raw/weighted order2/order3 loss、`raw_o3_minus_o2_factor_rl_loss`、positive-only order advantage、relative credit gate、credit gate max delta、triplet credit EMA、synergy / advantage-aware / direct-rank triplet 打分。
- 修改文件：`r_sddfg.py`、`adj_generator.py`、`wolfpack_runner.py`、训练脚本、预检。
- 已启用/触发证据：run29--run38 被分别用于检查 credit gate、relative gate、synergy scorer、raw order credit、graph-level positive advantage gate、advantage-aware scorer；日志中要求记录相应 gate、EMA、raw/weighted loss、positive fraction、marginal score 指标。
- 性能结论：实现和日志接线已逐步完成；截至 run38，advantage triplet marginal 接近或低于零、score multiplier 约为 1，未证实该类打分带来稳定性能增益。

### 3.5 graph/factor PPO guard 与 stale replay trust

- 当前版本：`base_runner.py` 支持 `adj_ppo_clip_stop_ratio`、`adj_ppo_factor_clip_stop_ratio`、`adj_ppo_min_epochs`，记录实际 epoch、early-stop 和最后 epoch 的 clamp；learner/runner 有 graph/factor stale trust 相关诊断。
- 已启用/触发证据：run37 是“真正启用 high-clamp early stop”的验证；run38 用于 stale trust 诊断。run38 raw clamp 仍较高，而 trusted clamp 因权重下降而降低。
- 性能结论：early stop 与 stale trust 的接线/诊断得到验证；run37、run38 未显示性能正效应，不能将 trusted clamp 下降解释为训练质量提升。

### 3.6 recent-episode adj replay

- 当前版本：`adj_buffer.py` 有 `_recent_episode_indices()`、`sample_inds(..., recent_episode_window=0)` 与最近 episode 统计；`base_runner.py` 将 `adj_recent_episode_window` 传入采样并记录样本 episode 数与比例。
- 目的（历史实现描述）：只让 adj-PPO 使用最近 episode，同时保留完整缓冲区用于基线/factor advantage。
- 当前代码存在证据：2026-07-16 源码检索确认参数、调用和 runner 指标均存在，训练脚本默认可传 `ADJ_RECENT_EPISODE_WINDOW=4`。
- 训练证据：run49保存配置/历史分析确认启用了recent-window minimum/emergency相关设置，后续run也持续记录recent/stale指标。
- 性能结论：该路径已进入训练，但通常与多项credit和PPO机制共同变化，无法从现有run单独归因其性能作用。

### 3.7 基线适配与兼容性修复

- VDN/QPLEX/QMIX：已建立同动态环境脚本，SDDFG 专属图参数不应传给基线。
- QMIX：曾在动态实验报 `torch.nan_to_num` 不存在（Python 3.7 环境下旧 PyTorch）；之后 200k 结果被保存为 QMIX `run2`，说明兼容性问题已修正到可训练。修复的精确代码替换本次未逐行复核。
- 预检兼容性：`math.comb` 在旧 Python 环境不可用，当前兼容实现已替代该调用；另有预检 finite-gradient 断言与 greedy-cap 断言曾失败，说明预检也必须与实际配置同步。

### 3.8 参数解析、运行时兼容性与扩展 batch 修复

- 已发生错误：训练入口曾拒绝 `--adj_recent_episode_window 4`，之后也曾拒绝 `--adj_delayed_triplet_credit_require_future_match`；二者均为 parser 未识别参数导致的启动失败。`train_wolfpack.py` 当前源码已出现这两个参数及 recent-window、delayed/future-match 相关参数定义。来源：本次历史对话中的 Traceback 与当前源码检查。
- 已发生错误：`validate_sddfg_dynamic_graph.py` 调用 `torch.minimum` 时，在服务器旧 PyTorch 环境报 `AttributeError: module 'torch' has no attribute 'minimum'`。该错误发生在 `adj_generator.py` 的 triplet 置信度路径；本次未逐行核验具体替换提交，不能将当前代码存在的兼容路径写成该 run 已成功验证。
- 已发生错误：run48 在约 6,600 env steps 的 `train_adj_on_batch` 中报 `RuntimeError: The size of tensor a (400) must match the size of tensor b (6) at non-singleton dimension 1`，位置为 `r_sddfg.py` 的 `factor_training_mask` 相关乘法。后续代码已扩展 `AdjBuffer`、`r_sddfg.py` 与 `r_ddfg.py` 的 batch 字段处理；run49 能完成 2M，说明该阻断性形状错误不再阻止该次长程训练，但不能据此证明所有 batch 组合均无错。
- 性能结论：上述事项证明 parser、兼容和 batch 接线必须随机制同步维护；它们是运行正确性修复，不是性能增益证据。

### 3.9 delayed、future-match、success gate 与 graph-return triplet credit

- 当前源码状态：训练入口、`AdjBuffer`、learner、runner、训练脚本与预检均有 delayed triplet credit、future exact/partial/matched、success gate、graph-return credit 及 `require_delayed_gate` 的参数和日志路径。当前源码检查只能证明代码与预检路径存在；每个 run 是否启用仍由保存 config 和 console 确认。
- 已进入训练的证据：run49 的保存 config/console 被历史分析为启用了 `ADJ_TRIPLET_GRAPH_RETURN_CREDIT_REQUIRE_DELAYED_GATE=1`、success-gate floor、partial-match 权重、recent-window 最小/紧急设置，并记录 delayed/future/graph-return 指标。run50、run51 的运行配置进一步明确启用了 capture-to-win、delayed 和 graph-return 相关路径。
- 已触发但未证明性能正效应的证据：run51 的 delayed credit active fraction 平均约 0.118、末值约 0.130；future matched/exact/partial 指标均大量非零，而 reward/capture/win 显著退化。run50 的 capture-to-win triplet credit 在 200k 节点为零。信号记录或非零不能推出 credit 已正确归因。
- 性能结论：截至 run51，delayed/future/success/graph-return 组合没有形成已证实的 200k 稳定性能提升；涉及多轮多变量改动，不能将退化归因于其中任一单独开关。

### 3.10 capture-to-win 与 pair/triplet complementary credit

- 当前源码状态：`utils/adj_buffer.py` 存储 `capture_to_win_triplet_credit`、`capture_to_win_quality_gate`、`pair_pursuit_credit`、`pair_pursuit_quality`、`pair_to_triplet_transition_score`、`triplet_capture_quality`；`r_sddfg.py` 和 `r_ddfg.py` 兼容扩展 batch，runner、训练脚本、预检和日志路径均已接入。
- run50 配置/触发证据：`use_adj_capture_to_win_credit=True`，历史记录的参数为 coef=0.15、min outcome advantage=0.50、scale=0.75、cap=0.25、require future match=True；200k 时 `capture_to_win_triplet_credit_mean=0`、active fraction=0、quality gate mean=0。该硬 outcome 信号在该节点没有提供训练 credit。
- run51 配置/触发证据：除上述 capture-to-win 机制外，`use_adj_pair_triplet_complementary_credit=True`，pair pursuit coef=0.10、window=20、cap=0.20、min reward=0.0；console 与保存 config 均记录启用。pair pursuit credit active fraction 平均约 0.923、末值约 0.943，而 capture-to-win active fraction 平均约 0.046、末值为 0。
- 性能结论：代码、训练启用和部分信号触发均有证据；run50/run51 未显示性能提升，且 run51 的 pair pursuit 信号覆盖面过宽与低性能同时出现，只能记录关联，不能断言因果。

### 3.11 centered capture outcome、身份匹配与质量守恒

- run56 将 capture-to-win 改为 centered episode outcome：仅已结束且含真实 capture factor 的 episode 进入成功率基线，成功标签为 `1-p_success`，失败标签为 `-p_success`，单一 outcome 时归零。run56 的 definition version=2 和 `capture_to_win_outcome_contrastive=1` 真实进入训练。
- run57 definition version=3 将一个 episode 的总 outcome 质量分摊到该 episode 的全部 capture factor，目标是消除 capture/triplet 数量造成的 episode 权重放大；`capture_outcome_window_center_error_ratio` 用于检查展开后是否仍居中。
- run58 definition version=4 引入真实 capture event participant identity。该次 41 个训练 capture event 全部是两参与者事件，而当时代码只允许 order3 identity，导致 matched=0、outcome/local delta 全为0。该结果证明精确身份数据已进入环境/runner，但 factor 阶数定义不匹配。
- run59 definition version=5 采用 `highest_exactly_representable_capture_factors`：两人事件只匹配 exact order2 pair，三人只匹配 exact order3，超过三人仅匹配真实参与者组成的 order3 子结构；event 总质量先在匹配 factor 内守恒，episode outcome 再在 event 间守恒。run59 的 matched fraction 从 run58 的0升到 8.89%，说明 exact order2 路径真实触发，但 active graph 覆盖仍低。
- 当前约束：普通正 reward 不得触发 pair/outcome credit；strict future 要求 offset>0；不恢复 floor；identity 缺失、重复 event、count 不一致、非法 factor 或非目标 local delta 均应显式失败。来源：run56--run59 console、CSV、manifest、Traceback 与源码检查。

### 3.12 outcome replay support 与 final optimizer cohort centering

- support v1（run60）允许从 buffer 补齐缺失的 outcome 类，但稀有 episode 可跨 adjacency update 重复补入；其 23 个 augmentation 窗口不能证明样本独立。
- support v2（run61）以 slot-generation 记录跨 update 一次性使用，同一 adjacency update 的多个 PPO epoch 复用同一 cohort；缺失任一类别且无法原子补全时只关闭 outcome gate/loss，pair、graph 和普通 advantage 继续训练。run61 记录 enabled=13、augmented=13、exhausted=28，证明无限重复被约束，但也暴露一次性消费后的信号耗尽。
- support v3（run62 起）在最终 optimizer cohort 确定后计算 centered baseline；run62 的 class-complete/center-valid 窗口均为7，center error=0，正负 gate episode count 为7/7。该修改修复了“full buffer baseline 与实际训练 cohort 不一致”的数学问题，成功范围是 cohort 口径和零中心，不代表覆盖充分。
- run63 使用 outcome-confidence scaling v2（detached graph advantage magnitude），run64 使用 stored graph-return advantage 并增加 ready-source 要求；run63 与 run62 的十个 eval 点完全相同，而 manifest 不同，随后确认 graph advantage replay 写入路径未提供预期训练来源。run64 修复写入后轨迹发生变化，但未形成整体性能提升。

### 3.13 factor-local loss 与 candidate identity supervision

- run65 的 outcome factor loss normalization v2 以有效 graph transition 为分母并保持 target-local，避免 local delta 被无关 factor 数量再次稀释；run65 last5=2.956、last10=1.214，但 final=1.754、win=0.3、capture=2.4，因此数学接线改善不能写成整体达标。
- run66 首次加入 candidate-level identity supervision：active 与 candidate 严格互斥，只给真实 exact identity candidate 信号，不进入 actor PPO、Q target、reward 或 priority。candidate-only 仍为84.21%。
- run67 candidate loss v2 使用 signed `-Δ log(s)`；真实训练中 loss 非零，但该目标只优化未归一化绝对 weight，不能保证条件选择概率和 rank。
- run68 candidate loss v3 改为当前可求导 canonical candidate catalog 上的 conditional log-probability，且完成2M；其前200k仍弱，但2M最终达到6.718/0.6/4.6，说明该组合具备长程学习能力。由于 run68 同时包含此前多轮机制，不能把长程结果归因于 candidate loss 单项。
- run69 candidate loss v4 保持 conditional selection 语义，并将 normalization v2 改为只除以 target-bearing transition 数；candidate/graph loss约0.801%，证明“被全batch无关 transition 稀释”被实质修复，但 optimizer 后正/负概率正确率仍仅约13.52%/25.85%，根因转移到梯度和实际更新链路。

### 3.14 candidate 梯度、Adam actual update 与 lifecycle

- run70 gradient projection v1 在同一 update 内投影 base gradient，保护 candidate 梯度方向；正/负 optimizer 后概率正确率提高到84.38%/82.14%，final=3.555、win=0.8、capture=3.4。成功范围是同 update 单步方向与该 run 的行为结果；rank 改善仍仅约2.34%/2.68%，candidate-only仍80%。
- run71 actual-update guard v1 开始检查 optimizer 后真实参数位移；随后出现 `Adam raw update reconstruction is inconsistent with the optimizer state`，证明欧氏梯度安全不等于 Adam 实际位移和状态轨迹安全。
- run72 state sync v2 支持 legacy 与 bias-corrected denominator 两类 Adam 重建公式并按位移误差选择；训练完成，但“每次按误差拟合公式”和仅重建一阶矩的长期自洽性不足。run72 candidate loss下降100%，正/负概率正确率86.67%/88.89%，rank仅1.11%/1.85%，行为低于 run70。
- run73 state sync v3 根据实际 optimizer/group 属性使用标准 Adam 公式，并加入 lifecycle v1；lifecycle 仅保护聚合梯度、TTL 曾按 optimizer step 计时，且 target-bearing update 未完整受旧缓存保护。run73 lifecycle protected update=52、单 transition violation窗口=23、rollback率32.21%。
- run74 实际 console/manifest 为 lifecycle definition version=3（用户请求中曾称v2，现按落盘证据修正）：以 adjacency round 为时钟、horizon=4，对 target-bearing/base-only update 应用逐 transition active-set 约束和 evidence supersession。投影后的欧氏最小 dot 约 `-6.91e-10`（容差内），但 Adam 实际位移最小 dot 约 `-1.625e-5`；rollback率33.78%，100.8k首次出现并在120--160k达到47.73%。这表明主要异常位于欧氏投影之后的 Adam 位移，而非 active-set 线性约束本身。
- run75 已训练 lifecycle v4：在实际 Adam realized displacement 空间逐 cached halfspace 投影，使用确定性非线性 backtracking、单次最终 state sync，修复 TTL off-by-one，并加入只读 observation archive 以精确统计 age1/5/10 保持。run75 中实际位移投影修正18个日志窗口、修正后负约束计数为0、rollback为0，证明实现真实生效；但 final=1.788、last5=1.003、capture=2.8，不能写成性能成功。run75审计后只修复了一个 target-bearing 诊断漏记问题，没有改变训练目标、梯度、optimizer或lifecycle保护逻辑。

### 3.15 candidate v10--v12、独立 residual Adam 与 lifecycle v9--v10

- 候选目标演进：本次历史对话确认过 candidate v10 的 first-reachable active-competitor signed hinge、normalization v3；run83起的 v12 在 PPO early-stop 后执行有限的 candidate residual。v12 要求 residual 前所有 adjacency 参数的 `.grad` 显式置为 `None`，无 candidate 梯度的参数不得进入 step，inactive 参数出现任何非零位移即断言失败。
- 双 Adam 隔离：主 adjacency epoch 仅使用 `adj_optimizer`；residual 仅使用 `candidate_residual_optimizer`。projection、backtracking、rollback、state sync、save/load 都必须按实际 optimizer 分离，且 residual 不携带主 PPO/factor/candidate 之外的动量，也不刷新 TTL、跨 update 缓存样本或创建新的 lifecycle 注册。run83以后的完整训练系列以该定义为主路径；当前可见对话没有逐 run 的完整 residual state 审计数值，故不能写为已证明性能增益。
- lifecycle v9--v10：v10 将跨版本保护限制为最终提交后产生 canonical rank 改善或 signed competitor goal crossing 的 target；纯微小 margin 改善不注册、不刷新 TTL、不更新 reference 或进入 projection。run84/85被描述为该机制和双 Adam checkpoint 的首次完整验证，但本对话未保留其全部定量日志；只能确认版本意图和后续训练曾完成，不能归因性能。
- 已报告的实现错误：run83阶段合成测试曾以参数位置索引 optimizer `state` 导致 `KeyError: 0`；训练还曾在 `r_sddfg.py` 的 `train_adj_on_batch` 中因 `final_candidate_info` 未赋值报 `UnboundLocalError`。后续 run84 完成表明这些阻断性错误已被处理到可训练，但当前对话没有保留精确补丁与回归输出，仍属历史修复证据不完整。
- 关联测试口径错误：pair机制演进期间，`debug_outcome_factor_loss_synthetic.py` 曾因缺少 `compute_capture_outcome_factor_ppo_loss` 导出报 ImportError；`validate_sddfg_dynamic_graph.py` 的 `validate_adj_buffer` 曾断言 `last_sample_episode_count == 2` 而失败。二者是测试/接口同步问题，不能把其发生时的中断误判为算法性能结果；当前可见对话没有保留各自精确修补提交。

### 3.16 terminal/事务 checkpoint 与 PyTorch 1.3.1 约束

- run84--run85阶段加入 terminal policy 强制保存，以及 periodic/best/terminal checkpoint 的 completion metadata。恢复路径要求模型、标准 adjacency Adam、residual Adam和metadata完整；缺失或不一致时 fail-loud，不允许静默重新初始化 optimizer 或 replay。
- 该路径的目的为复现正确性而非性能机制。run85之后的对话要求 checkpoint 区分完成标志、最后提交 step和双 Adam state；本次可见材料没有逐文件重新验算全部 run84/85 产物，不能把 checkpoint 正确性写为行为提升证据。
- 服务器训练环境为 PyTorch 1.3.1。新增路径不得依赖 `torch.linalg`、`torch.minimum`、`torch.maximum`、`torch.count_nonzero`、`torch.nanmean`、新版 AMP、`torch.load` 新参数等较新 API；run95之后工作区的 v5 修改仅在本地 PyTorch 1.8.1+cu111 完成定向测试和静态旧版 API 审查，尚无真实 1.3.1 长程运行证明。

### 3.17 trusted-control、population-total aggregation 与 recent replay 执行契约

- run86前已确认的数据流错误：trainer 计算并声明 trusted stale population/control 已启用，但 `base_runner.py` 的 early-stop 和 recent-window 仍读取 raw population，recent window 长期锁在1，adjacency batch平均只含约1.24个 episode。该问题是运行时主链错接，不能用阈值或学习率解释。
- 随后实现将 stale-trust 下的 selected graph/factor control 供同一 early-stop 和下一次 recent-window 状态机使用；actual early-stop 只表示实际少运行了 epoch，而非最后配置 epoch 命中阈值。run87历史摘要记录 window均值1.748、57.9% update为window=1、recovery287次、最长连续window=1为88个 update，说明修复后不再永久锁死，但仍存在后期窗口下降与性能不稳。
- run89前又确认聚合错误：逐 mini-batch `ratio_i=n_i/d_i` 被等权平均，不能代表真实 loss population。当前主路径应传递 raw/trusted graph/factor numerator 与 denominator，并以 `r_control=sum(n_i)/sum(d_i)` 聚合。控制与 loss population、selected/trained unique generation、chunk partition须按 update/transaction 记录。该修复改变训练轨迹，但 run89/90未显示稳定超过200k强基线，故仅能称为控制语义修复。

### 3.18 outcome-conditioned signed pair credit、identity-local PPO 与归一化

- run91引入 outcome-conditioned signed pair credit：只基于真实 capture、strict-future delay、nearest valid transition与 exact canonical identity；按 evidence cohort 中心化 outcome，single-outcome cohort 为零，每 episode与 cohort 保持质量守恒。该机制不进入 actor、Q target、priority、环境 reward或 shared graph advantage。
- run92修复了 pair credit 写入 shared `f_advt` 后向 non-target factor 广播残差的问题：pair credit 改为 identity-local PPO，只向精确 order-2 target 注入显式 pair advantage，pair-only 信号不改变 graph loss。softmax 的隐式参数耦合仍可能改变其他概率，不能将“non-target 显式 label 为零”误写为“non-target 概率必为零变化”。
- run93将 pair-local 分母从所有 valid transition 改为 pair-target-bearing transition（normalization v2）。后续审计发现 full-buffer 中心化后切片到 optimizer cohort 会形成单边标签；v2会放大这种错误。run94因此改为最终 optimizer cohort 重新中心化，严格令 one-sided optimizer cohort pair loss 为0。

### 3.19 pair-evidence support、原子 transaction 与历史 v5 代码状态

- run95前的根因是：generic capture support 不保证补齐 strict-future exact pair-evidence 的成功/失败类别，且双边 selected cohort 仍可能在不同 Adam step 消费。run95实现 pair-evidence class-complete support、mixed pair cohort 原子 partition、transaction内重新中心化和 optimizer-step signed-mass 守恒；非零 pair transaction 必须同时含正负 evidence，one-sided transaction pair loss为0。
- run95日志验证了质量守恒，但性能并未改善：42个非零 pair transaction的 signed-mass contract 全通过，最大中心误差约`3.725e-9`，正负总质量均为`1.058333`；pair loss绝对均值约`2.04e-4`、base factor loss约`2.679e-2`，比例约0.761%。因此质量守恒/接线正确不能代替性能成功。
- run95源码审计又确认原子 partition 的副作用：旧实现把全部 pair chunks 固定放在第0个 partition，余下 partition承接全部 non-pair chunks；在2个分区时，N=6 的pair transaction仅2 chunks而普通transaction有4 chunks，导致每个 transaction 的 base factor PPO batch人口系统性不同。当前工作区已将 `utils/pair_credit.py` 的 `partition_pair_contrast_optimizer_chunks` 改为将零-pair filler 填到普通 `array_split` 目标大小，并由 `utils/adj_buffer.py` 在 class-complete 且 chunk 数为奇数时用 replay RNG 选择公平 slot；`runner/base_runner.py`、训练脚本和 cohort 合成测试同步记录/验证 v5。该 v5 修复尚未经过真实训练，不能宣称性能效果。
- 当前v5定向测试：Python语法检查、pair/support/cohort/capture/identity/factor-loss合成测试、candidate identity共42项、stale-trust、eval graph、topology和完整动态验证均在本地PyTorch1.8.1+cu111环境通过；`git diff --check`通过。服务器PyTorch1.3.1只由run95之前路径的真实训练间接证明，v5仅完成静态兼容审查。

### 3.20 support v6与per-epoch optimizer transaction diagnostics v2

- run96运行时记录`adj_outcome_contrast_replay_support_version=5`；run97起记录version=6。support v6的源码语义是：class-complete pair evidence不再切分为多个不等人口partition，而是以一个完整selected replay population执行一个standard adjacency Adam transaction；pair-bearing与zero-pair chunks共享同一graph/base normalization，pair-local loss仍只读取exact target。selected/yielded/trained、generation去重、population-total和signed mass均须闭合。
- run98、run99、run100均为fresh seed=1、120k，六个eval点及公共训练字段一致。run99加入聚合pair gradient诊断；run100新增`run100_progress_train_adj_transaction.csv`，对每个真实PPO epoch独立记录pair/base/combined gradient、clip前后、Adam moment、raw/final displacement和exact signed score。run100与run99公共字段逐项一致，证明diagnostics v2轨迹中性。
- run100共有六条nonzero pair transaction：77.6k、89.6k、108k各两个PPO epoch。77.6k/0与89.6k/0在combined gradient阶段已反向；77.6k/1与89.6k/1的combined gradient转正，但Adam raw displacement仍反向；108k两个epoch的combined、Adam、final和exact score均正确。六次均未发生clip方向变化，raw Adam与final committed displacement相等，candidate/lifecycle/rollback不是最终位移反向来源。
- 关键数值：77.6k/0的pair-combined descent=`-1.9763e-3`、Adam pair-descent dot=`-3.2355e-5`、exact signed score change=`-9.9540e-4`；89.6k/0分别为`-6.2115e-4`、`-1.3833e-5`、`-2.7915e-4`。77.6k/1与89.6k/1的combined descent分别为`2.4751e-3`与`9.2704e-5`，但Adam pair-descent dot仍为`-1.0062e-5`与`-1.0861e-5`。108k/0与108k/1的combined descent分别为`1.3617e-3`与`4.4431e-4`，Adam pair-descent dot为`2.1009e-5`与`1.9479e-5`，exact signed score change为`5.3362e-4`与`4.8884e-4`。来源：run100 transaction CSV；来源：本次历史对话整理。

### 3.21 per-objective gradient diagnostics v3与graph advantage source v2

- run101因`r_sddfg.py`与`base_runner.py`的diagnostics版本未同步，在`_build_adj_transaction_row`抛出`RuntimeError: unexpected pair optimizer diagnostic version`后退出。该run无效，未用于性能、gradient、optimizer或checkpoint分析。
- run102是同步完整v3后的fresh seed=1、110k有效实验。transaction CSV保留v2字段并新增graph PPO、base-factor PPO、capture-outcome、pair、candidate、entropy各自的active、scalar loss、gradient norm、pair dot/cosine/descent，以及scalar/gradient reconstruction、projection前后与delta字段。run102与run100截至110k的公共v2字段和训练轨迹一致，证明v3只读诊断轨迹中性。
- run102在77.6k/0的per-objective pair dot中，graph=`-3.4724e-3`为最大负贡献，candidate=`-6.3623e-4`次之，base=`2.9727e-6`、outcome=`2.4897e-5`同向，entropy约`5.81e-8`可忽略；89.6k/0中graph=`-5.1616e-4`仍为最大负贡献，candidate=`-4.2557e-4`，base=`9.8015e-5`同向，outcome约`4.00e-7`、entropy约`-1.07e-7`。两次反向都在projection前已经发生，graph是共同的最大负投影来源；candidate在89.6k/0也构成实质冲突，但日志与源码没有证明candidate存在固定实现错误。
- 源码审计确认graph PPO错误使用混合后的`f_advts`均值，使identity-local、delayed pair/local credit被广播进共享graph objective；replay中已有独立graph-return advantage，但原路径解包后没有正确用于graph PPO。该问题不是正常objective冲突，而是advantage source责任范围错误。
- graph advantage source v2修复后，graph PPO只消费`α × replay graph-return advantage G`，factor-local路径使用`f_advt-αG=βL+D_local`，并fail-loud验证graph+factor-local精确重构原`f_advts`；pair/candidate/outcome loss、Adam、学习率、epoch、support和超参数均未修改。run103日志中`graph_advantage_source_version=2`，contamination契约成立；该修复从首次adjacency训练开始改变轨迹，故run103不要求复现run102的pair cohort step。

### 3.22 pair-evidence funnel v1/v2与candidate score→rank→active诊断

- run103为graph source v2后的fresh seed=1、110k实验。其20k--100k reward为`-5.187,-3.395,1.340,0.977,3.269`，win为`0.1,0,0.2,0.2,0.5`，capture为`0.8,1.8,3.2,2.7,3.4`；五点评估均值reward为`-0.5992`，100k单点较好但不构成持续性能成功。截至110k，negative pair evidence available update=38、positive=0、class-complete=0、pair target/gradient/optimizer transaction均为0。graph source错误已修复，但最早监督断点前移到成功capture→positive strict pair evidence。
- run104加入轨迹中性的pair-evidence funnel v1，并与run103公共轨迹一致。累计successful episode=4、successful capture episode=4，四个成功capture全部为candidate-only，successful active capture=0，positive pair evidence=0，successful capture without pair evidence=4；日志与源码未发现terminal off-by-one、participant=2、dynamic slot、canonical identity、support选择或capture provenance错误。最早断点是successful candidate capture尚未成为active factor。
- run105加入funnel v2、`run105_progress_train_pair_evidence_episode.csv`、episode-level reject reason与candidate boundary join；公共字段与run104逐项一致。reject reason集中在`CANDIDATE_ONLY_NOT_ACTIVE`，而不是identity/provenance/terminal错误。唯一成功replay generation为370，包含order-2 identity`0-4`与`2-5`，二者均收到正candidate target。
- run106加入`run106_progress_train_candidate_identity_transaction.csv`，每个真实unsatisfied candidate target按epoch记录同一合法population内的pre/post margin、rank、strictly-better/tie、next-better gap、boundary deficit、gradient、projection、Adam和lifecycle。run106与run105公共轨迹一致；四条successful-overlap行全部`same_population_rank_reconstruction_valid=1`。
- generation370的四条真实target表明：`0-4`在epoch0/1的signed margin分别改善`5.6267e-5`/`1.2898e-4`，rank始终23，next-better gap从`0.066002`降至`0.064315`，boundary deficit约2.234；`2-5`改善`2.3317e-4`/`5.3954e-4`，rank始终24，next-better gap从`0.109466`降至`0.108803`，boundary deficit约2.285。candidate gradient非零、combined/Adam committed方向正确，无lifecycle reject/rollback；rank不变的直接原因是实际margin改善远小于合法next-better gap，而不是score/rank缓存或population错误。
- run106只读counterfactual将四行全部分类为`IMPROVEMENT_BELOW_NEXT_BETTER_GAP`。实际更新占next-gap的比例依次约0.085%、0.213%、0.494%、0.198%，占active-boundary deficit约0.0025%、0.0102%、0.0236%、0.0058%；同一generation的两个PPO epoch不构成两个独立行为证据。该工具不导入训练路径，未修改candidate loss、系数、active cutoff、reuse或Adam。

### 3.23 provenance-complete candidate evidence与独立generation反事实

- run106后新增`runXXX_progress_train_candidate_evidence_provenance.csv`。一行严格表示“一个真实capture event × 一个canonical candidate identity × 一个sign”；字段闭合policy、generation、environment episode、event、prey、capture/terminal step、participants、identity/order、sign、event quality、identity分配、final target mass、target transition、base/support、policy age与去重/质量contract。PPO两个epoch和replay重复曝光不会新增独立evidence行；同event多identity按identity weight分配并重构event质量。
- 接线阶段曾出现三类阻断：`validate_sddfg_dynamic_graph.py`因provenance rows不能重构final candidate target tensor而失败；`debug_outcome_factor_loss_synthetic.py`的旧fixture仍为`[3,2]`而production要求`[3,3]`；`debug_candidate_identity_supervision_synthetic.py`的旧fixture为`(2,2)`而trace要求`(2,3)`。这些是provenance schema/fixture同步问题，不是训练算法结果；后续run107/108成功生成完整provenance CSV，说明阻断已处理到可训练。精确每个补丁的提交信息未保留。
- run107为fresh seed=1、200k provenance观测实验。其前110k与run106公共轨迹一致，证明provenance日志轨迹中性。provenance CSV有102行、102个唯一完整键，positive/negative candidate event row为35/67，`provenance_complete`、identity和quality contract失败均为0。
- v3独立generation工具使用first-consumption-only：later replay exposure与第二PPO epoch不作为新证据。`0-1`有3个独立generation，累计signed margin change=`0.0038519`且有1次方向抵消；`0-4`有2个generation，累计=`-0.0004511`并发生方向抵消；`2-5`有2个generation，累计=`0.0051920`且无抵消。三者均未在反事实中跨过next-rank或active boundary；population/competitor在generation间变化，工具明确标为scalar-transfer isolation，不等同于真实训练轨迹。

### 3.24 strict pair时窗重放、bounded pending与run108后的历史代码状态

- run107的严格pair证据有2个positive generation（689、709）与36个negative generation；successful active capture exposure=12，但现有recent replay内正负时窗交集为0，class-complete、pair target、pair gradient、pair transaction均为0。cohort-overlap v1确认`opposite_sign_overlap_pair_count=0`、`positive_blocked_only_by_shared_used_state_count=0`、nonzero commit=0、reuse=0；因此断点不是support漏选或one-sided used-state提前消费，而是negative自然离开recent replay后positive才到达。
- horizon sweep v2扫描0--7个额外adjacency update并按support v6结构规则重放。horizon=4首次形成结构性cohort：144.8k的`689(+) + 660(-)`，5 episodes、100 selected/yielded/trained chunks、单partition、无duplicate/reuse；horizon=7再形成156.8k的`748(-) + 709(+)`。当时两者均非fully trainable，因为run107日志没有evicted immutable payload、behavior policy age、hard资格、checkpoint pending state和transaction-time signed mass，故没有直接把离线TTL接入训练。
- 后续production实现增加真实capture event provenance、34字段detached immutable snapshot、ring-buffer覆盖安全、TTL adjacency-update时钟、transaction-time current-policy forward/stale-trust与effective signed mass重构、pair-only objective、pair/generic used-state分离、两epochouter atomic transaction、参数/optimizer/RNG/lifecycle/pending回滚和checkpoint/resume。默认配置为`pair_bounded_pending_evidence=false`、`pair_pending_max_adj_updates=0`；run108单变量启用`true/4`。
- run108确实创建26个snapshot并出现一次current/pending overlap与class-complete pair-only事务：144.8k的`660(-)` pending age=4、policy age=32，与`689(+)` current、policy age=4组合；raw正负质量均约0.05，第0 epoch stale trust正/负为0.625/1.0，effective mass约0.03154/0.05，pair loss=`0.0083798`、pair gradient norm=`0.0028520`，standard optimizer step从692到693。随后通用graph/base PPO early-stop被错误用于pair-only事务，事务在只完成1/2 epoch时以`EARLY_STOP`中止，参数、optimizer、RNG、lifecycle和pending状态完整回滚；logical commit=0、zero-target/zero-gradient abort=0、reuse=0。
- run108与run107的eval、progress、train、train-adj、transaction v3公共字段、topology和Q/RNN/capture字段截至200k逐项一致。run108没有净pair更新，因此不能把其行为指标解释为pending机制效果。`run108_progress_train_pair_pending_cohort.csv`曾把abort row的`objective_scope_contract_valid`硬编码为0；实际第0 epoch若scope失败会以scope错误中止而不会到`EARLY_STOP`，故该0是日志错误，当前runner已改为记录真实per-epoch scope结果。
- run108后源码修复将pair-only事务的control population定义为exact nonzero pair targets，记录raw/trusted clip numerator、denominator和ratio；pair-only明确`standard_ppo_early_stop_applicable=0`、`all_configured_epochs_required=1`，不再调用通用`should_stop_adj_ppo`，仍保留PPO clip、stale trust、finite/nonzero、optimizer-step、exception与完整rollback检查。普通graph/base adjacency PPO early-stop路径不变。`PAIR_PENDING_DIAGNOSTIC_VERSION`已从1升至2。
- 当前正确integration测试函数为`test_standard_ppo_early_stop_is_not_applied_to_pair_only`，要求通用early-stop调用次数为0、返回两个epoch transaction row并只logical commit一次。最新一次用户服务器Traceback仍执行旧函数`test_early_stop_rolls_back_without_crashing`并断言`rows == []`；当前工作区第548行已是新monkeypatch，第561--562行断言`calls==[]`和`len(rows)==2`。这是服务器测试脚本未同步的版本冲突，不能通过恢复旧生产行为来“修复”。当前目标测试脚本SHA256为`969b492e5ef51a04bf83ca2c027661b0d6b5ba9e8b87f8a3d56dc73cea8855c3`；同步后的最新服务器回归状态尚未确认。

### 3.25 v12--v14：逐target事务、selection boundary与联合exact提交

- v12由run117完整200k验证。step=0独立production preflight、True/4 pending、current-priority、candidate/pair/lifecycle组合、exact revalidation和原子事务均真实运行；pending及ordinary target-epoch共44/44方向正确，reverse=0、zero=0、rollback=0。成功范围是逐target exact score和事务正确性，不包含真实selection boundary、rank或active闭环。
- v13由run118部分验证，首次引入production generator定义的pair-specific first-reachable boundary competitor、signed margin、canonical rank与active诊断。已提交26个target全部exact正确且boundary margin改善，median margin delta约0.008966、worst约0.000148，但crossing/promotion/eviction均为0。run118约182.4k中断；boundary非线性修正后只复核boundary，没有在同一回溯循环联合复核candidate和已接受lifecycle，optimizer.step后终检暴露约束失效。
- v14把boundary、candidate、lifecycle和数值有限性放入同一真实forward backtracking与rollback。run119是fresh seed=1、True/4、20k完整阶段验证；无actionable boundary target时与run117前20k公共训练轨迹一致，无Traceback/NaN/Inf且terminal checkpoint完整，证明no-target路径和正式RNG未因v14改变。run119没有真实boundary target，不能证明boundary性能。
- 来源：run117--run119 console、config、transaction/boundary CSV、preflight输出、manifest与checkpoint；来源：本次历史对话整理。

### 3.26 v15--v17：deficit预算、water-filling与identity aggregate masking

- v15基于run118的真实deficit分布回收zero-deficit target的大额预算，仅保留其float32 strict floor，并把原v14总boundary预算分给deficit-bearing target。run120的required improvement真实完成率约99.8%，deficit reduction中位约0.813%，但所有linearized crossing均不可负担，rank保持3→3，crossing/promotion/eviction均为0。
- v16在总预算不变的前提下按pre-deficit升序做有界water-filling。run121证明最近exposure获得约92%--99%预算，但同一canonical identity的其他exposure仍是独立强约束；150.4k的`0-2(+)`多exposure事务最终scale为0.5与0.03125，最近member post deficit约0.225422、rank 11→10、crossing=0。总体deficit reduction中位仅由run120的0.813%升至约0.933%。
- v17把同一`canonical identity + sign`的member Jacobian与margin进展求和。run122已提交事务预算守恒，但group completion约99.96%时最近真实boundary member completion仅约28.9%，形成aggregate masking；149.6k ordinary strict-pair事务在optimizer.step后找不到联合exact安全scale，外层原子rollback完成后训练中断。run122不是完整160k终点实验，最后完整evaluation为140k。
- 来源：run120--run122 boundary/transaction/group CSV、console与本次历史对话整理。

### 3.27 v18--v20：progress member、最大固定方向scale与同预算多方向

- v18将identity group extra progress改为唯一nearest positive-deficit progress member，其他member继续保留exact score与boundary non-regression。run123完整160k，group required/actual统一为progress member同一口径，安全越过run122失败区间；但150.4k的seq720/721仍分别只完成约50.26%/3.12%，scale为0.5/0.03125，最近member post deficit约0.225422，crossing=0。
- v19修复dyadic-only backtracking：halving先建立真实safe lower/unsafe upper，再做12次production forward refinement并提交最大已验证安全scale。run124中seq720由0.5恢复到0.691650、completion升至69.40%；seq721最大安全scale约0.024597、completion约2.46%。最近member deficit约0.212795，rank 11→10但crossing=0。该run证明固定方向的搜索精度部分恢复，也证明继续增加同一方向refinement不能改变方向自身的可行域。
- v20在总budget、pair coefficient和学习率不变时构造Adam-based progress fractions `1,0.75,0.5,0.25,0.125`，每个方向独立做真实联合forward搜索，并以original-required progress排序。run125中seq720仍选full、scale0.691650、completion69.40%；seq721改选0.25、scale1、actual约0.032033、original completion约24.63%，约为run124对应actual的10倍；最近member deficit约0.183960，但crossing/promotion/eviction仍为0。
- run125训练时schema v6未完整落盘全部未选候选的cosine、active-set、safe/unsafe和limiter，因此当时只能验证selected方向效果，不能独立重建全部候选几何。
- 来源：run123--run125 console、boundary/transaction CSV与本次历史对话整理。

### 3.28 v21、run126与当时的v22代码状态

- v21在v20五个Adam候选之外加入三个`progress_tangent`候选，fractions为`1,0.5,0.25`；proposal为等norm的`0.5 * Adam descent + 0.5 * normalized progress-member Jacobian sum`，总候选8个。每个候选从同一事务参数快照独立执行halving、真实production forward和12次refinement；选择键依次为最差group original-required completion、mean completion、safe scale和确定性ordinal。
- run126实际产物已完成复核：fresh seed=1、True/4、160k、v21、boundary schema v7、candidate schema v1、training_complete=true、terminal checkpoint存在；manifest SHA256=`78ab7e7844eac8fb0064208e9f53cc9c103fe5099ca8756c21046b0f496388cb`。训练环境PyTorch1.3.1、CUDA不可用并明确skip；console/CSV无Traceback、NaN或Inf，transaction sequence 0--763连续。
- run126在6个strict-pair epoch生成完整8候选，另2个无多member正deficit目标的epoch只生成单一Adam候选。progress-directed相对full Adam的cosine范围约0.939818--0.983500，Adam fractions范围约0.838219--1，active constraint set也会变化，故新增候选不是纯标量缩放。selected候选为5次Adam、3次progress-directed；selected方向均scale=1，无candidate/lifecycle/nonfinite/competitor-switch限制。
- run126首次参数分叉发生在144.8k、transaction seq693：run125选择旧Adam family方向，run126选择`progress_tangent/1.0`。34个boundary member全部exact与boundary方向正确，reverse=0、zero=0；14个正deficit member的deficit reduction fraction中位约2.1198%，ordinary中位约2.1198%，但crossing/promotion/eviction仍为0。最近member post deficit约0.105473、rank 1→1；按当次actual约0.0034366作纯描述性比例，仍相当于约31个同量独立epoch，不能当作真实未来轨迹预测。
- run126审计确认v21 progress seed把每个identity group的代表member都求和，包括extra budget=0、deficit=0的group。seq736的零预算`1-3(+)` member仍进入方向，并在Adam/0.125候选中成为boundary limiter；这会让“只负责保护”的group重新参与“向谁前进”的方向定义。当前v22代码只让`identity_group_extra_budget>0`的唯一progress member进入`deficit_progress_seed`，但所有member exact与boundary non-regression硬约束保持不变；transaction diagnostics升级为v22、boundary schema v8、candidate schema v2。该v22状态通过静态、replay、candidate反序、CPU/CUDA production、rollback、checkpoint、RNG与no-target测试，尚无新正式训练run。
- run126没有同一production selection context下commit→下一ordinary→第二ordinary的margin重放字段；相邻事务transition/prefix不同，故未证明ordinary update系统性抹除progress，也未证明rank-deficit retention存在真实需求。
- 来源：run126 console、config、`logs/summary.json`、evaluation/transaction/boundary/candidate CSV、TensorBoard、manifest、terminal checkpoint与源码检查；v22状态来源：当前源码与本轮前一任务的测试输出；来源：本次历史对话整理。

### 3.29 freeze countdown、joint epsilon与post-capture greedy floor

- run147--run149把训练exploration统一为一次joint epsilon Bernoulli、`epsilon_start=1.0`、`epsilon_finish=0.05`、anneal horizon=228k。run148的fresh 60k比旧探索增加basic capture，但first-capture到第二只不同prey的转化仍弱：formal 268 episodes、58 capture episodes、67 captures、6 exact-matched episodes、1 training win；7个multi-capture no-win interval为26、65、66、132、134、143、167，median=132，均超过24步freeze窗口。
- `food_frozen_time`源码和production trajectory确认是remaining countdown而非elapsed time。run149把每只prey的normalized remaining countdown加入local observation与central state，维度由33/57变为35/59。真实trajectory验证unfrozen严格为0、frozen位于(0,1]、逐步下降、thaw/reset归0；offscreen frozen prey仍保留该字段，remaining=24与1在env、runner、replay、sampled minibatch和production RNN/Q输入各层均不同，未发现旧33/57 slice、slot identity、dynamic roster或train/eval漂移。
- run151的200个production-shaped counterfactual中，将countdown从1.0改为0.04使greedy joint action在118/200 context变化，factor-Q delta median约0.01395、joint best-Q delta median约0.00765，故“字段进入网络但没有functional sensitivity”被排除。run151 60k中first capture 89次、strict distinct 2次、training win 2次；失败窗口non-explore单步second-prey progress约+0.151，explore约-0.0079，成功窗口约+0.583对-0.111，说明最早行为断点是post-capture完整greedy exposure不足。
- run152起只在“至少一只prey frozen且至少一只仍active”时应用`post_capture_joint_greedy_floor=0.25`：effective epsilon为`min(base epsilon,0.75)`，仍只抽原有一次joint Bernoulli。run152 20k production中eligible/floor-active均为220、non-explore 52/220=23.636%，capture后的第一个可行动decision立即生效，thaw、双frozen、reset立即退出；无提前/滞后、额外RNG、双Bernoulli、非法action或dead-slot违规。run153 60k相对run151把strict distinct从2增至4、win从2增至4、interval median从15.5降至7，首次说明局部floor改善的是first→distinct→win而非basic capture。

### 3.30 terminal-gated 24-step completion credit与weighted replay lane

- run156长程轨迹确认成功与失败最早稳定分叉在first capture当下的双prey几何：成功时通常已有其他alive player靠近remaining prey；terminal transition可公平进入uniform replay，因此当时第一高概率瓶颈从输入、reward与replay接线后移到one-step TD传播速度。
- run157把全部transition无条件改为`q_n_step=24`。该实现把普通dense reward和penalty也累计进每个target，使Q-target std约放大5倍、loss/TD/gradient膨胀、gradient clipping约44%，basic capture坍缩且60k无training win。该run证伪“更长return天然有利”，并明确固定多步return必须按真实completion语义门控。
- run158改为terminal-gated 24-step：未来24步内可达真实win marker时才使用multi-step completion return；没有marker时严格保持legacy one-step；`continue_after_success`下done虽为false，win marker仍是completion return语义终点。run158确认数值稳定、basic capture恢复、completion bonus是target gain主体，但真实completion-credit极稀疏：5个terminal episode、uniform sampled 7次、168 gated transitions，约占基础训练transition的0.029%。
- run159保留uniform replay及其RNG不变：若一个train interval的第一次uniform update未自然命中成功episode、且buffer已有成功episode，则从确定性round-robin terminal lane附加1个成功episode，仅terminal前24个transition进入auxiliary loss。production中credit利用量约提高6.57倍；但forced auxiliary transition weight=1.0使24个高残差transition支配uniform Q population，首次forced sample后loss与Q/RNN gradient上升，greedy second-prey progress由正转负，basic capture和distinct/win退化。
- run160把forced auxiliary gated transition weight降为0.10，uniform与natural terminal仍为1.0，lane频率、q_n_step、gate、reward、epsilon、floor、gamma与graph不变。60k production确认aux signal仍非零但不再支配uniform objective：Q/policy/RNN gradient稳定、clip为0、basic capture恢复、first-capture geometry左移、+4/+8 progress改善、strict distinct与training win均由run159的1恢复为2。run161用相同语义扩展到80k，前60k与run160一致；60--80k没有重新出现loss population失衡，故temporal gate、replay quantity与0.10 population weighting在当前单seed预算内视为闭合。

### 3.31 bounded post-capture exploration与run162 production证据

- run161在Q与capture稳定后暴露新的行为断点：first capture后的nearest-player progress在+4/+8尚可，但+12减弱、+24明显回撤。transition级good-start failure分析显示greedy/non-explore仍为正progress，而legacy explore会把所有alive slots独立随机化，单次joint explore与greedy action的alive-slot Hamming median为4、p75为5、最大6；这会频繁破坏已形成的多agent追击几何。
- run162保持post-capture explore概率、一次joint Bernoulli、available-action legality、global epsilon与floor不变，只把eligible explore branch从“随机化所有alive slots”改为“均匀无放回随机至多1个alive slot，其余保持greedy”。非post-capture路径与`max_random_agents=0`精确保留legacy语义；checkpoint/exploration contract升级为v4并fail-loud记录该参数。
- production验证：首次bounded explore在step 7849，run161/run162此前的joint episodes、post-capture记录、capture与Q公共字段一致；eligible decisions=2877，explore=2124、non-explore=753，explore fraction=0.7383，与run161的0.7364相当。2124次explore全部replacement count=1，随机slot覆盖0--5，随机action覆盖0--6；explore joint action unique ratio=1971/2124=0.928。greedy-final Hamming由run161 median4、mean4.2407降为run162 median1、mean0.8583，且无non-explore→greedy mismatch、非法action、dead-slot或joint-flag错误。
- good-start failure的topology-clean explore单步progress由run161约-0.1162改善到run162约-0.0522，retreat fraction由33.42%降至23.62%；non-explore仍保持正progress。run162全部first-capture population的nearest-player progress到+24为+0.5042，而run161为-0.4722。run162 80k formal capture 204、strict later-distinct 16、training win 23；run161同长度为169、3、3。run162的16个later-distinct interval全部<=24，median=12。该链支持“限制每次post-capture explore的协调破坏范围”是有效机制，而非简单提高greedy概率。
- run162仍有54/141个first-capture episode以remaining-prey nearest distance>=8开局且该band strict distinct为0；但bounded机制只在post-capture触发，现有日志缺少足以安全修改pre-capture Q/factor/action的历史序列，故该残余不能用oracle位置、hard-coded slot、继续调floor或其他无证据机制处理。

### 3.32 run127--run146 transaction正确性链与阻断错误

- run127--run146延续run126后的strict-pair exact transaction、nonlinear backtracking、origin preservation、pending pair target和诊断版本演进。完整summary存在的run为127--129、132--134、136--138、141--146；run130、131、135、139、140无summary，不能作为完整性能实验。
- 历史阻断包括：`transaction_origin_preservation_invalid`，其根因是scale-zero参数来自完整atomic snapshot而boundary/candidate/lifecycle基线来自更早独立forward；修复为Adam前atomic snapshot后统一exact origin replay，并在scale=0时直接`copy_(before)`，避免`0*Inf=NaN`。后续production还暴露`standard_adam pair transaction did not improve every exact target`、两个capture event竞争同一strict pair transition、pending pair-only zero adjacency gradient及投影constraints infeasible；这些均按fail-loud处理，失败run不进入性能结论。
- pair/capture映射的正确语义是：strict pair transition只连接nearest future capture step上的唯一canonical event；同一target transition不能被不同capture event重复消费；zero aggregate/zero adjacency gradient不能伪装成已消费成功事务。后续完整run能继续训练，说明阻断路径得到修正；但各中间patch的逐文件差异没有在本次整理中完整重建，因此只保留错误、契约与完成状态，不补造单机制性能归因。

## 4. 实验有效性判断标准

- 每个 run 必须明确：seed、目标训练步数、实际 `num_env_steps`、直接对照 run、是否恢复 checkpoint/optimizer/replay、完整配置、源码状态或 manifest、console/stderr 是否有 Traceback/ERROR/NaN/Inf。
- 评估必须检查多个时间点而非只看final。具体评估点和总步数由本轮机制触发时间、历史异常区间与稳定性目标决定；不同长度优先比较相同step和相同预算下的指标。
- reward 判断至少同时查看 final、last5、best；best 只能表明曾达到某峰值，不能代表后期稳定性。
- 成功行为同时检查 `eval_win_rate`、`capture_events`、`first_success_step`。capture 增加但 win_rate 不增加、或 reward 上升但二者不一致，均不能直接称为有效协同提升。
- 动态图检查至少包括 `eval_adj_mean_order`、order2/order3 ratio、rollout/eval order3 gap、factor identity retention（若日志存在）、connected graph ratio、coverage、uncovered active agent ratio、invalid factor ratio。
- 训练机制检查至少包括 raw/weighted order2/order3 factor loss、`raw_o3_minus_o2_factor_rl_loss`、credit gate/EMA、positive/promoted fraction、entropy、adj/policy/critic loss、`q_target_mean`、`q_tot_mean`、graph/factor clamp、PPO epoch/early stop、stale/recent sample 统计（若启用）。
- CSV、TensorBoard、console、保存 config 应交叉验证；某字段未记录时应写“日志不足以验证”，不能以其他指标替代。
- 对 run56 以后的机制实验，还必须核对落盘 definition/support/loss/normalization/projection/guard/state-sync/lifecycle version、attribution 字符串和源码 manifest SHA256；用户描述与 console 不一致时，以实际落盘配置、console 和 manifest 为准，并保留修订说明。
- preflight 必须确认实际执行而不是仅有源码；同时区分“测试通过”“训练路径触发”“optimizer 中产生非零作用”和“行为性能改善”四个层次。
- identity/outcome 诊断必须检查 event count、participant 数、candidate/active exact match、互斥 unmatched 原因、event/episode 质量误差、non-target delta、positive/negative gate/loss；计数和 fraction 的分母必须按 event、episode、transition 或 target 明确标注。
- candidate/optimizer 诊断至少检查 delta→conditional probability→loss→gradient→实际参数位移→optimizer 后 probability/rank→active 的完整链路；欧氏 dot、Adam 位移 dot、真实非线性 loss 和 rollback 是不同安全层，不能互相替代。
- support/lifecycle 诊断必须以真实 episode ID、slot generation、adjacency round 和 canonical identity 区分去重、TTL、同 update epoch 复用、跨 update 消费、保持与严格 candidate→active 生命周期。
- 多变量修改不能对单一机制作因果归因；训练脚本打印参数和预检通过只能证明参数接线或预检条件，不证明机制曾改变策略或性能。
- 分析200k历史run时，可以参考final reward > 1.979、last5 > 1.163、capture events >= 3.0、win rate >= 0.5及对应图结构指标；这些只适用于同长度历史比较，不能作为所有训练的统一门槛或延长训练的固定前置条件。
- 分析2M历史run时，可以与QMIX 2M的final 6.947、last5 6.175、实际last10约5.559、win rate 0.8和capture约4.5比较；2M只是历史长度之一，长程训练可按明确问题选择其他总步数和检查点。
- 对可能改变训练分布的参数，必须确认 parser 接收、主脚本传入、保存 config/console 打印、progress/TensorBoard 记录与实际样本/credit 触发；任一环节缺失时不能以“默认应启用”代替证据。

## 5. 历史 run 总表

说明：表中“未确认”表示当前历史对话没有足够日志或配置证据；不能据此推断该 run 无效。run 编号为同一 SDDFG 实验目录的序号，基线目录各自独立。

| run | 直接对照 run | 实验目的 | 主要改动/状态 | 是否单变量 | seed | 训练步数 | 是否恢复状态 | 有效性 | 关键结果 | 最终结论 |
|---|---|---|---|---|---|---:|---|---|---|---|
| SDDFG run1--run2 | 未确认 | 初始 episode 内动态接线与 20k 冒烟 | 动态 roster、mask、动态图日志 | 未确认 | 1 | 约20k | 未确认 | 基本可用但早期性能低 | 动态事件完整；一次控制台在 18,600/20,000 时 reward -1.313、eval reward -2.155、eval win 0.1 | 接线可运行，不构成性能优势证据 |
| SDDFG run3--run4 | 前序 run | 2M 长程迭代 | 具体代码版本未确认 | 否，未确认 | 1 | 2M | 未确认 | 日志细节未保存于本次对话 | 未确认 | 不可作精确比较 |
| SDDFG run5 | 前序 run | 20k 冒烟 | 具体版本未确认 | 未确认 | 1 | 20k | 未确认 | 未确认 | 未提供指标 | 仅知已运行 |
| SDDFG run6 | 前序 run | 200k 验证 | 具体版本未确认 | 未确认 | 1 | 200k | 未确认 | 未确认 | 未提供指标 | 仅知已运行 |
| SDDFG run7 | 前序 run | 2M 验证 | 改进后版本 | 未确认 | 1 | 2M | 未确认 | 未确认 | 未提供完整指标 | 仅知已运行 |
| SDDFG run8--run17 | 各自前序 run | 连续 20k/200k 机制排查 | 多轮图/credit/训练逻辑改动 | 多变量风险高 | 1 | 20k或200k | 未确认 | 多数未形成稳定领先证据 | 仅部分日志结论被后续 run18--30 覆盖 | 保留过程，精确数值未确认 |
| SDDFG run18 | 前序 run | 2M；其中 200k 前缀作为历史对照 | 旧 SDDFG 强版本 | 未确认 | 1 | 2M | 未确认 | 可用于历史对照，版本与后续不同 | 200k 前缀 final 2.713、last5 1.282、win 0.3、capture 3.0、mean order 2.733、o2 0.139、o3 0.544、gap 0.162 | 旧版本 200k 表现优于多数后续调整，但 train/eval gap 较大 |
| SDDFG run19 | 前序 run | 20k 冒烟 | 未确认 | 未确认 | 1 | 20k | 未确认 | 未确认 | 未提供指标 | 仅知已运行 |
| SDDFG run20 | 前序 run | 2M 验证 | 未确认 | 未确认 | 1 | 2M | 未确认 | 未确认 | 未提供指标 | 仅知已运行 |
| SDDFG run21 | run18 前缀 | 200k 图结构调整 | order 分布调整 | 未确认 | 1 | 200k | 未确认 | 可比较但性能不强 | mean order 2.218、o2 0.516、o3 0.167、final reward 1.014 | 过于 pair-heavy，低于 run18 |
| SDDFG run22 | run21 | 200k 图结构调整 | order 分布调整 | 未确认 | 1 | 200k | 未确认 | 可比较但未达最优 | mean order 2.168、o2 0.547、o3 0.136、final reward 1.313、last5 0.877 | pair-heavy，低于 run18 |
| SDDFG run23 | run22 | bonus/entropy 等 schedule 验证 | order3 bonus 与 sampling temperature schedule | 否，至少含多机制 | 1 | 200k | 未确认 | 部分有效 | final reward 1.692；train/eval o3 gap 约 0.166 | reward 高于 run22，但图分布差仍大 |
| SDDFG run24--run25 | run23 | schedule 后复验 | 具体记录不完整 | 未确认 | 1 | 20k/200k，未确认 | 未确认 | 信息不足 | 未提供可核实终值 | 不能归因 |
| SDDFG run26 | run23/24 | quota 与 order-aware credit | quota、order-aware credit | 否 | 1 | 200k | 未确认 | 结构过度三阶 | final o2 0.055、o3 0.628、非空 triplet 约0.92、order3 factor fraction 0.772 | 三阶数量被维持但未带来有效捕获收益 |
| SDDFG run27 | run26 | order3 band | min/max band 同时用于 sample/evaluate_prob | 近似单机制，其他状态未确认 | 1 | 200k | 未确认 | 具体终值未保存 | 旨在修正 run26 triplet-heavy | 未能仅凭现有摘录判定性能 |
| SDDFG run28 | run27 | greedy mixture / identity mismatch | 误启动2M后于200k后手动结束 | 否 | 1 | 超过200k后中止 | 未确认 | 可取200k 前缀分析，非完整2M | 精确指标未保留 | 不能当作完整长程对照 |
| SDDFG run29 | run28 | credit gate / pair reserve / quality floor | absolute credit gate 等 | 否 | 1 | 200k | 未确认 | 被后续 soft quota/relative gate 迭代替代 | 精确终值未确认 | 无稳定性能成功证据 |
| SDDFG run30 | run29 | soft quota、triplet balance、credit gate min scale | soft quota、triplet balance scoring、gate min scale 0.70 | 否 | 1 | 200k | 未确认 | 本轮较优但未达完整标准 | final 1.979、last5 1.163、best 4.684、win 0.5、capture 2.9、o3 0.442、o2 0.242、gap 0.003 | 近期最佳之一；但 o3 偏低、capture <3、后期稳定性不足 |
| SDDFG run31 | run30 | relative credit gate | use relative gate，避免 absolute gate 误杀 | 未确认 | 1 | 200k | 未确认 | 精确终值未保留 | 检查 gate 不应长期 0.70 | 未有超过 run30 的证据 |
| SDDFG run32 | run31 | synergy scorer、greedy cap、gate max delta | synergy triplet feature、greedy cap、max delta | 否 | 1 | 200k | 未确认 | 精确终值未保留 | 检查 o3 结构与收益闭环 | 未有超过 run30 的证据 |
| SDDFG run33--run34 | run32 | raw credit、positive-only、graph advantage gate | raw o2/o3 credit、negative coefficient、positive graph advantage gate | 否 | 1 | 200k | 未确认 | 记录存在，但精确数值未确认 | 机制版本多次变化 | 未有稳定性能成功证据 |
| SDDFG run35 | run34 | advantage-aware triplet scorer | credit EMA / marginal / score multiplier | 否 | 1 | 200k | 未确认 | 未超过历史较优版本的证据不足 | 后续 run37/38 仍见 scorer 作用弱 | 未证实提升收益 |
| SDDFG run36 | run35 | adj PPO high-clamp early stop 的初次检查 | guard 逻辑存在但曾未真正启用 | 未确认 | 1 | 200k | 未确认 | clamp 约 graph 0.79、factor 0.54（历史比较值） | 不能据此判定 guard 有效 | 需与真正启用的 run37 区分 |
| SDDFG run37 | run36 | 真正启用 high-clamp early stop | clip/factor clip stop=0.35、min epochs=1 | 近似单机制，其他状态未确认 | 1 | 200k | 未确认 | 部分接线有效、性能无效 | final约1.183、last5约1.057、best约2.073、win约0.1、capture约2.6、o3约0.442 | early stop 不等于性能提升，低于 run30 |
| SDDFG run38 | run37 | stale replay trust | graph/factor stale trust 权重 | 近似单机制，其他状态未确认 | 1 | 200k | 未确认 | 诊断触发，性能无效 | final约0.895、last5约0.148、best约1.361、win约0.4、capture约2.8、o3约0.442；raw/trusted clamp 分离 | trust 机械降低 trusted clamp，未改善训练 |
| SDDFG run40 | 未确认 | 动态图验证/训练前检查 | 当时的 triplet 置信度路径使用 `torch.minimum` | 未确认 | 未确认 | 未进入可分析完整训练 | 未确认 | 无效：验证阶段崩溃 | `AttributeError: module 'torch' has no attribute 'minimum'` | 旧 PyTorch 兼容性阻断，不能比较性能 |
| SDDFG run41 | 未确认 | 200k 训练结果分析请求 | 具体机制未确认 | 未确认 | 未确认 | 用户称为 200k | 未确认 | 原始指标未在当前可见对话保留 | 未确认 | 不作性能结论 |
| SDDFG run42 | 前序动态图版本 | 2M 长程验证 | recent replay、delayed/future/success/graph-return 路径的具体组合；完整单变量关系未确认 | 否，机制组合 | 1，历史描述 | 2M | 未确认 | 完整长程可分析 | final 5.918、last5 5.700、last10 6.046、best 8.636@1.82M、final win 0.4、capture 4.1 | reward 峰值高但未超过 QMIX 2M；捕获与胜率、末段稳定性未闭环 |
| SDDFG run43 | run42/前序版本，直接对照未确认 | graph-return triplet credit 200k 验证 | graph-return credit；具体完整参数和指标未在当前可见对话保留 | 否，未确认 | 未确认 | 用户称为200k | 未确认 | 信息不足 | 仅确认其后续问题定义围绕 graph-return credit | 不能作性能归因 |
| SDDFG run44 | run43 | delayed triplet credit + raw-local gate + 调低 graph-return credit | 多机制组合 | 否 | 未确认 | 用户称为200k | 未确认 | 原始结果未保留 | 仅确认后续比较中提及 run44 的 delayed active fraction 接近 0.99 且性能未闭环 | 该问题描述不是完整性能结论 |
| SDDFG run45--run47 | 各自前序 run | positive-only delayed、future-match、success-window gate 等 200k 验证 | 多轮 delayed/success 机制调整 | 否 | 未确认 | 用户称为200k | 未确认 | 原始终值未在当前可见对话保留 | run46 被描述为 final 有提升但 last5/win/capture 不稳定；其余精确指标未确认 | 不作超过 run30 的结论 |
| SDDFG run48 | 前序 run | 动态图训练验证 | 扩展 credit/batch 字段路径 | 否，未确认 | 未确认 | 约6,600后中断 | 未确认 | 无效：训练中崩溃 | `factor_training_mask` 的 400 对 6 维度不匹配 | 不能用于性能比较 |
| SDDFG run49 | 前序 run | 2M 长程验证 | delayed/future/success/graph-return、recent emergency、pair reserve 等组合；机制因果不可分 | 否 | 1，历史描述 | 2M | 未确认 | 完整长程可分析 | final 6.296、last5 5.751、last10 5.230、best 7.815@1.94M、win 0.9、capture 4.5 | 曾是已知 SDDFG 长程最高终值；后被 run68 的 final/last5/last10 超过，但 run49 win 更高 |
| SDDFG run50 | run49 后的代码版本，直接对照未确认 | capture-to-win triplet credit 200k 验证 | capture-to-win credit，含 future-match gate；其他机制仍存在 | 否 | 1，历史描述 | 200k | 未确认 | 完整200k可分析 | final 1.829、last5 0.546、last10 -0.438、best 2.796@140k、win 0.4、capture 2.6；200k capture-to-win credit为0 | 信号过稀且性能未过 run30 门槛 |
| SDDFG run51 | run50 后的代码版本，直接对照未确认 | pair/triplet complementary credit 200k 验证 | 在 capture-to-win 基础上加入 pair pursuit 等 complementary 字段 | 否 | 1，历史描述 | 200k | 未确认 | 完整200k可分析 | final -0.864、last5 -1.593、last10 -1.605、best 0.545@100k、win 0.2、capture 1.9；pair credit active fraction末值约0.943 | 性能显著退化，不能将信号大量触发视为有效 |
| SDDFG run52 | run51 | 收窄 pair credit | strict-future same-pair capture、去普通正reward/offset0/floor | 近似单机制 | 1 | 200k | 未见恢复 | 有效 | final -0.501、last5 -0.084、last10 -1.237、best 1.347@160k、win 0.1、capture 2.1；pair active均值0.74% | pair过宽被结构性修复，性能未恢复 |
| SDDFG run53 | run52 | 复验 pair/capture 链路 | 同一修复链的日志与代码调整 | 近似 | 1 | 200k | 未见恢复 | 有效 | final 0.496、last5 1.362、last10 -0.164、best 2.495@180k、win 0.2、capture 2.0；pair active0.76% | 后期均值改善但行为未过门槛 |
| SDDFG run54 | run53 | 机制复验 | capture/success事件字段与图链路修订 | 近似 | 1 | 200k | 未见恢复 | 有效 | final 1.799、last5 1.850、last10 0.296、best 4.206@120k、win 0.2、capture 2.2；pair active0.89% | reward改善未同步win/capture |
| SDDFG run55 | run54 | 真实capture事件pair credit验证 | `capture_count+episode_success_now`，拒绝shaped-return gate | 近似 | 1 | 200k | 未见恢复 | 有效 | final 2.728、last5 1.111、last10 0.026、best 2.728@200k、win 0.4、capture 3.0；pair active1.14% | final/capture改善，last5/win仍未整体过run30 |
| SDDFG run56 | run55 | centered outcome contrast | definition v2，成功/失败capture episode中心化 | 近似单机制 | 1 | 200k | 未见恢复 | 有效 | final 4.215、last5 2.005、last10 0.688、best 4.215@200k、win 0.3、capture 3.5 | reward/capture强，但win未过0.5；不能据单次结果证明数学展开正确 |
| SDDFG run57 | run56 | episode总outcome质量归一化 | definition v3，episode total distributed across capture triplets | 近似单机制 | 1 | 200k | 未见恢复 | 有效 | final 0.742、last5 1.659、last10 0.308、best 2.429@160k、win 0.3、capture 2.9 | 中心误差修复未转化为行为提升 |
| SDDFG run58 | run57 | capture participant identity | definition v4，只向identity matched triplet注入local delta | 近似单机制 | 1 | 200k | 未见恢复 | 有效对照 | final -1.053、last5 0.680、last10 0.551、best 2.657@140k、win 0.2、capture 1.4；41事件、matched0 | 全部事件为2人而代码只匹配order3，outcome信号断流 |
| SDDFG run59 | run58 | 两人capture exact order2 | definition v5 `highest_exactly_representable_capture_factors` | 是，主要代码差异 | 1 | 200k | 未见恢复 | 有效 | final -0.814、last5 0.874、last10 0.237、best 1.988@80k、win 0.3、capture 2.2；45事件、matched4(8.89%) | exact pair路径恢复，但active覆盖极低 |
| SDDFG run60 | run59 | outcome replay support v1 | 稀有正负episode补样 | 近似单机制 | 1 | 200k | 未见恢复 | 有效但机制有重复使用缺陷 | final -1.569、last5 0.000、last10 -0.334、best 1.858@100k、win 0.2、capture 2.5；40事件、active match12.5%、augmentation窗口23 | 补样进入训练，但同一episode可跨update无限复用 |
| SDDFG run61 | run60 | support v2一次性消费 | slot-generation跨update最多一次、同update epoch复用、原子补全 | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 0.278、last5 1.148、last10 0.276、best 3.140@100k、win 0.3、capture 2.5；enabled13、exhausted28 | 无限复用修复，但support快速耗尽 |
| SDDFG run62 | run61 | support v3与最终cohort中心化 | baseline基于final optimizer cohort | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 1.290、last5 1.604、last10 0.721、best 2.922@140k、win 0.2、capture 2.9；center error0、enabled7 | 零中心数学性质成立，信号仍稀疏 |
| SDDFG run63 | run62 | outcome-confidence scaling v2 | `abs_detached_graph_advantage` | 是，manifest不同 | 1 | 200k | 未见恢复 | 有效但训练效应未显现 | 与run62十个eval点完全相同：final 1.290、last5 1.604、last10 0.721、win 0.2、capture 2.9 | 暴露graph advantage replay来源/写入未生效 |
| SDDFG run64 | run63 | graph advantage replay写入修复 | scaling v3，使用ready的stored graph-return advantage | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 0.428、last5 0.972、last10 0.312、best 3.288@120k、win 0.3、capture 2.9；active match20.83% | 路径真实改变训练，但性能退化 |
| SDDFG run65 | run64 | outcome factor loss normalization v2 | target-local、按有效graph transition归一化 | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 1.754、last5 2.956、last10 1.214、best 3.990@140k、win 0.3、capture 2.4 | 局部loss尺度改善，末点和行为未达标 |
| SDDFG run66 | run65 | candidate-level identity supervision | candidate loss v1，active/candidate互斥 | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final -0.867、last5 0.817、last10 0.118、best 3.462@100k、win 0.1、capture 1.6；candidate-only84.21% | candidate监督接线成功，Bernoulli式loss语义错误 |
| SDDFG run67 | run66 | candidate loss v2 | signed log-weight | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 1.878、last5 1.043、last10 0.192、best 3.066@100k、win 0.4、capture 2.7；candidate-only76.74% | 绝对weight目标未保证conditional rank |
| SDDFG run68 | run67 | candidate loss v3长程验证 | current differentiable conditional probability；完整2M | 主要机制+长步数 | 1 | 2M | 未见恢复 | 完整有效长程 | final 6.718、last5 6.610、last10 6.527、best 7.724@1.86M、win 0.6、capture 4.6；candidate-only75.89%；前200k final0.344 | 当前SDDFG最高长程终值；奖励强但win未达QMIX，前200k不强 |
| SDDFG run69 | run68前200k/新代码 | candidate loss v4 normalization v2 | 只按target-bearing transition归一化 | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final -0.543、last5 -0.075、last10 -0.340、best 2.095@100k、win0、capture1.2；candidate/graph loss0.801%；candidate-only81.82% | 无关transition稀释被修复，optimizer后方向仍错误 |
| SDDFG run70 | run69 | candidate gradient projection v1 | 同update base/candidate欧氏冲突投影 | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final 3.555、last5 1.592、last10 0.628、best3.555@200k、win0.8、capture3.4；正/负概率正确84.38%/82.14%；candidate-only80% | 后期机制系列最强200k；即时方向改善，rank/active转化仍弱 |
| SDDFG run71 | run70 | actual update guard v1 | 检查Adam真实位移并尝试安全修正 | 是，主要差异 | 1 | 200k | 未见恢复 | 完整训练有效 | final -1.353、last5 0.667、last10 0.166、best2.214@80k、win0.3、capture2.7；candidate-only85% | 单步概率多正确但rank为0，行为退化；随后状态重建断言暴露 |
| SDDFG run72 | run71 | optimizer state sync v2 | 自动选择两类Adam重建公式并反解exp_avg | 是，主要差异 | 1 | 200k | 未见恢复 | 有效 | final1.221、last5 1.896、last100.781、best3.036@120k、win0.1、capture2.1；loss下降100%，candidate-only78.79% | 短期方向正确，rank与长期保持不足；状态数学自洽性未建立 |
| SDDFG run73 | run72 | state sync v3+lifecycle v1 | 标准Adam公式、有限no-forget缓存 | 不完全，两个相关机制 | 1 | 200k | 未见恢复 | 有效 | final1.608、last5 1.734、last100.418、best3.390@160k、win0.4、capture2.4；candidate-only89.47%；rollback32.21% | lifecycle覆盖/TTL/逐transition保护存在缺陷 |
| SDDFG run74 | run73 | lifecycle实际v3逐约束保护 | adjacency-round TTL、active-set、target/base统一约束、rollback | 近似单机制 | 1 | 200k | model_dir空，未恢复 | 严格有效 | final-0.047、last5 0.889、last10-0.029、best2.167@180k、win0.3、capture1.8；按事件计数汇总candidate-only89.57%；rollback33.78% | 欧氏约束安全但Adam实际位移违反，频繁rollback压低有效graph更新 |
| SDDFG run75 | run74 | lifecycle v4训练验证 | Adam realized-displacement逐约束投影、确定性非线性backtracking、单次最终state sync、TTL/observation archive修复 | 配置完全相同；源码为同一lifecycle修复包，含多个耦合子项 | 1 | 200k | model_dir空；未见checkpoint、optimizer或replay恢复 | 严格有效 | final1.788、last5 1.003、last10-0.130、best1.947@120k、win0.5、capture2.8、o2=0.248、o3=0.435；candidate-only93.07%；rollback0 | v4修复run74的实际位移/rollback问题，但未达200k门槛且后期波动大；另暴露target-bearing诊断漏记 |
| SDDFG run76--run80 | 未确认 | 本次可见对话未保留完整记录 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 | 仅知run78是后续200k强比较值 | 不补造未提供的run细节 |
| SDDFG run78 | 直接对照未确认 | 200k强比较值 | 具体机制未在当前对话完整保留 | 未确认 | 1（历史比较表） | 200k | 未确认 | 指标被后续反复引用 | reward3.860、win0.7、capture3.7、matched7.17%、positive rank改善5.55%、topology retention0.690、Q-grad P95=1.89 | 当前后期系列必须同时超过的200k行为基线；不应归因到未保存的具体代码 |
| SDDFG run81 | 前序run未确认 | candidate v10前后的历史对照 | 具体改动未完整保留 | 未确认 | 未确认 | 240k | 未确认 | 部分指标仅来自用户比较要求 | 200k reward-0.720、220k 2.880、240k 1.305、last5 1.099、active matched18、candidate-only88.42%、base-gradient removed7.55%、crossing0 | 指标显示candidate/active闭环仍弱；完整有效性与配置未确认 |
| SDDFG run82 | run81 | candidate v11、PPO early-stop后的有限 residual | v10 first-reachable signed hinge、normalization v3、lifecycle v9；严格skip satisfied、无跨update replay/TTL刷新 | 未确认 | 未确认 | 240k（用户描述） | 未确认 | 完整结果未保留 | 未提供最终性能数值 | 只能记录机制目标，不能认定残差有效 |
| SDDFG run83 | run82 | candidate v12与独立 residual Adam | `.grad=None`隔离、inactive零位移断言、独立optimizer state | 相关修复包 | 未确认 | 260k | 未确认 | 训练后来完成，但本对话还保留早期测试/训练阻断错误 | 无完整性能数值 | 实现曾报KeyError与UnboundLocalError；性能结论未确认 |
| SDDFG run84 | run83 | lifecycle v10、terminal save、双Adam checkpoint | behavior-progress-gated lifecycle、final terminal policy、双Adam持久化 | 相关修复包 | 未确认 | 260k | 未确认 | 用户称完整完成；逐产物审计结果未保留 | 未提供最终性能数值 | 代码启用/性能作用未确认 |
| SDDFG run85 | run84 | 事务式 checkpoint 与 lifecycle v10复验 | periodic/best/terminal completion metadata、fail-loud restore | 相关修复包 | 未确认 | 260k | 未确认 | 用户称完整完成；逐产物审计结果未保留 | 未提供最终性能数值 | checkpoint是复现修复，非性能归因 |
| SDDFG run86 | run85 | trusted-control与actual early-stop修复 | trusted selector应驱动early-stop/recent-window；raw作诊断 | 相关修复包 | 未确认 | 280k | 未确认 | 用户称完整完成 | 200k0.974、240k3.487、260k-0.648、280k1.042、last5 1.114、last10 1.330、best3.487@240k、final win0.2、capture2.5 | 有240k高点但260k同步回撤，未建立稳定收益 |
| SDDFG run87 | run86 | trusted-control运行时主链与2M验证 | 同一selected ratio驱动early-stop和下一次window、runtime contract | 相关修复包 | 1 | 2M | 未确认 | 用户称完整完成；后续摘要给出窗口统计 | final5.628、last5 6.221、last10 6.348、best8.076@1.88M、win0.4、capture4.0；200k-0.764、400k1.596 | 长程窗口恢复但未稳定优于run68；不可仅凭best归因 |
| SDDFG run88 | 未确认 | 本次可见对话未提供记录 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 | 不补造 |
| SDDFG run89 | run87前400k | population-total trusted control | `sum(n)/sum(d)`、loss/control人口统一、generation/chunk执行契约 | 相关修复包 | 未确认 | 400k（实际需以产物核验） | 未确认 | 用户称完成；完整审计结果未保留 | 200k1.080、400k1.238、last5 1.330、last10 1.298、best2.400@260k、final win0.1、capture2.2 | 聚合语义修复改变轨迹，但未超过强基线 |
| SDDFG run90 | run89 | 600k性能与capture质量审计 | run89后控制/采样修复代码 | 非单变量历史链 | 未确认 | 600k | 未确认 | 用户称完成 | 200k1.080、win0.4、capture3.1；best4.320@580k；final0.768、last5 2.437、last10 1.948、win0.2、capture2.4；200k失败episode capture43.8%、capture episode win44.4% | capture-to-win质量显著弱于run70/run78，长期不稳定 |
| SDDFG run91 | run90 | outcome-conditioned signed pair credit | success/failure capture episode centered signed pair credit | 主要机制，完整差异未逐项核验 | 未确认 | 400k | 未确认 | 用户称完成 | 200k-0.188、win0.3、capture2.0；best3.608@160k、400k0.400、last5 0.864、last10 0.556、final win0.5、capture2.7 | signed pair未形成稳定正效应，capture下降且行为未达基线 |
| SDDFG run92 | run91 | identity-local pair PPO | pair credit退出shared `f_advt`，仅target factor local PPO | 主要机制 | 未确认 | 400k | 未确认 | 用户称完成 | 80k3.529、200k-0.418、win0.1、capture2.2、400k1.984、last5 1.426、last10 1.193、final win0.4、capture2.9 | 修复显式广播污染，但高点未保持且200k退化 |
| SDDFG run93 | run92 | pair-local normalization v2 | 分母改为pair-target-bearing transition | 主要机制 | 未确认 | 400k | 未确认 | 用户称完成 | 80k5.331、200k1.759、win0.4、capture2.8、400k1.025、last5 0.811、last10 0.569、final win0.3、capture2.5 | v2放大稀疏信号；后审计发现one-sided optimizer cohort导致错误放大 |
| SDDFG run94 | run93 | final optimizer cohort重新中心化 | one-sided pair loss=0；transaction cohort重新生成credit | 主要机制 | 未确认 | 400k | 未确认 | 用户称完成 | 160k4.345、200k2.786、win0.4、capture2.5、400k1.267、last5 1.382、last10 1.428 | 消除了one-sided非零pair损失，但监督变稀疏且未稳定过基线 |
| SDDFG run95 | run94 | pair-evidence support v4与原子transaction | exact evidence class-complete support、mixed pair atomic transaction、step mass守恒 | 主要机制 | 1 | 400k | fresh；无恢复（console/产物审计） | 严格有效 | best4.097@120k、200k2.116/0.3/2.3、final0.104/0.2/2.1、last5-0.024、last10 0.495；42个非零pair transaction均守恒 | 守恒和support接线正确但性能更差；发现原子partition改变base PPO batch人口，当前v5代码已修复但未训练验证 |
| SDDFG run96 | run95 | support v5批人口修复的400k训练验证 | class-complete pair partition filler/公平slot；运行时support version=5 | 主要机制，完整源码差异未逐项复核 | 1 | 400k | fresh；`model_dir`空 | 有效 | final0.886、last5 1.091、last10 1.477、best3.565@260k、win0.1、capture2.1 | v5可训练但没有恢复强200k/400k行为表现 |
| SDDFG run97 | run96 | support v6 full-population transaction | class-complete时单一完整selected population、一个standard Adam transaction | 主要机制 | 1 | 400k | fresh；`model_dir`空 | 有效 | final1.481、last5 1.069、last10 1.576、best3.262@220k、win0.5、capture3.0；runtime support version=6 | 修复transaction人口语义；性能仍低于历史强200k基线，不能称为性能成功 |
| SDDFG run98 | run97前缀/同support v6代码 | 120k机制基线 | support v6；具体新增诊断目的未确认 | 未确认 | 1 | 120k | fresh；`model_dir`空 | 有效 | final@120k0.862、last5 0.544、best2.324@60k、win0.2、capture2.7 | 作为run99/run100轨迹对照；无独立性能结论 |
| SDDFG run99 | run98 | pair gradient聚合诊断v1 | 增加pair gradient/位移/score聚合诊断，不改训练 | 是，诊断 | 1 | 120k | fresh；`model_dir`空 | 有效 | 与run98六个eval点一致；后续run100复现六个真实pair epoch | 诊断轨迹中性；两epoch聚合不足以定位方向翻转阶段 |
| SDDFG run100 | run99 | per-epoch optimizer transaction diagnostics v2 | 新增84字段`progress_train_adj_transaction.csv` | 是，诊断 | 1 | 120k | fresh；`model_dir`空 | 严格有效 | 与run99公共轨迹一致；77.6k/0、89.6k/0 combined反向，两个epoch1 combined正确但Adam反向，108k两epoch全链正确 | 首次把最早断点区分到combined、Adam、final与score；诊断本身无性能作用 |
| SDDFG run101 | 无有效对照 | diagnostics v3首次启动 | `r_sddfg.py`与`base_runner.py`版本不同步 | 否，版本错配 | 1 | 目标110k，实际早期退出 | fresh；不得恢复 | 无效 | `RuntimeError: unexpected pair optimizer diagnostic version`；无完整eval/transaction CSV | 仅是代码同步错误，不得用于任何性能或根因分析 |
| SDDFG run102 | run100截至110k | per-objective gradient diagnostics v3 | graph/base/outcome/pair/candidate/entropy独立gradient与重构 | 是，诊断 | 1 | 110k | fresh；`model_dir`空 | 严格有效 | 公共v2轨迹与run100一致；100k reward-0.235、win0.5、capture3.1；graph为77.6k/0和89.6k/0最大负投影 | v3轨迹中性；源码确认graph advantage source责任范围错误 |
| SDDFG run103 | run102 | graph advantage source v2 | graph仅用独立replay graph-return，factor residual重构原`f_advts` | 是，训练逻辑 | 1 | 110k | fresh；`model_dir`空 | 有效 | 20k--100k reward`-5.187,-3.395,1.340,0.977,3.269`；100k win0.5/capture3.4；positive pair evidence=0、class-complete=0、pair transaction=0 | source隔离真实改变轨迹并修复污染；110k性能只在100k单点较好，pair监督断点前移 |
| SDDFG run104 | run103 | pair-evidence funnel v1 | update级successful/capture/active/candidate/pair evidence漏斗 | 是，诊断 | 1 | 110k | fresh；`model_dir`空 | 有效 | 与run103轨迹一致；4个successful capture全部candidate-only，active successful=0、positive pair=0 | funnel轨迹中性；最早断点定位为successful candidate未转active |
| SDDFG run105 | run104 | funnel v2与episode reject/boundary join | episode-level CSV、reject reason、candidate rank/margin/boundary | 是，诊断 | 1 | 110k | fresh；`model_dir`空 | 有效 | 与run104轨迹一致；reject集中`CANDIDATE_ONLY_NOT_ACTIVE`；generation370含`0-4`,`2-5`正target | 排除provenance/terminal漏失，断点细化为score→rank→active |
| SDDFG run106 | run105 | per-target same-population rank transaction v1 | 每target记录rank分解、next-better gap和boundary | 是，诊断 | 1 | 110k | fresh；`model_dir`空 | 有效 | 与run105轨迹一致；4条successful-overlap均margin正确但rank不变；实际改善仅占next-gap约0.085%--0.494% | rank不变由更新远小于合法gap解释；未发现rank/cache/population实现错误 |
| SDDFG run107 | run106前110k | provenance-complete与200k观测 | event级candidate evidence CSV、独立generation反事实；训练算法不变 | 是，诊断 | 1 | 200k | fresh；`model_dir`空 | 严格有效 | final2.177、last5 1.035、last10 0.218、best3.269@100k、win0.3、capture2.9；strict pair positive/negative generation=2/36，class-complete=0 | provenance轨迹中性；正负strict evidence因recent replay时窗不重叠而无法启动pair监督 |
| SDDFG run108 | run107 | pair-specific immutable bounded pending | `pair_bounded_pending_evidence=true`、TTL=4、pair-only outer atomic transaction | 是，训练变量 | 1 | 200k | fresh；`model_dir`空 | 有效，但机制事务未净提交 | 与run107所有公共轨迹一致；144.8k形成`660(-)+689(+)`，epoch0 pair loss0.00838/grad0.002852后被通用early-stop中止并完整rollback，commit=0 | pending补齐时窗但被错误control scope阻断；run108不能证明pending性能效果 |
| SDDFG run109 | run108后修复状态 | 验证pair/non-pair base-factor人口诊断 | diagnostics v3；True/4 | 相关正确性修复 | 1 | 目标200k；实际约164.8k | fresh；`model_dir`空 | 部分有效，中断 | 最后完整eval160k reward0.926/win0.2/capture3.0；164.8k报`pair/non-pair base-factor population split failed to reconstruct the full transaction` | 暴露人口分解诊断/事务接线错误；不得当200k终点 |
| SDDFG run110 | run109 | 修复人口分解后完整验证 | diagnostics v3；True/4 | 相关正确性修复 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | final reward-1.309、win0.2、capture2.1；terminal checkpoint | 阻断性人口分解错误未再发生；性能弱，完整机制版本细节未在当前对话独立归因 |
| SDDFG run111 | 未确认 | 历史编号/口头误用 | 无对应run目录 | 未确认 | 未确认 | 未确认 | 未确认 | 无可验证实验 | 当前结果根目录不存在run111 | 不得凭编号编造实验或指标 |
| SDDFG run112 | run110/后续pending-on链 | pending-off完整对照 | pending=False/TTL=0；diagnostics v4 | 主要为单变量对照 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | final2.177、last5 1.035、last10 0.218、win0.3、capture2.9；全程failed-capture55.51%、matched15.81%、capture-episode-win28.72% | pending-off基线；后续pending机制没有整体超过该对照 |
| SDDFG run113 | run112/同阶段代码 | True/4中间机制验证 | diagnostics v4 | 未确认 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | final-0.875、win0.2、capture2.5；terminal checkpoint | 中间版本可运行，具体单一机制因果未确认 |
| SDDFG run114 | run113 | 事务诊断/保护链中间版本 | diagnostics v5；True/4 | 未确认 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | final-0.504、win0.2、capture2.3；terminal checkpoint | 中间版本性能弱；详细版本语义证据不足 |
| SDDFG run115 | run114 | v7逐target ordinary与pending过滤 | forced/non-actionable过滤、ordinary逐target保护；True/4 | 主要机制 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | ordinary逐target46/46正确；163.2k pending multi-target为20 correct/20 reverse但aggregate正；final0.239、last5 0.528、win0.3、capture2.6 | aggregate pair正确掩盖pending逐target反向；促成逐target pending保护 |
| SDDFG run116 | run115 | v11逐target联合事务 | pending/ordinary逐target、current-priority、lifecycle筛选与exact revalidation | 相关事务修复包 | 1 | 200k | fresh；`model_dir`空 | 最终正式attempt完整；此前存在未落盘失败attempt | 44/44 correct、reverse0、zero0、rollback0；final1.036、last5 0.298、last10 -0.151、win0.5、capture2.8；全程failed55.89%、matched16.73%、capture-episode-win30.11% | exact事务闭合；selection-boundary/rank/active仍未验证 |
| SDDFG run117 | run116 | v12 production preflight与统一事务正确性对照 | step=0独立preflight、统一容差、真实transaction replay；True/4 | 正确性/诊断修复 | 1 | 200k | fresh；`model_dir`空 | 完整有效 | 与run116正式轨迹一致；44/44 correct、reverse0、zero0；final1.036、last5 0.298、last10 -0.151、win0.5、capture2.8 | v12正确性闭合；未记录真实boundary/rank/active |
| SDDFG run118 | run117 | v13 production selection-boundary | pair-specific competitor、margin、rank、active诊断 | 主要机制 | 1 | 目标200k；实际约182.4k | fresh；`model_dir`空 | 部分有效，中断 | 26/26 exact、26/26 margin改善，median delta0.008966；crossing/promotion/eviction=0；最后完整eval180k reward0.157/win0.4/capture2.9 | 闭合score→margin；联合candidate/lifecycle复核缺口导致中断，非正式200k结果 |
| SDDFG run119 | run118后v14 | 20k no-target与联合回溯冒烟 | boundary/candidate/lifecycle联合exact及rollback；v14 | 正确性阶段 | 1 | 20k | fresh；`model_dir`空 | 完整有效 | 20k reward-5.187/win0.1/capture0.8；无actionable boundary target；与run117前20k一致 | no-target、RNG、preflight与schema通过；不构成boundary性能验证 |
| SDDFG run120 | run118/run119 | v15 deficit-aware预算 | 回收zero-deficit大额预算、deficit-bearing分配 | 主要机制 | 1 | 160k | fresh；`model_dir`空 | 完整有效 | required完成率约99.8%、median deficit reduction0.813%；crossing/promotion/eviction=0；160k reward0.475/win0.3/capture2.8 | 预算利用改善但未形成rank crossing |
| SDDFG run121 | run120 | v16最近boundary water-filling | exposure级预算集中，总预算不变 | 主要机制 | 1 | 160k | fresh；`model_dir`空 | 完整有效 | 最近exposure获约92%--99%预算；150.4k scale0.5/0.03125，post deficit0.225422；median reduction0.933%；crossing0；160k reward-3.073/win0.3/capture2.4 | 同identity其他exposure仍限制联合位移 |
| SDDFG run122 | run121 | v17 identity-group sum | group Jacobian与actual均按member sum | 主要机制 | 1 | 目标160k；149.6k中断 | fresh；`model_dir`空 | 部分有效，中断 | group completion约99.96%、nearest member约28.9%；optimizer.step后联合exact无安全scale，atomic rollback；最后完整eval140k reward1.686/win0.5/capture3.3 | aggregate masking被真实证伪；不得写160k终点 |
| SDDFG run123 | run122 | v18 nearest progress-member | 每group唯一progress member；其他member non-regression | 主要机制 | 1 | 160k | fresh；`model_dir`空 | 完整有效 | seq720/721 completion50.26%/3.12%，scale0.5/0.03125；crossing0；160k reward-3.073/win0.3/capture2.4 | aggregate口径修复，固定方向缩步仍严重 |
| SDDFG run124 | run123 | v19最大固定方向安全scale | halving bracket+12次production refinement | 主要机制 | 1 | 160k | fresh；`model_dir`空 | 完整有效 | seq720 scale0.691650/completion69.40%，seq721 0.024597/2.46%；post deficit0.212795；crossing0；160k reward1.603/win0.4/capture3.1 | 固定方向搜索精度部分闭合，方向可行域仍限制progress |
| SDDFG run125 | run124 | v20同预算多Adam方向 | 五个progress fraction、独立safe search、original-required排序 | 主要机制 | 1 | 160k | fresh；`model_dir`空 | 完整有效 | seq721选0.25、scale1、actual0.032033、completion24.63%；post deficit0.183960；crossing0；160k reward0.743/win0.2/capture2.7 | 多方向提高关键actual但仍未crossing；schema v6未落全部候选几何 |
| SDDFG run126 | run125 | v21八候选与全候选诊断 | 五Adam+三progress-directed；boundary v7/candidate v1 | 主要机制+诊断 | 1 | 160k | fresh；`model_dir`空 | 严格完整有效 | 8候选真实生成；34/34 exact/boundary正确；median deficit reduction2.1198%；crossing/promotion/eviction=0；160k reward0.391/win0.2/capture2.5 | 候选几何和选择闭合，rank crossing仍未闭合；审计暴露zero-budget identity污染progress seed |
| SDDFG run127--run129 | run126后transaction链 | v22后strict-pair/transaction迭代 | 具体逐run单变量差异未完整重建 | 否/未确认 | 1 | 各160k | fresh，summary存在 | 完整 | final reward/win/capture依次为0.391/0.2/2.5、0.004/0.1/2.0、0.004/0.1/2.0 | 保留完成事实；不能从相近编号推断具体修复收益 |
| SDDFG run130--run131 | 前序run | transaction错误排查 | 无完整summary | 未确认 | 1 | 未确认 | 未确认 | 不完整/证据不足 | 无可核实正式终点 | 不用于性能比较 |
| SDDFG run132--run134 | 前序transaction链 | origin/exact transaction演进 | 具体逐run差异未完整重建 | 否/未确认 | 1 | 160k、160k、146.4k | fresh，summary存在 | 完整至各自target | final reward/win/capture依次为-0.763/0.2/2.4、-1.120/0.2/2.1、1.686/0.5/3.3 | run134行为较好，但非已证明单变量收益 |
| SDDFG run135 | 前序run | transaction错误排查 | 无summary | 未确认 | 1 | 未确认 | 未确认 | 不完整/证据不足 | 无正式终点 | 不用于性能比较 |
| SDDFG run136--run138 | 前序transaction链 | 20k smoke与160k复验 | strict-pair/exact正确性链；具体差异未完整重建 | 否/未确认 | 1 | 20k、160k、160k | fresh，summary存在 | 完整 | run136 -0.971/0/1.3；run137与138均2.793/0.3/3.1 | run137/138可重复终点，但机制归因信息不足 |
| SDDFG run139--run140 | 前序run | production failure fixture来源 | 无summary；保留失败fixture | 未确认 | 1 | 未确认 | 未确认 | 非正式完整run | run140 fixture后来仍以scale0.0009765625满足正boundary与actual progress | 只作transaction回放证据 |
| SDDFG run141--run146 | 前序transaction链 | approximately-zero exact、origin与joint exploration前置演进 | 具体逐run差异未完整重建 | 否/未确认 | 1 | 180k、20k、40k、60k、20k、60k | fresh，summary存在 | 完整 | final reward/win/capture：1.548/.2/2.6；-.971/0/1.3；-1.640/.3/2.2；.185/.2/2.0；-2.528/.2/1.7；.694/.4/3.2 | 正确性链可运行；不同预算与多版本不能直接作单变量性能结论 |
| SDDFG run147 | 后续run148/149基线 | joint epsilon 20k smoke | fresh、228k joint exploration、旧33/57输入 | 主要探索语义 | 1 | 20k | fresh | 完整 | final eval -3.251/win0.1/capture0.7 | 作为同seed短程探索基线 |
| SDDFG run148 | run147 | joint epsilon 60k正式行为基线 | 228k joint exploration、旧33/57输入 | 长度扩展 | 1 | 60k | fresh | 完整 | formal episodes268、capture episodes58、captures67、exact events6、training win1；eval -1.188/.2/1.8 | basic capture提高，但multi-prey no-win interval median132且0/7<=24，state aliasing成为首个代码问题 |
| SDDFG run149 | run147同长度 | freeze countdown 20k smoke | local/state 35/59；offscreen countdown可见 | 主要observation语义 | 1 | 20k | fresh | 完整 | formal68、capture episodes9、captures12、exact events4、win0；eval -1.942/.1/1.0 | env→runner→replay→RNN/Q输入链闭合，capture未坍缩；20k不足判断win |
| SDDFG run150--run151 | run148 | countdown 60k与transaction-origin修复后复验 | 35/59、228k joint exploration、无floor；run151统一transaction origin | 主要输入语义+正确性修复 | 1 | 各60k | fresh | 完整 | 两者formal268、capture episodes89、captures102、exact events19、strict distinct2、wins2；eval -0.110/.1/2.3 | countdown有functional Q/action sensitivity；首个未闭合层后移到post-capture exposure |
| SDDFG run152 | run149 | post-capture greedy floor 20k smoke | floor0.25、exploration contract v3 | 主要探索局部语义 | 1 | 20k | fresh | 完整 | eligible/floor-active220/220，greedy52=23.636%；formal68、capture episodes8、captures10、win0；eval -2.438/0/.5 | timing/lifecycle/RNG/action contract闭合；短程行为样本不足 |
| SDDFG run153 | run151 | floor0.25的60k正式验证 | 其余保持countdown与228k schedule | 主要局部探索语义 | 1 | 60k | fresh | 完整 | formal268、first81、strict distinct4、wins4、interval median7；eval .225/.1/1.1 | first→distinct由2.25%升至4.94%，floor方向有早期正证据 |
| SDDFG run154--run155 | run153后transaction链 | standard_adam exact-target报错排查 | 训练中fail-loud | 正确性修复 | 1 | 未完成 | fresh | 无效 | `standard_adam pair transaction did not improve every exact target` | 不用于性能比较；错误保留为transaction契约证据 |
| SDDFG run156 | run153 | floor+countdown长程验证 | one-step TD、floor0.25 | 长度扩展 | 1 | 180k | fresh | 完整 | formal868、capture episodes550、captures854、strict distinct27、wins38；eval1.943/.3/2.8 | floor收益在长程存在但first→distinct约4.91%；成功最早分叉在first-capture双目标几何 |
| SDDFG run157 | run156前60k | unconditional 24-step TD | 所有transition累积24步reward | 主要Q-target语义 | 1 | 60k | fresh | 完整但性能失败 | formal268、capture episodes12、captures13、wins0；eval-.809/0/.1；target variance/gradient与clip异常 | 无条件n-step放大dense reward并摧毁basic capture，机制被证伪 |
| SDDFG run158 | run157 | terminal-gated 24-step | 无win marker严格one-step；marker前24步multi-step | 主要Q-target修复 | 1 | 80k | fresh | 完整 | formal368、capture episodes142、captures180、strict distinct4、wins5；eval.503/.2/2.3 | 数值和capture恢复；completion credit正确但仅168 gated transitions，利用稀疏 |
| SDDFG run159 | run158 | terminal replay lane | forced episode frequency增加、aux weight1.0 | 主要replay利用 | 1 | 80k | fresh | 完整但性能退化 | formal368、capture episodes119、captures145、strict distinct1、wins1；eval-1.140/0/1.0 | credit量约增6.57倍但aux population支配uniform objective，Q/RNN与行为退化 |
| SDDFG run160 | run159前60k | weighted lane | forced aux weight0.10 | 主要loss population修复 | 1 | 60k | fresh | 完整 | formal268、capture episodes85、captures102、strict distinct2、wins2；eval.795/.2/1.9 | Q/capture恢复，geometry与+4/+8 progress改善；需验证更长post-trigger窗口 |
| SDDFG run161 | run160 | weighted lane 80k长度验证 | 与run160同语义 | 长度扩展 | 1 | 80k | fresh | 完整 | formal368、capture episodes141、captures169、strict distinct3、wins3；eval.584/.1/1.6 | loss population长期稳定；+8到+24 persistence成为首个未闭合行为层 |
| SDDFG run162 | run161 | bounded post-capture exploration | eligible explore最多随机1个alive slot；contract v4 | 主要explore branch语义 | 1 | 80k | fresh | 严格完整 | formal368、capture episodes141、captures204、strict later-distinct16、wins23；interval median12且16/16<=24；eval1.588/.3/2.7 | Hamming与destructive retreat下降，persistence、distinct和win显著改善；单seed80k尚非最终论文优势 |
| VDN run2 | 同环境基线 | 200k baseline | 动态掩码适配 | 基线配置，细节未确认 | 1 | 200k | 未确认 | 可作基线参考 | final reward约-3.269、last5约-2.970、win约0.4 | reward弱，不能作为动态优势上界 |
| QPLEX run1 | 同环境基线 | 200k baseline | 动态掩码适配 | 基线配置，细节未确认 | 1 | 200k | 未确认 | 可作基线参考 | final reward约-1.487、last5约-3.231、win约0.1 | reward弱 |
| QMIX run1 | 同环境基线 | 20k 冒烟 | 初始动态适配 | 未确认 | 1 | 20k | 未确认 | 初次运行后有兼容性报错 | `torch.nan_to_num` 不存在 | 修复兼容性后重跑 |
| QMIX run2 | QMIX run1 | 200k baseline | 兼容性修复后 | 未确认 | 1 | 200k | 未确认 | 可作基线参考 | final reward 1.524、last5 -0.913、best 1.608、win 0.1 | 200k非强但高于VDN/QPLEX reward |
| QMIX run3 | QMIX run2 | 2M baseline | 修复后长程 | 未确认 | 1 | 2M | 未确认 | 强长程基线 | final reward 6.947、last5 6.175、实际last10约5.559、win 0.8 | 当前明确的长程基线参考 |
| QPLEX 2M（run号未确认） | QPLEX run1 | 2M baseline | 动态配置长程 | 未确认 | 1，历史目录描述 | 2M | 未确认 | 有效性存在历史争议 | final8.932、last5 6.661、last10 6.400、win0.9 | 结果文件存在但缺少历史排除依据；保留冲突，不用于确认优势 |

## 6. 重要 run 详细记录

### 初始动态 20k（SDDFG run1--run2，具体序号未确认）

#### 实验目的

- 验证 episode 内 shock、join、recover、mask 与动态图统计是否贯通训练和 eval。

#### 训练完整性与关键日志

- 一次控制台在 18,600/20,000 steps 时记录：`num_players_mean=4.46`、最小 2、最大 6、`join_events=2`、`leave_events=4`、`recover_events=4`、`roster_change_events=6`、`pending_recovery_final=0`、`recovery_completion_rate=1.0`、`activation_state_resets=6`、`active_ratio_mean≈0.743`。
- 同一记录的图指标：`adj_valid_factor_ratio≈0.908`、`adj_empty_factor_ratio≈0.092`、`adj_invalid_factor_ratio=0`、训练 `adj_mean_order≈2.080`；eval `adj_mean_order≈2.038`、eval order2 ratio≈0.883、eval order3 ratio≈0.025。
- 性能：训练 reward≈-1.313，eval reward≈-2.155，eval win rate=0.1。

#### 有效性判断与结论

- 动态人数轨迹、加入/退出/恢复和有效图因子被日志直接证实。
- 该短程结果不支持性能优势；其用途限于运行、生命周期和掩码接线验证。
- 来源与证据：本次历史对话中的控制台输出摘录。

### run18（旧 SDDFG）

#### 实验目的与状态

- 2M 长程训练，后续反复用其 200k 前缀作为旧 SDDFG 强对照。

#### 关键日志与指标

- 200k 前缀：final reward=2.713、last5=1.282、win rate=0.3、capture events=3.0、mean order=2.733、order2 ratio=0.139、order3 ratio=0.544、train/eval order3 gap=0.162。

#### 结论

- 该前缀同时具备较高三阶比例和相对较高 reward，但 train/eval 图分布差较大，不能据此证明三阶结构或某个后续改动是因果来源。
- 来源：本次历史对话中的跨 run 比较表。

### run21 与 run22

#### 实验目的

- 调整动态图阶数分布，降低旧版本的结构问题。

#### 关键日志与比较

- run21：mean order=2.218、order2=0.516、order3=0.167、final reward=1.014。
- run22：mean order=2.168、order2=0.547、order3=0.136、final reward=1.313、last5=0.877。
- 二者均比 run18 更 pair-heavy，且 reward 低于 run18。

#### 结论

- 将图推向 pair-heavy 没有显示出超过 run18 的性能；结构比例变化本身不足以解释性能。
- 来源：本次历史对话中的已知比较指标。

### run23

#### 实验目的

- 检查 order3 bonus、entropy/sampling temperature schedule 等机制。

#### 关键日志与比较

- final reward=1.692，高于 run22 的 1.313；train/eval order3 gap 约 0.166。

#### 结论

- 相对 run22 有 reward 改善，但较大的 train/eval 图结构差仍存在；不能证明 schedule 已形成稳定性能闭环。
- 来源：本次历史对话中的 run23 复盘要求及比较值。

### run26

#### 实验目的

- 验证 quota 与 order-aware credit 是否能维持高阶协同。

#### 关键日志与指标

- final order2 ratio=0.055、order3 ratio=0.628；非空 factor 中 triplet 占比约 0.92；`order3_factor_fraction=0.772`。

#### 结论

- quota/order-aware credit 确实改变结构并推高三阶比例，但未带来有效捕获收益。该结果反驳“维持更多 triplet 即可提高性能”的解释。
- 来源：本次历史对话中的 run26 问题定义与结论摘录。

### run30

#### 实验目的

- 验证 soft quota、triplet balance scoring 与提高后的 credit gate min scale。

#### 关键日志与指标

- final reward=1.979、last5=1.163、best reward=4.684、final win rate=0.5、capture events=2.9。
- final order3 ratio=0.442、order2 ratio=0.242、train/eval order3 gap=0.003。

#### 有效性判断

- 相对本轮多数改动版本，run30 是已记录的近期较优结果：reward、win rate、图 train/eval gap 有改善。
- 但 capture events 未达到 3.0；order3 ratio 低于阶段性 0.50--0.58 目标；last5 明显低于 best，说明后期稳定性不足。

#### 最终结论

- 不能据此进入长程性能结论；run30 是基准而非已证实的最终成功版本。
- 来源：本次历史对话中的 run30 对照数据。

### run35--run38

#### 实验目的与代码演进

- run35：advantage-aware triplet scorer。
- run36：检查 adj PPO high-clamp early-stop，后续确认其初次配置未真正启用。
- run37：真正启用 `adj_ppo_clip_stop_ratio=0.35`、`adj_ppo_factor_clip_stop_ratio=0.35`、`adj_ppo_min_epochs=1`。
- run38：检查 stale replay trust。其分析要求包括 raw/trusted clamp、trust weight、order3 raw credit 和 triplet marginal 诊断。

#### run37 关键日志与结论

- final reward≈1.183、last5≈1.057、best≈2.073、win rate≈0.1、capture events≈2.6、eval order3 ratio≈0.442。
- early stop 的启用或 clamp 变化没有带来超过 run30 的性能；该机制只能称为接线/触发验证，不能称为性能成功。

#### run38 关键日志与结论

- final reward≈0.895、last5≈0.148、best≈1.361、final win rate≈0.4、capture events≈2.8、eval order3 ratio≈0.442、order2 ratio≈0.242。
- raw graph clamp≈0.742、trusted graph clamp≈0.465；raw factor clamp≈0.511、trusted factor clamp≈0.245；graph trust weight≈0.468、factor trust weight≈0.643。
- `raw_o3_minus_o2_factor_rl_loss≈0.0621`、`adv_triplet_marginal_mean≈-0.0083`、`adv_triplet_score_multiplier_mean≈0.998`。
- trust 权重使 trusted clamp 数值下降，但 reward/last5/best 均低于 run30；因此不能将 trusted clamp 下降解释为克服 stale graph 或性能改善。

#### 来源与证据

- 本次历史对话摘要中的 run37/run38 训练日志摘录。

### run40、run48：不可用于性能比较的阻断性错误

#### 训练完整性

- run40 的预检/验证在 `adj_generator.py` 的 triplet 置信度路径因旧 PyTorch 不支持 `torch.minimum` 退出；未形成可分析训练曲线。
- run48 在约 6,600 env steps 进入 `train_adj_on_batch` 时因 `factor_training_mask` 相关张量维度 400 与 6 不匹配退出；未形成 200k 结果。

#### 最终结论

- 两个 run 的共同结论限于运行错误暴露了兼容性或 batch 形状接线问题；不得与任何完成的 run 比较 reward、capture 或 win。
- 来源：本次历史对话中的 Traceback 摘录。

### run42

#### 实验目的与配置状态

- 完整 2M 长程 SDDFG 动态训练。历史分析确认它启用了本轮的 recent replay、delayed/future/success/graph-return 路径；具体所有参数、commit、checkpoint/optimizer/replay 恢复状态未在当前可见记录中完整保留。

#### 关键日志与指标

- final reward=5.918、last5=5.700、last10=6.046、best=8.636@1.82M、final win rate=0.4、capture events=4.1。
- best 明显高于 final；final reward和last5低于QMIX 2M的6.947/6.175，last10=6.046高于QMIX实际last10约5.559、但低于项目历史采用过的严格6.175门槛；win rate低于0.8。

#### 有效性判断与结论

- 该 run 说明该版本可以出现较高 reward 峰值和较高 capture，但没有形成稳定胜率与末段回报闭环。它不能证明 delayed/future/success/graph-return 中的任一单独机制有效。
- 来源：本次历史对话整理的完整 run42 结果摘录。

### run49

#### 实验目的与配置状态

- 完整 2M 长程 SDDFG 动态训练。保存 config/console 被历史分析为启用 `ADJ_TRIPLET_GRAPH_RETURN_CREDIT_REQUIRE_DELAYED_GATE=1`、`ADJ_DELAYED_TRIPLET_SUCCESS_GATE_FLOOR=0.10`、`ADJ_DELAYED_TRIPLET_PARTIAL_MATCH_WEIGHT=0.35`、recent-window minimum=1、shrink patience=1、emergency window=1 和相关 stale 阈值；pair reserve 与 future exact/partial/matched 日志也已记录。全部机制同时存在，不能归因于单一变量。

#### 完整趋势

- 200k：reward 0.094、last5 1.180、last10 0.009、best 3.321@120k、win 0.4、capture 2.4。
- 400k：reward 0.498、last5 0.870、last10 0.583、win 0.1、capture 2.4；600k：reward 2.499、last5 2.210、last10 2.171、win 0.3、capture 2.7。
- 800k：reward 2.569、last5 1.128、last10 1.120、win 0.4、capture 3.1；1M：reward 1.786、last5 3.172、last10 2.849、best 4.538@960k、win 0.2、capture 2.3。
- 1.2M：reward 1.583、last5 3.267、last10 2.806、best 4.701@1.12M、win 0.5、capture 2.7；1.4M：reward 5.067、last5 4.541、last10 4.899、win 0.5、capture 3.6。
- 1.6M：reward 5.073、last5 4.913、last10 4.406、best 6.103@1.52M、win 0.4、capture 3.9；1.8M：reward 6.709、last5 5.323、last10 5.202、win 0.7、capture 4.4。
- 2M：reward 6.296、last5 5.751、last10 5.230、best 7.815@1.94M、win 0.9、capture 4.5。

#### 有效性判断与结论

- run49 超过 run30、run18 200k 前缀和 QMIX 200k 的终值级别，也超过 VDN/QPLEX 的已知 200k 结果；这是跨训练步数和版本的历史比较，不构成单变量归因。
- 相比 run42，它的 final reward、final win rate 和 capture 更高；但其 1.8M--2M last5/last10 仍低于 QMIX 2M 对照，且 best 后未保持。该判断是run68出现前的历史结论；run68后来超过其reward终值和滑动回报，但win低于run49。
- 来源：本次历史对话整理的 run49 完整 CSV/日志分析摘录。

### run50

#### 实验目的与配置状态

- 检查 capture-to-win triplet credit。历史记录的启用参数：`use_adj_capture_to_win_credit=True`、coef=0.15、min outcome advantage=0.50、scale=0.75、cap=0.25、require future match=True。其余图机制未作为单变量关闭，因此不是纯单变量实验。

#### 关键日志与趋势

- 20k/40k/60k/80k/100k/120k/140k/160k/180k/200k 的 reward 依次为 -4.940、-1.927、1.562、-0.109、-1.691、0.026、2.796、-1.778、-0.145、1.829；对应 win rate 为 0.0、0.0、0.2、0.2、0.4、0.3、0.5、0.2、0.4、0.4；capture 为 0.8、1.1、2.6、2.6、1.9、2.6、3.3、2.1、2.6、2.6。
- final reward=1.829、last5=0.546、last10=-0.438、best=2.796@140k、win rate=0.4、capture=2.6。
- 200k 的 `capture_to_win_triplet_credit_mean=0`、active fraction=0、quality gate mean=0。

#### 有效性判断与结论

- 该 run 未超过 run30 的 final、last5、win 或 capture 门槛，且 best 后回落。硬 outcome credit 在末节点没有触发，不能称为对策略的有效稳定训练信号。
- 来源：本次历史对话整理的 run50 eval/train-adj 日志摘录。

### run51

#### 实验目的与配置状态

- 在 capture-to-win 路径上启用 pair/triplet complementary credit。保存 config/console 记录 `use_adj_pair_triplet_complementary_credit=True`、pair pursuit coef=0.10、window=20、cap=0.20、min reward=0.0；capture-to-win 仍为 coef=0.15、min outcome advantage=0.50、cap=0.25、require future match=True。

#### 关键日志与趋势

- 20k/40k/60k/80k/100k/120k/140k/160k/180k/200k 的 reward 依次为 -1.824、-3.625、-0.294、-2.889、0.545、-1.106、-4.646、-0.316、-1.034、-0.864；对应 win rate 为 0.1、0.0、0.1、0.0、0.4、0.3、0.2、0.1、0.0、0.2；capture 为 1.3、1.7、2.5、1.6、2.9、2.8、2.2、2.0、1.9、1.9。
- final reward=-0.864、last5=-1.593、last10=-1.605、best=0.545@100k、final win rate=0.2、capture=1.9。
- pair pursuit credit active fraction 平均约 0.923、末值约 0.943；pair pursuit quality mean 平均约 0.540、末值约 0.487；triplet capture quality mean 平均约 0.0666、末值约 0.1278。capture-to-win credit active fraction 平均约 0.0459、末值 0；delayed credit active fraction 平均约 0.118、末值约 0.130；future matched/exact/partial fraction 平均约 0.958/0.843/0.809。adj graph/factor stale ratio 平均约 0.443/0.260、末值约 0.480/0.305。

#### 有效性判断与结论

- run51 低于 run30、run50、run18 和 QMIX 200k。pair pursuit credit 高覆盖与训练退化同时出现，说明“字段启用且大量触发”不是性能有效证据；不能从这一单个多变量结果断言退化完全由 pair credit 引起。
- 来源：本次历史对话整理的 run51 config、console、eval 与 train-adj 日志摘录。

### run52--run55：pair credit 从过宽广播改为真实 capture 锚定

#### 实验目的与修改

- 直接处理 run51 中 pair credit active fraction 约92.3%的过宽激活：普通正 shaping reward 不再触发；仅 strict-future、same canonical pair、真实 capture event 可产生 credit；offset=0 严格为0；移除历史 floor。
- 该阶段同时把 `capture_count`、`success_now/episode_success` 的真实事件字段贯穿 runner、buffer、batch 和 learner，并持续检查多环境与时间索引。

#### 关键日志与结论

- run52--run55 的 pair active均值依次约0.74%、0.76%、0.89%、1.14%，相较 run51 的92.3%发生数量级下降。普通正reward与offset0不再形成大面积激活，说明结构修复真实进入训练。
- 性能从 run52 的 final -0.501 逐步到 run55 的2.728，capture从2.1到3.0；但 run55 last5仅1.111、win0.4，未整体超过run30。故“pair误激活修复”是正确性成功，不是稳定性能成功。
- 历史中出现旧 PyTorch `torch.minimum` 不可用、`validate_adj_buffer` 数值断言失败等阻断；这些错误属于兼容/测试口径，不能据中断run评价机制性能。
- 来源：run52--run55完整eval CSV、train-adj、console与源码检查；来源：本次历史对话整理。

### run56--run59：centered outcome、episode总量和capture identity

#### 版本演进

- run56 definition v2（manifest前缀 `067b`）：centered success/failure capture outcome。
- run57 definition v3（manifest前缀 `f1c`）：episode total distributed across capture triplets。
- run58 definition v4（manifest前缀 `4000`）：participant identity matched factor。
- run59 definition v5（manifest前缀 `a186`）：`highest_exactly_representable_capture_factors`，两参与者精确匹配order2。

#### 关键日志与判断变化

- run56 final4.215、last5 2.005、capture3.5，是该阶段最强reward/capture，但win仅0.3；随后发现 factor展开会破坏episode级中心化，不能把run56直接认定为正确机制成功。
- run57 修复episode总质量后 final0.742；训练前后曾出现 `class_sum==1`、AdjBuffer expectation 等验证断言，说明测试需同步新定义。归一化逻辑后续中心误差接近0并被冻结。
- run58 的41个训练capture event全部为两参与者，identity matched=0；outcome credit/local delta全零。最初“真实capture必须对应order3”假设被环境事实否定。
- run59 支持exact order2后，45个event中4个进入active exact match（8.89%），证明两人匹配接线有效；但绝大多数真实factor只存在于candidate或不在active，行为仍弱。期间曾因把identity匹配到非法/非pair-triplet factor而抛出显式RuntimeError，另有 `dones.transpose` 维度错误；均在后续完成训练前修复。
- 来源：run56--run59 CSV、console、manifest、验证Traceback和源码检查。

### run60--run65：outcome support、cohort中心化、confidence与factor loss

#### 支持机制

- run60 support v1：补齐稀有正负episode，但相同稀有episode可跨adjacency update重复使用；23个augmentation窗口不能代表23个独立episode。
- run61 support v2（manifest前缀 `e041`）：slot-generation跨update只消费一次，同一update多epoch复用；13个enabled/augmented窗口和28个exhausted窗口表明重复使用受控，但信号快速耗尽。
- run62 support v3（manifest前缀 `3c21`）：baseline改为最终optimizer cohort；7个class-complete/center-valid窗口中心误差为0。run62 final1.290、last5 1.604、last10 0.721，但win0.2。

#### 后续链路修复

- run63（manifest前缀 `de77`）引入 detached absolute graph-confidence；其全部10个eval点与run62完全相同。结合源码发现 graph advantage replay 写入/ready source 未正确提供训练量，说明“代码版本不同”不等于“训练张量不同”。
- run64（manifest前缀 `2c12`）修复 stored graph-return advantage 写入后，eval轨迹改变，证明新来源真实进入训练；final0.428，性能未改善。
- run65（manifest前缀 `1744`）修复 outcome factor loss normalization；last5/last10达到2.956/1.214，但final1.754、win0.3、capture2.4，best后未保持。
- 此阶段还出现 support age 断言口径错误、旧PyTorch缺少 `torch.count_nonzero` 等预检问题；其修复范围是测试兼容性。
- 来源：run60--run65 CSV、console、manifest、support/outcome诊断和Traceback。

### run66--run70：candidate identity loss 从接线到单步方向

#### 版本与真实作用

- run66 candidate loss v1（manifest前缀 `2d62`）使用不合适的Bernoulli形式；candidate-only84.21%，final -0.867。
- run67 v2（manifest前缀 `87ca`）改为signed log-weight；candidate-only降至76.74%，但绝对weight并不等同于条件选择概率。
- run68 v3（manifest前缀 `4af2`）使用当前可求导canonical catalog的conditional probability。2M最终6.718/0.6/4.6，best7.724@1.86M；前200k仅final0.344、win0.1、capture2.3，说明长程结果不能由短程峰值或单项loss推断。
- run69 v4（manifest前缀 `0e31`）将分母改为target-bearing transition，candidate/graph loss升至约0.801%，证明无关transition稀释问题被修复；但optimizer后正/负概率正确率仅13.52%/25.85%，说明主要失败层已从loss尺度转移到梯度合并/optimizer实际更新。
- run70 加入candidate gradient projection v1，正/负概率正确率提升到84.38%/82.14%，final3.555、win0.8、capture3.4；candidate-only仍80%、正/负rank变化约2.34%/2.68%。该run证明欧氏冲突投影能改善即时方向，但未证明跨update保持或严格active转化。
- 来源：run66--run70 CSV、train-adj diagnostics、console、manifest与源码diff。

### run71--run75：Adam实际位移、状态同步和candidate lifecycle

#### run71--run72

- run71 actual-update guard v1 试图在Adam step后保证candidate下降方向；随后验证阶段出现 `Adam raw update reconstruction is inconsistent with the optimizer state`，暴露服务器Adam公式/bias correction与反解假设不一致。
- run72 state sync v2（manifest前缀 `1183`）自动在两种公式间选重建误差较小者。该run完成训练、candidate loss下降率100%、正/负概率正确率86.67%/88.89%，但rank仅1.11%/1.85%，final1.221、win0.1。按误差拟合公式和只重建`exp_avg`不能证明下一步Adam轨迹自洽。

#### run73--run74

- run73 state sync v3（manifest前缀 `e50b`）按实际optimizer属性确定标准Adam公式，并加入lifecycle v1。lifecycle只保护聚合梯度、未覆盖全部target-bearing路径，TTL按optimizer step，rollback还曾虚增candidate policy version。protected update=52、单transition violation窗口=23、rollback率32.21%；final1.608、candidate-only89.47%。
- run73后的联合candidate/lifecycle验证曾抛出 `RuntimeError: joint candidate/lifecycle constraints have no strict current-candidate descent direction`。该错误证明旧cache与当前真实target可能形成不可严格同时下降的约束集合，促使run74版本加入按真实证据优先级的supersession与逐transition诊断；该Traceback本身没有可比较性能。
- run74（manifest前缀 `7dff`）实际为lifecycle v3而不是请求文本所称v2：adjacency-round horizon=4，旧cache与当前target统一逐transition active-set约束，并用证据supersession处理冲突。30个target窗口、46.5 target质量；欧氏投影后最小dot约`-6.91e-10`（容差内），但Adam实际位移最小dot约`-1.625e-5`，导致74个受保护update中频繁非线性失败和33.78% rollback。
- rollback首次约100.8k；100--120k rollback率32.35%，120--160k升至47.73%，160--200k仍30%。同期candidate loss下降率83.33%、正/负即时概率正确率83.33%/50%、rank约0.208%/0，final -0.047、win0.3、capture1.8。日志时间顺序支持“Adam位移违反cached halfspace→事务rollback→graph有效更新受限→rank/active无积累→行为弱”的当前主要因果链。
- run74结束后的工作区改为lifecycle v4，直接在Adam realized displacement空间满足每个cache halfspace并用确定性backtracking检查真实loss；同时修复TTL off-by-one和保持率观察档案。该状态随后由run75训练验证。

#### run75有效性、直接对照与实际结果

- run75只有一个可定位目录和一个对应控制台日志。目录包含35个文件：282项config、10点eval CSV、65点progress/train、242点train-adj、完整轨迹CSV、1164个TensorBoard scalar tag、18个最终/最佳模型文件和空的`summary.json`。目标步数200k，train-adj和eval均到200k；console常规训练打印到198.6k后完成最终200k eval和模型/TensorBoard落盘。
- seed=1、`num_env_steps=200000`、`num_eval_episodes=10`、`eval_interval=20000`、`model_dir`为空。console无Traceback、RuntimeError、AssertionError、NaN或Inf，也未出现checkpoint、optimizer或replay恢复记录。10个eval step无重复，CSV和TensorBoard的final reward/win/capture一致。
- run74与run75的282项保存配置逐项完全相同；源码manifest分别为`7dffb2ca...`和`5f6347ac...`。run74运行时header为lifecycle v3，run75为v4。两run前20k、40k、60k的全部eval字段完全相同，首个candidate target在68.8k出现，80k开始轨迹分化，支持run74作为直接对照。配置层是单变量；源码层是一个相关的lifecycle v4修复包，包含实际位移投影、backtracking、state sync、TTL和observation archive多个耦合子项，因此不能把性能差异归因到其中单一子项。
- run75十点reward为`-1.178,-2.985,-1.726,-0.677,0.247,1.947,1.689,-1.632,1.223,1.788`。前期持续为负，100--140k改善，120k达到best1.947，140k仍为1.689，160k骤降到-1.632，末两点恢复但未超过best。final=1.788、last5=1.003、last10=-0.130、win=0.5、capture=2.8；高于run74的final/win/capture，但低于run70的3.555/0.8/3.4，也低于历史200k参考值，且后期稳定性不足。
- final eval的10局共28次capture，5局出现success；首次成功步的5个有效样本均值为105。每局均有leave=4、join=2、recover=4、6次topology change，最终pending recovery为0；100个eval episode均为200步。65个训练active-mask轨迹均完整覆盖`t=0..199`，首步固定4个active槽位、末步6个active槽位，未见跨episode lifecycle污染、重复时间步或终止边界异常。
- candidate loss在31个日志窗口非零，target质量和为43.5；正/负optimizer后概率方向正确窗口率均为100%，但正/负rank改善均值仅1.61%/4.89%。事件计数汇总为115.5，active matched为8、unmatched为107.5，candidate-only约93.07%，比run74约89.57%更高。non-target delta最大值为0，outcome cohort center error为0，说明identity局部性和中心化未破坏；低active覆盖仍是行为转化瓶颈，但没有证据支持本轮改算法。
- lifecycle约束在83个日志窗口触发，protected target质量和349.25。实际Adam位移投影在18个窗口修正；修正前最小约束dot为`-8.24e-6`，修正后负约束计数为0，最小正dot约`1.23e-8`；仅记录一次聚合值0.25的非线性backtracking，current-candidate nonlinear violation为0，rollback/reject为0，242个adj update全部推进policy version。age1/5/10可观测样本分别为82/76/74，probability保持率为50.00%/62.17%/64.53%，rank保持率为74.70%/74.67%/81.76%；各age cohort不同，不能据此声称保持率随年龄单调上升。
- graph/factor/总loss和梯度均为有限值：adj总loss均值约-0.0050、graph约0.00176、factor约-0.00676，adj grad norm均值0.173、最大0.557。target窗口candidate/graph绝对loss比例均值0.326、中位0.175、最大2.244，没有持续主导总loss。价值训练loss均值0.975、范围0.445--1.882，TD绝对误差均值0.373；policy/Q/RNN梯度均有限。raw graph/factor clamp均值约0.435/0.219，236/242个窗口触发early stop，属于持续存在的PPO更新压力，但与run74相近，日志不足以支持本轮调整。
- 奖励/credit不是“打印即有效”：graph-return和delayed credit分别在167和231个日志窗口非零；capture-to-win/outcome factor只在21个窗口非零，active fraction全程均值约`6.0e-5`；pair pursuit只在16个窗口非零，active fraction均值约`4.97e-4`。support class-complete/credit-enabled为31/242窗口，support exhausted为40/242窗口。信号真实进入训练但非常稀疏，不能用新增reward或提高loss尺度掩盖覆盖问题。

#### run75根因与本轮代码修复

- 已确认的训练机制问题：run74的Adam实际位移违反cached halfspace在run75中没有复现。run75实际投影修正非零、修正后负约束为0、rollback为0，故不能继续把run75性能不足归因于run74同条件的位移/rollback错误。
- 已确认的日志统计错误：run75在step155200记录`candidate target_count=1.5`、candidate loss=`-0.01236`、candidate grad norm=`0.01254`，但`capture_candidate_identity_lifecycle_target_bearing_update=0`。源码中该字段原先只在`lifecycle_cache_size>0`分支内赋值；当当前mini-batch有真实candidate target但进入分支前cache为空时，诊断会错误记0。该错误只影响日志分母/统计，不影响loss、梯度、参数更新或cache注册。
- 本轮只修改`algorithms/sddfg/r_sddfg.py`，使target-bearing诊断直接由当前`candidate_target_present`决定，不再依赖旧cache是否已经存在；并在`scripts/debug_candidate_identity_supervision_synthetic.py`新增回归测试。未修改reward、loss、梯度投影、optimizer、replay、mask、超参数或环境。
- 语法/import、19项candidate identity合成测试、完整SDDFG preflight、动态图验证和`git diff --check`均通过。代码修复已完成，但性能效果尚未经过新run验证。
- 源码manifest限制：console只保存21个文件hash列表的二次聚合SHA，没有逐文件hash artifact；Git commit/tree又为unavailable。因此能够确认run74/run75源码集合不同和运行时机制header不同，但无法从落盘产物逐文件重建精确server源码diff。该限制不影响本次已由运行时指标证明的v4触发结论，但限制字节级追溯。
- 来源：run74/run75全部落盘CSV、TensorBoard、console、config、模型/最佳点元数据、训练脚本manifest逻辑、当前源码与本轮测试。

### run81--run86：candidate residual、lifecycle v10与trusted-control前史

#### 代码演进与有效性边界

- run81是后续 candidate 系列的比较点：200k reward=-0.720、220k=2.880、240k=1.305、last5=1.099；candidate-only=88.42%、crossing=0。完整 config、恢复状态和源码差异未在当前对话保留。
- run82--run83从 candidate v11 发展到 v12：采用 first-reachable active-competitor signed hinge、normalization v3、PPO early-stop 后有限 residual；v12将 residual Adam 与标准 adjacency Adam 隔离。run83阶段的 optimizer state 索引测试与 `final_candidate_info` 未初始化曾分别阻断合成测试和训练；后续 run 完成仅证明阻断已处理，不能反推独立 residual 提升性能。
- run84--run85把 lifecycle v10、terminal model和双 Adam checkpoint 纳入正式路径。v10只保护已产生rank/crossing行为进展的 target；checkpoint缺模型、双Adam或completion metadata应失败。当前对话没有保留run84/85全量CSV审计，所以仅把“用户报告完成训练”记录为训练状态，性能保持未确认。
- run86的控制日志修复目标是 trusted stale population 同时控制 early-stop 与 recent window，而不是只改变日志。结果在240k出现3.487，但260k降至-0.648、280k仅1.042；因此从结果上不能称为稳定性能成功。

#### 来源与证据

- 来源：本次历史对话中的run81--run86用户提供的配置、Traceback、目标机制和比较指标；除run86给出的eval点外，其余run完整产物未在本次整理中重新读取。

### run87--run90：runtime control、population-total与capture质量诊断

#### 运行时控制链

- run87用于验证 trusted control 运行时接线并运行至2M。后续历史摘录记录window均值1.748、57.9% update仍为window=1、recovery287次；说明窗口不再永久锁死，却不能证明实际 adjacency replay 多样性已足以改善价值学习。run87 final5.628、last5/last10为6.221/6.348、best8.076@1.88M，均低于run68的final与滑动窗口，final win/capture为0.4/4.0。
- run89修正 control ratio 的错误平均：必须以总量比值而非 mini-batch ratio 的均值控制。run89 200k=1.080、400k=1.238、last5/last10=1.330/1.298，未恢复run70/run78的200k水平。
- run90达到600k。它在580k出现4.320 best，但final仅0.768；last5/last10=2.437/1.948。200k capture=3.1却仅win=0.4，失败episode capture比例43.8%、capture episode win率44.4%，明显劣于run70/run78约25%的失败capture比例和77.8%--88.9%的capture episode胜率。这一事实把性能缺口从纯捕获数量指向capture质量/最终成功转化，但并未单独证明某个credit机制是唯一原因。

#### 来源与证据

- 来源：本次历史对话中run87--run90用户提供的对照表、控制链审计要求和结果摘录；未见完整原始CSV的数值不作补造。

### run91--run95：signed pair credit 到 optimizer transaction 的连续修正

#### run91：signed pair credit

- run91以成功/失败capture episode的centered outcome为pair信用符号。其200k=-0.188、win0.3、capture2.0，400k=0.400、last5/last10=0.864/0.556；相对run90没有稳定改善。期间还出现 `debug_outcome_factor_loss_synthetic.py` 无法导入 `compute_capture_outcome_factor_ppo_loss` 的测试接口错误；后续run继续完成，但可见对话未保留精确兼容导出补丁。

#### run92--run94：local loss、v2分母和optimizer cohort

- run92将pair从shared advantage移到identity-local PPO后，80k达到3.529但200k=-0.418、win0.1、capture2.2；400k=1.984。它证明早期高点不能证明机制有效。
- run93以pair-target-bearing transition为分母后80k达到5.331、200k1.759，但400k1.025、last5/last10仅0.811/0.569。事后审计显示full-buffer中心化与实际optimizer子cohort不一致：154个pair evidence update中73个非零pair update，52个为one-sided（71.23%），正负质量不平衡（0.002582对0.001399），最长有效监督空窗107.2k。因此run93的v2不是可接受的性能成功证据。
- run94改用最终optimizer cohort重新中心化，one-sided pair loss为0。它的200k为2.786、win0.4、capture2.5，400k1.267、last5/last10=1.382/1.428；比run93在200k局部提高但仍低于run70/run78，且非零pair update仅24、200k前仅3次、首次target155.2k，监督覆盖明显不足。

#### run95：support v4、原子transaction与当前修复

- run95有效性：seed=1、目标/实际400k、20个20k eval、fresh且无恢复；控制台/CSV/TensorBoard无Traceback、NaN或Inf。运行产物和当时manifest支持其使用 signed pair、identity-local PPO、normalization v2、class-complete exact pair support、cohort centering、one-sided zero与transaction质量守恒。服务器该run为真实PyTorch1.3.1训练。
- run95性能：20k至400k reward依次为`0.399,-1.081,2.324,-0.756,1.825,4.097,0.221,2.074,0.747,2.116,1.529,1.858,-0.699,0.245,2.138,-1.129,0.434,-0.654,1.124,0.104`；best=4.097@120k，200k=2.116/win0.3/capture2.3，final=0.104/win0.2/capture2.1，last5=-0.024、last10=0.495。120k的capture4.2、win0.7随后同步回撤；260k、320k和末段capture/win均低，不能写成稳定正效应。
- run95 pair support实际覆盖：492个adjacency update中raw evidence=165、both-class available=79、class-complete=42、support augmentation=29行/33 episode、support exhausted=37；非零pair transaction=42，首次23.2k，0--100k/100--200k/200--300k/300--400k分别2/4/7/29次，最长空窗72k。每个非零transaction正负质量守恒、one-sided=0、最大中心误差约3.725e-9；但覆盖仍呈长空窗。
- run95行为与其他诊断：candidate target rows=210、positive/negative rank改善约3.115%/1.990%、positive crossing约0.0595%、identity match约18.39%；topology connected=1、retention约0.626，Q梯度P95约2.640、RNN梯度P95约0.896。120k之后static/dynamic slot reward同步下降（static约0.756至0.028、dynamic约0.610至0.045），现有证据不足以将其归为slot实现错误。
- run95源码审计确认且已修复的独立bug：atomic pair transaction把pair chunks固定置于第0 partition，使pair和普通base factor PPO transaction的chunk数不一致。例如N=6、两分区时pair transaction=2 chunks、另一transaction=4 chunks。该差异会改变base factor PPO的每步平均人口/Adam交易语义，是性能相关批构造错误。当前工作区已以filler与公平slot修复，尚未由新run验证。

#### 来源与证据

- 来源：本次历史对话整理中对run95控制台、CSV、TensorBoard、源码与定向测试的完整审计；run91--run94的数值主要来自用户提供比较表和后续问题描述，缺少本次可直接复读的完整落盘产物时均按此边界记录。

### run96--run100：support v5/v6与optimizer方向链

#### 实验目的与代码演进

- run96验证run95后v5的pair partition人口平衡修复；运行时仍记录support version=5。run97改为support v6：class-complete evidence使用完整selected population、单partition和standard adjacency Adam transaction，不再以pair partition改变base PPO人口。
- run98是fresh seed=1、120k机制基线；run99增加聚合pair gradient诊断；run100增加per-epoch transaction diagnostics v2。run98、run99、run100的六个eval点及公共训练字段一致，故两版诊断均未改变训练轨迹。

#### 训练完整性与性能

- run96、run97均fresh seed=1、400k，`model_dir`为空。run96 final=`0.886`、last5=`1.0908`、last10=`1.4766`、best=`3.565@260k`、final win=`0.1`、capture=`2.1`；run97 final=`1.481`、last5=`1.0694`、last10=`1.5760`、best=`3.262@220k`、final win=`0.5`、capture=`3.0`。二者均完成训练且可用于机制比较，但都没有建立稳定超过run70/run78的性能证据。
- run98--run100的120k共同结果为final=`0.862`、last5=`0.544`、best=`2.324@60k`、final win=`0.2`、capture=`2.7`。该共同轨迹主要用于诊断中性与optimizer链定位，不构成性能成功。

#### 关键日志、最终结论与经验

- run100共有六个nonzero pair epoch transaction。77.6k/0、89.6k/0在raw combined gradient阶段已经与pair目标反向；77.6k/1、89.6k/1的combined gradient正确，但Adam历史moment使raw/final displacement继续反向；108k两个epoch的combined、Adam、final和exact score均正确。
- 六次clipping均未改变方向，raw Adam与final committed displacement一致，candidate correction、lifecycle、backtrack和rollback不是这些最终位移的原因。run100因此把最早断点区分为objective合并和Adam历史moment两层。
- 有效性判断：run96--run100均有效；run99/100属于轨迹中性诊断。来源：saved config、eval CSV、train-adj CSV与run100 transaction CSV；来源：本次历史对话整理。

### run101--run102：diagnostics v3与graph负投影根因

#### run101无效性

- run101目标为fresh seed=1、110k，但`r_sddfg.py`与`base_runner.py`的diagnostics版本不同步，在runner构造transaction row时抛出`RuntimeError: unexpected pair optimizer diagnostic version`并退出。该run没有完整训练、eval或v3 transaction产物，不得用于性能、gradient、optimizer、checkpoint或算法根因分析。

#### run102有效结果

- run102是同步完整v3后的fresh seed=1、110k实验。v3按真实total adjacency loss分解graph、base factor、capture outcome、pair、candidate、entropy，并验证scalar、独立gradient和projection前后重构；公共v2字段与run100截至110k一致。
- 77.6k/0中graph pair dot=`-3.4724e-3`，candidate=`-6.3623e-4`，base=`2.9727e-6`，outcome=`2.4897e-5`，entropy约`5.81e-8`；89.6k/0中graph=`-5.1616e-4`，candidate=`-4.2557e-4`，base=`9.8015e-5`，outcome约`4.00e-7`，entropy约`-1.07e-7`。graph在两个最早反例中均为最大负投影，且反向在projection前发生。
- 源码确认graph PPO误用了包含identity-local/delayed local credit的混合`f_advts.mean()`，而独立replay graph-return虽已存在却未用于graph loss。这是advantage source责任范围错误，不是仅凭负cosine推断的自然冲突。
- 有效性判断：run102严格有效；v3诊断轨迹中性；graph source错误有CSV与源码双重证据。来源：run102 transaction CSV、saved config与源码检查；来源：本次历史对话整理。

### run103--run106：graph source v2后监督漏斗与candidate rank断点

#### graph source v2与run103

- 修复后graph PPO只消费`αG`，factor-local residual使用`f_advt-αG=βL+D_local`并精确重构原`f_advts`。run103为fresh seed=1、110k，启动与日志确认`graph_advantage_source_version=2`及contamination contract。
- run103的20k--100k reward为`-5.187,-3.395,1.340,0.977,3.269`，五点评估均值`-0.5992`；100k win=`0.5`、capture=`3.4`。截至110k，negative pair evidence available update=38、positive=0、class-complete=0、pair target/gradient/transaction均为0。graph污染被修复，但pair链尚未启动。

#### run104--run106逐层定位

- run104增加funnel v1且与run103公共轨迹一致。4个successful episode全部含capture，4个successful capture全部为candidate-only，successful active capture=0、positive pair evidence=0；terminal、participant=2、dynamic slot、canonical identity、support selection与capture provenance均未发现实现错误。
- run105增加funnel v2、episode reject-reason CSV和candidate boundary join；公共轨迹仍一致。唯一成功generation 370包含order-2 identity`0-4`与`2-5`，均获得正candidate target，reject集中于`CANDIDATE_ONLY_NOT_ACTIVE`。
- run106增加same-population per-target transaction CSV。四条successful-overlap target均有非零candidate gradient、正确combined/Adam committed方向和正signed margin change，但rank与boundary均未跨越；没有lifecycle reject或rollback。
- `0-4`两个epoch的margin改善为`5.6267e-5`、`1.2898e-4`，rank保持23，next-better gap由`0.066002`降至`0.064315`；`2-5`改善`2.3317e-4`、`5.3954e-4`，rank保持24，gap由`0.109466`降至`0.108803`。实际改善只占合法next-better gap约0.085%--0.494%，占active boundary约0.0025%--0.0236%。
- 有效性判断：run103--run106均fresh seed=1、110k且有效；run104--run106新增诊断轨迹中性。已确认断点由“positive evidence缺失”细化为“candidate更新正确但小于合法rank gap”，不能归因于target漏失、sign、identity、Adam、rank cache或invalid population。来源：eval、funnel、episode与candidate identity transaction CSV；来源：源码检查；来源：本次历史对话整理。

### run107：200k provenance观测、独立generation与strict pair时窗

#### 配置、完整性与性能

- run107为fresh seed=1、200k，未恢复model、optimizer或replay；前110k公共字段与run106一致。20k--200k reward依次为`-5.187,-3.395,1.340,0.977,3.269,-0.823,1.686,0.264,1.873,2.177`；final=`2.177`、last5=`1.0354`、last10=`0.2181`、best=`3.269@100k`、final win=`0.3`、capture=`2.9`。100k后的波动和回撤使其不能被视为稳定性能成功。

#### provenance与独立证据

- candidate provenance CSV共102行、102个唯一完整键，positive/negative event row为35/67；provenance、identity和quality contract均成立。一行是一个真实event、canonical identity和sign，不把PPO epoch或replay exposure当作独立行为证据。
- first-consumption-only v3反事实中，`0-1`有3个独立generation，累计signed margin=`0.0038519`且有一次抵消；`0-4`有2个，累计`-0.0004511`且有抵消；`2-5`有2个，累计`0.0051920`且无抵消。三者均未跨next-rank或active boundary，且generation间population/competitor变化，故只是标量隔离而不是真实训练反事实。

#### strict pair evidence生命周期结论

- strict pair positive/negative generation为2/36，successful active capture exposure=12，但class-complete、pair target、pair gradient和pair transaction均为0。overlap重建确认现有recent replay中opposite-sign overlap=0，shared used-state单独阻断数=0。
- horizon sweep v2按support v6结构规则重建：额外4个adjacency update可在144.8k组合`689(+)+660(-)`；额外7个可在156.8k再组合`748(-)+709(+)`。run107日志缺少evicted immutable训练payload，因而当时只能证明结构机会，不能声称fully trainable。
- 有效性判断：run107严格有效；provenance日志轨迹中性；最早strict pair断点是正负证据在现有replay有效窗口内不重叠，而非support漏选、提前used或optimizer错误。来源：run107 eval、candidate provenance、pair evidence episode、adj/transaction CSV、独立反事实文本、overlap与horizon JSON；来源：本次历史对话整理。

### run108：bounded pending首次production触发与完整回滚

#### 配置与运行完整性

- run108为fresh seed=1、200k，唯一训练变量是`pair_bounded_pending_evidence=true`、`pair_pending_max_adj_updates=4`。production路径包含event provenance、34字段immutable snapshot、pair-only objective、transaction-time stale/mass重构、two-epoch outer atomic transaction、single-use和checkpoint state。
- run108的20k--200k eval与run107逐点一致，final=`2.177`、last5=`1.0354`、last10=`0.2181`、best=`3.269@100k`、final win=`0.3`、capture=`2.9`。这是完整rollback导致无净训练差异的证据，不是pending性能成功。

| eval step | reward | win | capture |
|---:|---:|---:|---:|
| 20k | -5.187 | 0.1 | 0.8 |
| 40k | -3.395 | 0.0 | 1.8 |
| 60k | 1.340 | 0.2 | 3.2 |
| 80k | 0.977 | 0.2 | 2.7 |
| 100k | 3.269 | 0.5 | 3.4 |
| 120k | -0.823 | 0.1 | 2.5 |
| 140k | 1.686 | 0.5 | 3.3 |
| 160k | 0.264 | 0.5 | 2.8 |
| 180k | 1.873 | 0.3 | 3.8 |
| 200k | 2.177 | 0.3 | 2.9 |

#### pending漏斗与唯一事务

- 共创建26个snapshot，最终均进入过期状态；出现一次current/pending overlap、一次class-complete prepared和一次pair-only训练attempt。144.8k的cohort由pending generation660 negative与current generation689 positive构成，pending age=4/0、policy age=32/4，raw mass约0.05/0.05。
- epoch0 stale trust正/负为0.625/1.0，effective mass约0.03154/0.05，pair loss=`0.0083798`、pair gradient norm=`0.0028520`，standard optimizer step从692到693。事务随后被通用graph/base PPO early-stop在1/2 epoch后中止，所有参数、optimizer moment/step、RNG、lifecycle和pending状态完整回滚；logical commit=0、reuse=0、zero-target/zero-gradient abort=0。

#### 运行后根因与代码状态

- pair-only事务把graph/base标准early-stop错误用于只含pair objective的control population。由于pair-only明确令graph、base、outcome、candidate和entropy为0，该control scope与事务成功条件不一致。
- 当前工作区已把pair-only control改为exact nonzero pair target population，记录raw/trusted clip ratio，标记standard early-stop不适用，并要求全部配置epoch成功后才commit；普通adjacency PPO early-stop未改。abort row的objective-scope硬编码日志也已改为真实per-epoch结果。
- 最新服务器Traceback仍执行旧测试`test_early_stop_rolls_back_without_crashing`并要求`rows == []`，而当前测试要求不调用standard early-stop、返回两条epoch row并logical commit一次。该Traceback证明服务器测试脚本尚未同步到当前工作区，不能作为恢复旧production行为的依据。
- 有效性判断：run108训练本身有效，但pending事务未产生净更新；它验证了snapshot、stale/mass计算和外层rollback的production触发，未验证两epoch成功commit或行为收益。来源：run108 saved config、eval、pending update/cohort及transaction CSV、源码与测试脚本检查；来源：本次历史对话整理。

### run109--run114：pending事务从首次错误到中间完整版本

#### 实验目的、直接对照与代码版本

- run109承接run108后的pair-only control/population修复；run110修复run109暴露的人口分解问题。run112是pending=False/TTL=0完整对照；run113、run114分别记录transaction diagnostics v4/v5的True/4中间状态。run111没有落盘目录，只是历史编号/口头误用。

#### 配置、训练完整性与关键日志

- run109--run114可定位run均为seed=1、目标200k、`model_dir`空。run109在164.8k报`RuntimeError: pair/non-pair base-factor population split failed to reconstruct the full transaction`，最后完整eval为160k，故部分有效。run110、run112、run113、run114均到200k并写terminal checkpoint，无相同阻断性错误。
- 落盘final reward/win/capture分别为：run110 `-1.309/0.2/2.1`；run112 `2.177/0.3/2.9`；run113 `-0.875/0.2/2.5`；run114 `-0.504/0.2/2.3`。run112 pending=False/0，其他已列run为True/4。

#### 有效性、结论与来源

- run109只用于人口分解错误定位；run110证明该错误被处理到可完成训练。run112是后续True/4链的pending-off基线；run113/114是可运行的中间机制状态，当前对话没有足够源码差异把其性能归到单一子机制。
- 来源：各run config、eval CSV、transaction CSV、console terminal/Traceback与目录检查；来源：本次历史对话整理。

### run115

#### 实验目的、直接对照与代码机制版本

- v7完整200k验证，直接承接run114。目标是过滤forced/non-actionable pair，并在ordinary路径执行逐target保护；pending pair-only当时仍以aggregate pair约束为主。

#### 配置与训练完整性

- fresh seed=1、True/4、200k、diagnostics v7；terminal checkpoint存在，无中途异常或更新范数塌缩。

#### 关键事务、指标与漏斗

- ordinary逐target为46/46正确，candidate真实下降保持。163.2k multi-target pending cohort出现20 correct、20 reverse，但aggregate仍为正，直接证明aggregate正确不能替代逐target正确。
- final reward0.239、last5约0.528、win0.3、capture2.6；全程failed-capture约58.20%、matched18.75%、capture-episode-win26.88%。

#### 有效性、结论、经验与来源

- run115完整有效；forced过滤和ordinary逐target层通过，最早失败层是pending逐target约束。来源：transaction/pending CSV、eval/capture统计、console与本次历史对话整理。

### run116

#### 实验目的、直接对照与代码机制版本

- v11直接修复run115 pending aggregate masking：pending与ordinary均逐target；current priorities先修复，再筛选历史lifecycle，并做exact revalidation。

#### 配置与训练完整性

- 历史记录包含至少两个未落盘失败执行和一个正式完整attempt。正式attempt为fresh seed=1、True/4、200k、diagnostics v11、training complete；性能分析只使用正式attempt。

#### 关键事务、指标与漏斗

- pending及ordinary target-epoch共44/44正确，reverse=0、zero=0、rollback=0，无更新范数塌缩。final1.036、last5约0.298、last10约-0.151、win0.5、capture2.8；全程failed-capture55.89%、matched16.73%、capture-episode-win30.11%。

#### 有效性、结论、经验与来源

- v11闭合了exact target score与事务原子性，但性能没有超过run112，160--200k仍振荡。最早未闭合层转为selection-boundary margin→rank→active。来源：transaction/pending/capture/eval CSV、console、terminal checkpoint与本次历史对话整理。

### run117

#### 实验目的、直接对照与代码机制版本

- v12以run116正式attempt为直接正确性对照，增加统一norm-scaled float32方向容差、float32严格下降floor、run116 fixture/replay、组合preflight和正式launcher前独立进程验证。

#### 配置与训练完整性

- fresh seed=1、True/4、200k、diagnostics v12；step=0 preflight先于正式模型/optimizer/训练CSV，formal RNG未污染，terminal checkpoint完整，无Traceback/NaN/Inf或CSV污染。

#### 关键事务、指标与漏斗

- 与run116正式轨迹一致；44/44 exact正确、reverse0、zero0。final1.036、last5约0.298、last10约-0.151、win0.5、capture2.8；全程failed-capture55.89%、matched16.73%、capture-episode-win30.11%。

#### 有效性、结论、经验与来源

- v12成功范围是启动前fail-fast、RNG隔离和事务正确性；当时没有pair-specific boundary competitor、margin、rank或active因果证据，不能写作selection性能成功。来源：preflight/console、config、manifest、transaction/pending CSV、checkpoint与本次历史对话整理。

### run118

#### 实验目的、直接对照与代码机制版本

- v13相对run117新增production selection-boundary competitor、target/competitor score、signed margin、boundary deficit、canonical rank、crossing和active诊断。

#### 配置与训练完整性

- fresh seed=1、True/4、目标200k；实际到约182.4k后中断，最后完整eval为180k。不得计算正式200k final、last5或last10。

#### 关键事务、指标与漏斗

- 26/26 exact正确，26/26 boundary margin改善，reverse=0、zero=0，median margin delta约0.008966、worst约0.000148，competitor区域切换2；crossing/promotion/eviction=0。14个target有正deficit，12个更新前deficit为0；正deficit总体median reduction fraction约0.4052%，pending约1.0712%，ordinary约0.0438%。
- 中断根因：boundary非线性修正改变最终位移后只复核boundary，没有在同一回溯中联合复核candidate和已接受lifecycle；optimizer.step后终检暴露失效。

#### 有效性、结论、经验与来源

- run118是部分有效机制实验：首次闭合exact score→boundary margin方向，但事务联合exact存在缺口，且margin远小于deficit。来源：boundary/transaction CSV、console、eval CSV与本次历史对话整理。

### run119

#### 实验目的、直接对照与代码机制版本

- v14直接修复run118的boundary/candidate/lifecycle联合真实forward backtracking和完整rollback；run119只承担20k阶段正确性验证。

#### 配置与训练完整性

- 独立fresh seed=1、True/4、20k、diagnostics v14；terminal complete，无Traceback/NaN/Inf，preflight在正式run前完成并保持formal RNG。

#### 关键事务、指标与漏斗

- 20k内没有actionable strict-pair boundary target，boundary零样本未写伪造0行；公共训练轨迹与run117前20k一致。20k reward-5.187、win0.1、capture0.8仅用于冒烟判断。

#### 有效性、结论、经验与来源

- 阶段1通过，证明no-target路径、schema、launcher和RNG隔离；未触发boundary分支，不能据此证明v14边界性能。来源：console/config/manifest/CSV/checkpoint/preflight与本次历史对话整理。

### run120

#### 实验目的、直接对照与代码机制版本

- v15相对run118/v14引入deficit-aware budget；总boundary预算不提高，zero-deficit target只保留strict floor，余额分给deficit-bearing target。

#### 配置与训练完整性

- fresh seed=1、True/4、160k、diagnostics v15、terminal complete，无中途异常。

#### 关键事务、指标与漏斗

- required improvement完成率约99.8%，无backtracking缩步、competitor switch或范数塌缩；deficit-bearing target reduction fraction中位约0.813%。两个positive deficit target仍共享有限预算，deficit约1.46--2.22而required仅约0.010--0.018，全部affordable crossing=0、rank3→3、crossing/promotion/eviction=0。
- 160k reward0.475、win0.3、capture2.8；final-eval topology重算failed-capture71.43%、matched17.86%、capture-episode-win30%。

#### 有效性、结论、经验与来源

- v15真实改善预算利用和required兑现，但没有产生rank crossing；行为不可归因于active链。来源：summary、boundary/transaction CSV、eval/topology CSV与本次历史对话整理。

### run121

#### 实验目的、直接对照与代码机制版本

- v16在run120总预算内做最近boundary优先和water-filling，意图避免多个不可达exposure之间平均碎片化。

#### 配置与训练完整性

- fresh seed=1、True/4、160k、diagnostics v16、schema v3、terminal complete。

#### 关键事务、指标与漏斗

- 最近exposure实际获得约92%--99%预算，但同identity其他exposure仍独立约束。150.4k `0-2(+)`事务scale为0.5/0.03125，主target completion约50.3%/3.12%，post deficit0.225422、rank11→10、crossing0。总体median reduction约0.933%，只小幅高于run120。
- 160k reward-3.073、win0.3、capture2.4。无active变化，不能把行为退化归因于v16机制闭合。

#### 有效性、结论、经验与来源

- schema预算守恒和exposure级集中通过；最早断点转为同identity多exposure联合可行域。来源：summary、schema v3 boundary/transaction CSV、console与本次历史对话整理。

### run122

#### 实验目的、直接对照与代码机制版本

- v17将同一identity/sign的member Jacobian和signed-margin delta求和，以一个group进展预算替代多个独立extra预算。

#### 配置与训练完整性

- fresh seed=1、True/4、目标160k、diagnostics v17/schema v4；149.6k ordinary strict-pair事务中断，最后完整evaluation为140k，故没有160k性能结论。

#### 关键事务、指标与漏斗

- 已提交事务预算守恒，但group completion约99.96%、最近boundary member仅约28.9%，容易member掩盖关键member。失败事务在optimizer.step后联合exact无可行scale，外层原子rollback恢复参数、optimizer及事务状态；失败事务不作为commit。

#### 有效性、结论、经验与来源

- 部分有效；identity级sum产生aggregate masking，且真实149.6k几何未被旧preflight覆盖。来源：console、transaction/group/boundary CSV、最后eval与本次历史对话整理。

### run123

#### 实验目的、直接对照与代码机制版本

- v18以唯一positive-deficit最小member作为progress member，group required/actual都使用该member，其他member只承担exact与non-regression。

#### 配置与训练完整性

- fresh seed=1、True/4、160k、diagnostics v18/schema v5、terminal complete；安全越过run122失败区间。

#### 关键事务、指标与漏斗

- 150.4k seq720 required0.065799、actual0.033069、completion50.26%、scale0.5；seq721 required0.101410、actual0.003167、completion3.12%、scale0.03125。candidate/lifecycle均无、competitor switch0，post deficit0.225422、rank11→10、crossing0。
- 160k reward-3.073、win0.3、capture2.4，与run121几乎一致；无active变化，不能作行为因果归因。

#### 有效性、结论、经验与来源

- v18修复aggregate masking与required/actual口径；最早断点成为dyadic-only安全scale损失。来源：summary、schema v5 boundary/group/transaction CSV、console与本次历史对话整理。

### run124

#### 实验目的、直接对照与代码机制版本

- v19相对run123只修改联合exact scale搜索：halving bracket后12次真实production refinement，安全条件不变。

#### 配置与训练完整性

- fresh seed=1、True/4、160k、diagnostics v19、terminal complete；全部38个member exact/boundary为38/0/0，无rollback或范数塌缩。

#### 关键事务、指标与漏斗

- seq720 scale由0.5恢复到0.691650，actual0.045666、completion69.40%；seq721最大安全scale0.024597、actual约0.003198、completion2.46%。最近member deficit0.261658→0.212795、rank11→10、crossing0。candidate/lifecycle无、competitor switch0；安全端最小gap指向non-progress boundary约束。
- 160k reward1.603、win0.4、capture3.1；final-eval failed-capture48.39%、matched19.35%、capture-episode-win40%。无crossing/active，行为改善不能归因于boundary闭环。

#### 有效性、结论、经验与来源

- v19证明固定方向上近最大安全scale可被确定性恢复；seq721说明固定方向本身仍受真实约束限制。来源：summary、transaction/boundary CSV、eval/topology CSV、console与本次历史对话整理。

### run125

#### 实验目的、直接对照与代码机制版本

- v20在run124同预算下构造5个Adam-based progress fraction，每个候选独立搜索并按original-required progress选优。

#### 配置与训练完整性

- fresh seed=1、True/4、160k、diagnostics v20/schema v6、terminal complete。

#### 关键事务、指标与漏斗

- seq720仍选full、scale0.691650、completion69.40%；seq721选择fraction0.25、scale1、actual0.032033、original completion24.63%，相对run124该epoch actual约提高10倍。最近member post deficit0.183960，crossing/promotion/eviction=0。
- 160k reward0.743、win0.2、capture2.7。用户本轮摘要写failed-capture77.78%、matched22.22%、capture-episode-win10%；实际160k final evaluation topology CSV按“失败episode中的capture数/全部capture数、matched event/全部capture、含capture episode中成功episode比例”重算为66.67%、22.22%、20%。两组口径冲突均保留，正式落盘重算值用于文件证据结论。

#### 有效性、结论、经验与来源

- v20证明降低fraction可产生更高真实actual，但没有crossing；schema v6不能独立重建全部未选候选几何。来源：summary、boundary/transaction/eval/topology CSV、用户摘要冲突与本次历史对话整理。

### run126

#### 实验目的、直接对照与代码机制版本

- v21直接对照run125，在五个Adam候选之外增加三个progress-directed候选及全候选诊断，用于验证候选几何、隔离、safe search和selection。

#### 配置与源码状态

- config确认fresh seed=1、True/4、160k；运行时diagnostics v21、boundary schema v7、candidate schema v1。source manifest SHA256为`78ab7e7844eac8fb0064208e9f53cc9c103fe5099ca8756c21046b0f496388cb`，审计时当前文件曾与run126 manifest逐文件一致；本次对话此前完成v22修改后工作区不再等同于run126源码。

#### 训练完整性

- `logs/summary.json`确认target/actual均160000、training_complete=true、checkpoint kind terminal、last eval160k；单attempt，transaction sequence 0--763连续，无step回退、CSV追加、Traceback、NaN或Inf。TensorBoard 1372个scalar tag，最后reward/win与CSV一致。训练环境PyTorch1.3.1、CUDA不可用并明确skip。

#### 关键事务、日志与机制漏斗

- 144.8k seq693首次选择新增progress-directed方向并与run125参数分叉。6个strict-pair epoch各有8候选，另外2个无适用多member progress组的epoch为单Adam。progress-directed相对full cosine为0.939818--0.983500，active-set会变化；候选不是同方向缩放。选择规则按worst original completion、mean、scale、ordinal，未发现以scale掩盖actual。
- 34/34 member exact与boundary正确，reverse0、zero0；14个正deficit membermedian reduction2.1198%，pending0.9486%、ordinary2.1198%。最近member post deficit0.105473、rank1→1；crossing/promotion/eviction=0。
- 160k reward0.391、win0.2、capture2.5；final-eval topology重算failed-capture68%、matched28%、capture-episode-win20%。0--160k全evaluation累计口径为failed60.89%、matched16.83%、capture-episode-win24%。无active变化，因此行为差异不能归因于v21机制闭合。

#### 与直接对照、有效性与最终结论

- 相比run125，v21验证了正式训练中的8候选、真实几何差异和original-required selection，并把总体/ordinary deficit reduction提高；但不同identity/context使关键数字不是严格同样本反事实。run126完整有效，最后稳定通过层是boundary progress与部分deficit reduction，第一个未闭合层仍是member-level rank crossing。

#### 已验证经验、暴露问题与来源

- run126同时暴露v21 progress seed包含zero-extra-budget identity代表member；seq736零预算`1-3(+)`仍旋转方向并成为一个低fraction候选的boundary limiter。当前v22只过滤progress seed，不删除任何member硬约束；v22仅有代码/测试证据，无正式run性能结论。
- run126没有可比selection context下的commit后两次ordinary margin retention记录，rank-deficit retention需求仍未确认。
- 来源：run126 console、config、summary、eval/topology、transaction/boundary/all-candidate CSV、TensorBoard、manifest、terminal checkpoint、源码与本次历史对话整理。

### run147--run153：joint exploration、freeze countdown与局部greedy floor

- run148建立了旧33/57输入的60k正式基线：268 episodes、58 capture episodes、67 captures、1 win。basic capture虽随joint exploration改善，但first capture后切换到另一只prey失败；7个multi-capture no-win的distinct interval median=132，均超过24步。
- run149确认`food_frozen_time`是remaining countdown，并以两个per-prey feature把local/state改为35/59。production反事实和完整forward path证明offscreen frozen与offscreen active已不再alias，remaining=24与1能真实改变最终policy/Q输入。run151最终checkpoint进一步证明feature会改变Q与greedy joint action。
- run151 60k正式行为为89 first captures、2 strict distinct、2 wins。non-explore/greedy在post-capture窗口的second-prey单步progress为正，explore近零或负；因此run152只在one-frozen+one-active窗口把full-greedy joint branch概率下限设为25%。run152 production中52/220=23.636%，与二项随机波动一致，且capture后下一decision立即生效。
- run153在60k把strict distinct由run151的2增至4、win由2增至4，interval median由15.5降至7；first capture数由89降至81，故收益不是basic capture数量，而是first→distinct转化由2.25%增至4.94%。

### run154--run156：transaction fail-loud与长程行为定位

- run154/run155均因`standard_adam pair transaction did not improve every exact target`中断，不能用于性能。相关链还出现过origin基线错位、重复capture event竞争同一strict pair target、zero adjacency gradient与minimum-dot constraints infeasible。修复原则保持严格：统一transaction origin、scale-zero直接copy snapshot、唯一event/target消费、zero gradient不消费、失败外层原子回滚；没有用skip/no-op或放宽boundary掩盖失败。
- run156在修复后的代码上fresh 180k完整训练：868 formal episodes、550 capture episodes、854 captures、27 strict distinct、38 wins，final eval reward/win/capture为1.943/0.3/2.8。first→distinct约4.91%，与run153短程改善同方向但仍低。
- 成功/失败对齐发现最早稳定分叉在first capture时：成功轨迹通常已有其他alive player靠近remaining prey；terminal transition进入uniform replay没有系统性欠采样。该证据把第一根因从输入、reward、post-capture floor与replay接线后移到completion credit能否及时训练pre-capture双目标分工。

### run157--run161：completion credit、terminal lane与objective population

- run157的unconditional n=24从第一个optimizer起就改变所有普通transition，target std约扩大5倍、loss/TD/gradient膨胀、clip约44%、capture episodes仅12、wins0。该失败直接证明累计全窗口dense reward不是completion-specific credit。
- run158的terminal-gated实现满足：无marker严格legacy one-step；marker窗口最多24步；不跨episode/done；`continue_after_success`以win marker截断；bootstrap使用对齐future state/RNN timestep。80k恢复到142 capture episodes、180 captures、5 wins，真实target gain主要来自completion bonus，但仅7次uniform terminal sample/168 gated transitions。
- run159的round-robin lane把terminal credit利用量提高约6.57倍，但weight=1.0使forced 24-transition auxiliary MSE支配uniform objective；首次forced sample后loss与RNN gradient领先于greedy progress和capture退化。run159 80k仅119 capture episodes、1 strict distinct、1 win。
- run160将forced weight设为0.10，真实optimizer中weighted contribution非零但不再主导。60k恢复85 capture episodes、102 captures、2 distinct、2 wins；Q/RNN无clip，first-capture geometry和early second-prey progress改善。run161扩展到80k后Q仍稳定、141 capture episodes、169 captures、3 distinct、3 wins；未再出现run159的loss population失衡。

### run162：bounded post-capture exploration的80k正式验证

- 有效性：fresh seed=1、target/actual=80k、training_complete=true、terminal checkpoint存在；q-target contract v4配置n=24、terminal-gated lane、aux0.10；exploration contract v4配置228k、floor0.25、`post_capture_explore_max_random_agents=1`。无Traceback、RuntimeError、NaN/Inf、strict-pair failure、origin failure或rollback。
- 干净前缀：首次bounded eligible explore发生在step7849；run161/run162此前joint episode、post-capture、capture、reward和Q公共字段一致，说明新helper在不eligible时未额外消费RNG或改变行为。
- 机制曝光：2877 eligible decisions中2124 explore、753 non-explore，explore fraction0.7383；run161为0.7364。2124次eligible explore均只替换1个alive slot，slot/action覆盖完整，unique joint ratio0.928；greedy-final Hamming median从4降至1、mean从4.2407降至0.8583。未改变explore概率、available-action legality、dead-slot stay或non-explore greedy contract。
- transition机制：good-start failure中topology-clean explore progress由-0.1162改善到-0.0522，retreat fraction由33.42%降至23.62%；non-explore保持正progress。全体first-capture episode的nearest-player +24 progress由run161的-0.4722改善到run162的+0.5042。
- 行为结果：80k formal 368 episodes；run161→run162的capture为169→204，strict later-distinct为3→16，training win为3→23。16个later-distinct interval为21、7、3、12、11、24、19、13、7、24、16、8、12、19、10、11，全部<=24，mean13.5625、median12。final eval从0.584/0.1/1.6提高到1.588/0.3/2.7。
- failure cohort没有支持“必须固定同一B-side agent”：run162 strict-success到+24的initial-nearest identity retention约0.533，good-start failure约0.526，成功窗口反而有更多身份切换，说明team substitution可用。随机slot命中initial critical slot在残余good-start failure中关联更差即时progress，但全cohort未来4/8/12步该关系不稳，证据不足以引入oracle/critical-slot保护。
- 当前残余是pre-capture far-start：remaining-prey nearest distance>=8的54/141 first captures中strict distinct为0。现有bounded机制只在post-capture起效，且没有足够pre-capture per-agent Q/factor/action历史支持安全修改，因此run162之后没有继续改production代码。

## 7. 已验证成功的尝试

- **candidate residual的双Adam隔离作为实现约束被建立。** run83以后主路径明确区分`adj_optimizer`和`candidate_residual_optimizer`，要求residual前`.grad=None`、无candidate梯度参数跳过、inactive参数零位移、事务性rollback/state sync和双state持久化。成功范围是代码设计与后续完整训练可运行；本次可见资料没有逐run state差分统计，不能外推为residual改善了reward。
- **trusted-control错接与ratio聚合错误均被识别并修复到数据流层。** raw/trusted控制、early-stop、recent-window与population-total `sum(n)/sum(d)` 的语义已成为主路径要求。run87显示window确有恢复，run89显示新聚合改变训练轨迹；成功范围限于控制语义，不是性能成功。
- **pair显式credit的身份局部性和optimizer-step守恒得到验证。** run92取消pair进入shared `f_advt`，run94实现one-sided optimizer cohort为零，run95的42个非零pair transaction全部双边且正负质量相等、最大中心误差约3.725e-9。该成功证明标签与质量守恒链，不证明factor score、active、capture或win已经改善。
- **run95的atomic partition批人口错误已由源码审计修复。** `utils/pair_credit.py`、`utils/adj_buffer.py`、`runner/base_runner.py`、相应cohort合成测试和训练脚本已同步更新为support v5。成功范围是划分语义与本地定向测试；尚未由服务器PyTorch1.3.1训练验证。

- **episode 内动态 lifecycle 与掩码接线成功。** 对应初始 20k；训练/eval 都记录 leave=4、join=2、recover=4、recovery completion=1.0，人数范围 2--6，且无 invalid factor。成功范围是动态事件、状态恢复和固定槽位掩码可运行；不外推为策略性能成功。
- **动态基线脚本建立并保持环境设置一致。** 对应 VDN/QMIX/QPLEX 动态脚本；公共 shock、人数、episode 长度配置可由包装脚本/共同 baseline 脚本传入。成功范围是实验入口与环境对齐；各算法网络公平性仍取决于每次保存 config。
- **QMIX 旧 PyTorch 兼容性错误被修复至可训练。** 初始报错为 `torch.nan_to_num` 不存在；之后 QMIX 完成 200k 和 2M 训练。成功范围是运行兼容性，不意味着 QMIX 参数或性能最优。
- **图合法性与动态结构诊断被接入。** 早期日志显示 invalid factor=0，并持续记录有效/空因子、阶数与平均阶数。成功范围是可观测性。
- **order3 quota/band/gate 等机制确实能改变结构分布。** run26 清楚显示三阶比例被推高；run30 结构回到 o2=0.242、o3=0.442 且 gap=0.003。成功范围仅为控制/记录结构分布，不是性能成功。
- **run30 相对近期多轮版本有局部性能改善。** final=1.979、win=0.5、best=4.684，且 train/eval o3 gap 很小。成功范围是该单 seed 200k 的相对结果；不能外推为对 run18、QMIX 2M 或全部基线的稳定领先。
- **PPO guard、stale trust、recent window 的接线和日志能力被加入。** 当前源码检索证实参数、训练脚本、runner 指标和预检路径存在。成功范围是实现存在；除 run37/38 的部分触发外，性能影响未确认。
- **参数解析与预检已扩展到新增图 credit 路径。** 当前 `train_wolfpack.py` 已识别 recent-window、delayed future-match、capture-to-win 与 pair/triplet complementary 参数；`validate_sddfg_dynamic_graph.py` 也有相应参数打印、断言和扩展 batch 检查。成功范围是当前源码接线和可观测性，不能反推历史启动失败的具体修复提交或性能效果。
- **run49 完成 2M 且显示长程能力。** 其 final=6.296、win=0.9、capture=4.5，说明该代码组合在单 seed 下可完成长程训练并在末段产生较高成功率。成功范围限于该完整 run；其 reward 滑动指标低于后来的run68和QMIX final，不构成单机制证明。
- **capture-to-win 与 pair/triplet complementary 的日志、缓冲区和 batch 接线得到实际运行证据。** run50/run51 保存 config、console 和 train-adj 中相应字段存在，run51 的 pair/delayed/future 信号实际非零。成功范围是启用与记录；run50/run51 性能均未成功。
- **pair credit 的过宽激活被结构性修复。** run51 active fraction约92.3%，run52--run57降到约0.74%--1.14%；普通正reward、offset0和floor不再造成广播。成功范围是信用约束正确性，不能外推为reward成功。
- **episode outcome中心化和质量守恒得到数学验证。** support v3/final cohort阶段的center error为0，event/episode总量不随capture或factor数量增长；成功范围是统计总体和质量守恒。
- **两人capture可精确映射到order2。** run58因只允许order3而matched=0；run59 exact order2后matched=8.89%。成功范围是身份和阶数接线，active覆盖仍不足。
- **稀有outcome跨update无限重复被修复。** run61 support v2以slot-generation限制跨update一次，同update多epoch复用；成功范围是去重和原子补全，随后exhausted=28说明覆盖并未一并解决。
- **最终optimizer cohort centering真实生效。** run62 class-complete/center-valid窗口为7、center error=0；不能外推为outcome信号足够密集。
- **graph advantage replay写入缺陷被真实对照识别。** run62/run63 eval轨迹完全相同而manifest不同，run64修复stored source后轨迹改变。成功范围是训练源接线，不代表run64性能提升。
- **target-local factor loss归一化得到训练证据。** run65保持non-target不受local delta且滑动reward改善；final/win/capture未同步，故仅属于loss口径成功。
- **candidate identity监督逐层接通。** run66--run69依次证明candidate字段、signed loss、current differentiable conditional probability、target-transition normalization真实进入训练。run69 candidate/graph loss约0.801%，成功修复无关transition稀释，但optimizer方向仍错。
- **欧氏梯度冲突投影改善即时candidate方向。** run70正/负optimizer后概率正确率84.38%/82.14%，并取得final3.555、win0.8、capture3.4。成功范围是该run的即时方向和行为表现；rank、active覆盖与跨update保持未解决。
- **run68完成当前最强SDDFG长程终值。** final6.718、last5 6.610、last10 6.527、capture4.6，证明该代码组合在单seed 2M可形成高回报；win0.6低于QMIX0.8，且前200k弱，不能写为全面超过基线或单机制成功。
- **run74把主要异常定位到正确层级。** 欧氏逐约束dot在容差内而Adam realized displacement仍违反，rollback随之上升；成功范围是因果定位和诊断分层，不是训练性能成功。
- **run75验证lifecycle v4修复了run74的实际位移/rollback错误。** 实际位移投影修正18个日志窗口，修正后负约束计数为0，rollback为0，policy version在242个adj update中全部推进。成功范围是实现正确性和事务完整性；run75未过200k性能门槛。
- **age1/5/10 observation archive真实进入训练。** run75分别记录82/76/74个可观测样本，证明过期后只读诊断档案可用。不同age的cohort不同，不能据保持率数值直接推断单调长期记忆。
- **target-bearing诊断已按当前真实target修复。** run75 step155200暴露cache门控漏记；本轮回归测试保证无旧cache时真实target仍记为1。成功范围仅为日志统计正确性。
- **support v6修复了class-complete transaction的人口语义。** run97起class-complete使用一个完整selected population和一个standard Adam transaction；run100的per-epoch日志可在同一人口上追踪combined、Adam与score。成功范围是训练人口与诊断单位一致；run97的400k性能没有超过历史强基线。
- **optimizer diagnostics v2/v3建立了从objective到exact score的闭环审计。** run100区分了combined反向、Adam moment反向、clip、final commit与score；run102进一步重构graph/base/outcome/pair/candidate/entropy及projection。run98--run102公共轨迹证明这些诊断只读；成功范围是根因可定位性。
- **graph advantage source责任范围错误已修复。** run102用两个独立反例和源码确认identity-local/local-delayed credit污染graph PPO；run103起graph只消费replay graph-return，factor residual精确重构原advantage，contamination为0。成功范围是source隔离正确，run103没有证明性能提升。
- **pair-evidence funnel与candidate same-population诊断排除了多条错误解释。** run104--run106确认成功capture存在、身份/provenance/terminal join正确、candidate正target进入训练、gradient/Adam/score方向正确；rank不变由更新小于合法next-better gap解释。成功范围是逐层定位，不是active或性能已恢复。
- **provenance-complete candidate evidence实现了独立行为证据计数。** run107的102行全部通过provenance/identity/quality contract，PPO epoch和replay exposure不再伪装为独立generation。成功范围是event级可复现审计。
- **pair时窗不重叠被真实数据重建。** run107确认positive/negative strict generation为2/36但recent-window overlap为0；horizon v2精确给出4/7 update的结构机会，排除了support漏选和shared-used提前消费。
- **bounded pending production基础与外层rollback在run108真实触发。** immutable snapshot、transaction-time stale/mass、pair-only scope和outer atomic state均进入真实144.8k attempt；1/2 epoch中止后参数、optimizer、RNG、lifecycle与pending完整恢复，run108/run107公共轨迹一致。成功范围是安全回滚，不是pair transaction成功或性能成功。
- **run108的pair-only control scope错误已被定位并在当前工作区修正。** 当前pair-only transaction以exact nonzero pair target population记录控制量，不适用standard graph/base early-stop，且仍要求所有配置epoch完成后才能logical commit。普通adjacency PPO路径保持不变；最新同步后的服务器回归尚未确认。
- **v12逐target事务与step=0 preflight得到完整训练验证。** run117中pending/ordinary target-epoch 44/44正确，reverse0、zero0；preflight先于正式模型/optimizer/CSV并保持formal RNG。成功范围是事务正确性，不是selection性能。
- **v13首次把exact score连接到真实production boundary margin。** run118的26/26 exact与26/26 margin均正确，证明score→boundary方向闭合；crossing/active仍为0，且该run因联合复核缺口中断。
- **v14 no-target隔离得到短程验证。** run119与run117前20k公共轨迹一致，boundary零样本不伪造行，证明诊断、联合回溯代码和preflight没有污染普通训练路径。
- **v15--v16预算回收与集中真实生效。** zero-deficit target不再消耗大额进展预算，run120 required completion约99.8%；run121最近exposure获约92%--99%预算。成功范围是budget accounting和分配，不是crossing。
- **v18修复identity aggregate masking。** run123中group required/actual统一到唯一nearest progress member，安全越过run122失败区间；其他member exact/non-regression继续成立。
- **v19恢复固定方向上被dyadic搜索丢失的安全位移。** run124 seq720 scale从0.5提高到0.691650、completion到69.40%；成功范围是固定方向最大安全scale，不是rank crossing。
- **v20证明同预算下不同联合方向可提高真实progress。** run125 seq721选择0.25方向后actual约提高10倍，completion由2.46%提高到24.63%；仍无crossing。
- **v21在正式训练中生成几何不同的8候选并按original-required选择。** run126的progress-directed cosine显著低于1，active-set变化，候选选择与全候选CSV可审计；34/34 exact/boundary正确。成功范围止于progress与部分deficit reduction。
- **v22过滤zero-budget identity的progress seed。** 当前代码仍保留其exact与boundary硬约束，只把extra progress方向限定为正extra-budget member；静态、反序隔离、CPU/CUDA production、rollback、checkpoint、RNG与no-target测试通过。该状态尚无正式训练性能证据。

- **freeze countdown真实解决了state aliasing并被Q使用。** run149闭合35/59 production数据链，run151 counterfactual在118/200 context改变greedy joint action；该输入不是仅增加维度。
- **局部greedy floor改善first→distinct，而非basic capture。** run151→run153的strict distinct 2→4、wins 2→4、interval median15.5→7，且floor production timing/RNG合法。
- **terminal-gated return与0.10 weighted lane同时保留completion signal和uniform Q稳定性。** run158闭合gate，run160/161闭合forced population weighting；与run157/run159的失败形成直接反事实。
- **bounded post-capture exploration在run162形成目前最强的短程真实行为链。** branch概率不变、Hamming破坏范围4→1、explore retreat下降、+24 progress转正、strict distinct3→16、wins3→23，且capture同步169→204。

## 8. 未成功或存在问题的尝试

- **candidate v11/v12与lifecycle v9/v10未形成已证实的性能收益。** run82--run85包含多项耦合改动且当前对话缺少完整量化对照；不能把独立residual Adam、行为进展门控或双Adam checkpoint视为已恢复run70/run78的证据。
- **trusted-control修复没有自动带来中程基线优势。** run86虽在240k达到3.487但260k回撤至-0.648；run87的2M滑动回报仍低于run68；run89的200k/400k为1.080/1.238。控制人口正确与策略价值是不同层次。
- **outcome-conditioned signed pair credit未稳定改善capture质量。** run91的200k capture降至2.0；run92、run93、run94和run95均出现早期或中期高点后回撤。run90的失败capture比例43.8%、capture episode win率44.4%，与run70/run78差距大；run95在120k局部达到capture4.2/win0.7后未保持。
- **pair-local normalization v2曾放大错误的单边optimizer cohort。** run93的v2/v1尺度约369.76倍、pair/base loss约40.1%，但73个非零pair update中52个单边；因此不能将“target-bearing分母”本身写作性能成功或继续用系数掩盖cohort构造错误。
- **atomic transaction初版修复守恒但改变base PPO batch人口。** run95的pair chunks固定第0分区造成2/4、2/3等不均衡transaction；这是已确认的性能相关bug。generic capture support优先级是否还会引入训练预算问题尚未被充分证实，不应写成已修复根因。
- **候选到active的闭环持续弱。** run95 positive crossing约0.0595%、正rank改善约3.115%，尽管identity match约18.39%；候选/score局部指标不足以证明可改善active graph和行为。

- **固定规模 stage3 难以体现可扩展优势。** 历史上 VDN/QPLEX 在固定规模接近满分，导致 SDDFG 差异化不足；这是转向 episode 内动态人数的背景，不是对所有固定规模任务的普适否定。
- **仅提高或强制 order3 比例无效。** run26 的 quota/order-aware credit 使 o3=0.628、triplet fraction=0.772，非空 triplet 约0.92，却未形成捕获收益；结构数量与结构质量被混淆。
- **pair-heavy 回调也没有形成优势。** run21/run22 的 order3 很低且 reward 低于 run18，说明单向压低三阶不能解释性能改善。
- **bonus、entropy/temperature、band、quota、greedy、gate 等经常成组变化。** 多变量混杂使得无法将 run23--run35 的结果归因到某一参数或单一模块。
- **absolute order3 credit gate 存在误压制风险。** 历史分析指出，仅以 order3 loss 的绝对正负判断会将并非相对更差的三阶一并压制；随后改为 relative gate。其性能改善未被稳定证实。
- **relative gate、synergy scorer、positive-only / graph-positive promotion、advantage-aware scorer 未证明性能成功。** 相关 run31--run35 的精确部分指标缺失，后续 run37/38 仍低于 run30；只能说明这些机制没有形成已证实的稳定收益闭环。
- **PPO high-clamp early stop 未证明正效应。** run37 真正启用后 reward、win、capture 均低于 run30；这可能涉及欠更新、耦合或其他变量，但当前证据不能归因于某一个原因。
- **stale trust 未证明正效应。** run38 trusted clamp 显著低于 raw clamp，但 reward/last5/best 更低；该结果说明数值被降权后较小不能等价于策略更稳定。
- **捕获、胜率和 reward 多次分离。** 多个 run 中 capture events 约 2.6--2.9 而 win rate 或后期 reward 不稳定；因此只提高 capture 不能作为成功结论。
- **预检自身曾阻断训练。** `math.comb`、PyTorch `nan_to_num`、finite-gradient 断言、greedy cap 假设均出现过报错/失败；预检不能与训练脚本配置脱节。
- **新增参数曾未被 parser 接受。** `--adj_recent_episode_window` 与 `--adj_delayed_triplet_credit_require_future_match` 分别在不同历史阶段导致 Unknown command line arguments；参数在脚本变量中出现不等于训练入口实际接收。
- **旧 PyTorch API 与扩展 batch 形状曾阻断 SDDFG 训练。** run40 的 `torch.minimum`、run48 的 400/6 mask 维度不匹配均发生在训练前或早期；这些 run 没有可比较性能。后续可完成 run49 只能说明阻断未重现于该组合，不能概括到所有环境与 batch 形状。
- **delayed/future/success/graph-return credit 未证明性能闭环。** run42 有高峰值但 final win=0.4；run50 的 capture-to-win credit 在 200k 为零；run51 的 future/delayed/pair 指标大量非零但 reward、capture、win 全部较低。它们同时含多项机制，不能进行单一原因归属。
- **capture-to-win 信号可能过稀。** run50 的 200k capture-to-win mean、active fraction、quality gate 均为零，且 final/last5/capture 未过 run30 门槛。该事实只表明该 run 的末段没有该信号，不能外推到每个 rollout。
- **pair pursuit 信号可能过宽。** run51 的 pair pursuit active fraction 末值约 0.943，而 final reward 为 -0.864、win=0.2、capture=1.9；覆盖率高与性能差并存，不能把覆盖率当作协同质量。
- **centered outcome 的“episode级零均值”曾被factor展开破坏。** run56之后检查发现同一episode多个capture/triplet会重复计数；run57改为episode总量分摊。历史run56的高reward不能反证该数学缺陷不存在。
- **capture identity最初假定所有真实capture都能表示为order3。** run58的41个事件全部为两参与者，matched=0，导致outcome/local delta完全断流；该假定已被环境事件证伪。
- **精确identity减少错误广播后暴露active coverage瓶颈。** run59--run75 active identity match长期偏低，candidate-only从历史约75.89%上升到run75按事件计数汇总的93.07%；精确匹配正确不等于监督覆盖充分。
- **support v1无限复用稀有episode。** run60的augmentation可反复使用同一episode；这会把少数样本误写为持续独立证据。
- **support v2一次性消费造成快速耗尽。** run61 enabled13而exhausted28；无限复用被修复后，outcome监督仍可长期关闭。
- **support/cohort中心化正确但信号稀疏。** run62仅7个有效窗口；零中心不能代替训练覆盖。
- **graph-confidence代码存在但replay源未写入。** run63与run62十个eval点完全相同；只有run64修复stored source后轨迹才改变。
- **candidate loss的多次表面修正未自动改变真实selection。** run66的Bernoulli语义错误、run67绝对weight不对应条件概率；run68修正条件概率后，run69仍显示optimizer后方向错误。
- **按全transition归一化会稀释稀疏identity监督。** run68前200k弱；run69 target-transition normalization使candidate/graph loss达到0.801%，但性能仍低。说明归一化问题被修复后存在更上游/下游的新瓶颈。
- **欧氏梯度安全不等于Adam实际更新安全。** run70单步方向改善；run71状态重建断言、run72短期正确但rank弱、run74实际位移违反cached constraint，连续证明梯度层保证不能替代optimizer位移和非线性loss检查。
- **仅反解`exp_avg`无法证明Adam长期状态自洽。** run72即时candidate loss下降100%，但rank和行为未保持；一阶矩、二阶矩和下一步更新是否来自同一隐式gradient当时未建立。
- **lifecycle no-forget曾存在覆盖、时钟和事务问题。** run73未统一保护target-bearing更新、TTL按optimizer step、rollback曾虚增policy version；这些是代码正确性问题，不是调参问题。
- **旧cache与当前真实target可形成无严格下降方向。** run73后验证以显式RuntimeError终止；该问题不是“梯度太小”，而是约束集合和证据时序冲突。run74加入supersession后可完成训练，但随后暴露Adam位移层问题。
- **逐约束欧氏active-set会被Adam预条件破坏。** run74投影后dot容差内，actual Adam dot最小约`-1.625e-5`，rollback33.78%，rank几乎不变，按事件计数汇总的candidate-only约89.57%；频繁rollback与graph有效更新受限同时出现。
- **消除rollback没有自动恢复run70行为水平。** run75 rollback为0、实际约束安全，但final1.788、last5 1.003、capture2.8、o3=0.435，仍低于run70且在160k发生明显退化；实现保护层正确不等于策略能力和末段稳定性充分。
- **精确identity覆盖在run75进一步偏低。** active matched仅8/115.5、candidate-only约93.07%；即时概率方向100%也没有转化为足够rank、active、capture和稳定reward。
- **target-bearing日志字段曾受旧cache错误门控。** run75 step155200存在真实target/loss/gradient而字段为0；这会污染按target/base更新划分的统计，但不是训练算法退化根因。
- **后期版本没有稳定保持run70行为水平。** run71--run75的final/last5/capture整体未稳定保持run70；实现保护层不能被解释为性能单调改善。
- **旧PyTorch与测试口径反复阻断。** 除早期`nan_to_num`/`minimum`外，还出现`torch.count_nonzero`不可用、AdjBuffer transpose维度错误、support age断言和class-sum断言；测试必须与服务器版本和定义同步。
- **run96/run97的support人口修复未带来稳定性能优势。** 两个400k run均可训练，但final/last5与历史强200k参照之间没有形成稳定优势；support语义正确不能代替capture质量和后期稳定性。
- **run100暴露两类真实方向失败。** 77.6k/0和89.6k/0在objective combined阶段已反向；对应epoch1虽combined正确，Adam历史moment仍使raw/final displacement和exact score反向。clip、candidate/lifecycle和rollback均不是这些反例的原因。
- **run103--run106的strict positive pair监督没有启动。** run103 positive evidence为0；run104的成功capture全部candidate-only；run105/106虽有正candidate target和正确score方向，更新量远小于合法rank gap，successful active capture与positive pair evidence仍为0。
- **run107的strict正负pair evidence在时间上不重叠。** 2个positive和36个negative generation全局存在，但现有recent replay没有任何opposite-sign overlap，故class-complete与pair optimizer链为0。全局类别齐全不能替代同一有效时窗内的class-complete。
- **run108形成了cohort但没有完成任何logical commit。** 唯一attempt在epoch0产生非零loss/gradient和一次暂时optimizer step，随后因错误复用standard early-stop中止并完整回滚；最终pair机制对模型和性能的净作用为0。
- **provenance扩展曾出现schema/fixture同步失败。** candidate target tensor重构失败、`[3,2]`对`[3,3]`、`(2,2)`对`(2,3)`均属于诊断schema接线阻断；不能将这些Traceback解释成训练算法退化。
- **最新服务器integration Traceback仍来自旧测试语义。** 服务器执行`test_early_stop_rolls_back_without_crashing`并断言空rows，而当前工作区测试要求pair-only不调用standard early-stop并完成两epoch。该状态是文件同步不完整，当前修复后的production测试尚未得到服务器结果。
- **run109的人口分解诊断在真实训练中失败。** 164.8k无法重构pair/non-pair base-factor population，run109中断；run110完成表明阻断修复，但run109不能用于200k性能比较。
- **v13提交后联合exact复核不足。** run118在boundary修正后没有共同复核candidate/lifecycle，optimizer.step后终检失败并中断；margin方向正确不能替代原子提交正确。
- **exposure级预算集中仍受同identity其他member限制。** run121把预算集中到最近exposure，150.4k仍缩到0.5/0.03125，crossing为0。
- **identity group sum产生aggregate masking。** run122 group completion约99.96%而nearest member仅28.9%，并在149.6k无可行联合exact scale后rollback中断。group aggregate不能替代关键member。
- **nearest progress member仍受粗dyadic scale损失。** run123修正口径后seq721只完成3.12%；v19恢复搜索精度后该固定方向仍只完成2.46%，说明单纯口径或更多halving不足。
- **固定方向最大scale并不等于充分progress。** run124 seq720改善、seq721仍受保护约束限制；继续细化同一方向不能产生新几何自由度。
- **多方向actual提升仍未产生crossing。** run125 seq721 actual约提高10倍，run126总体deficit reduction也提高，但两run crossing/promotion/eviction仍全0。局部改进不能冒充selection闭环。
- **v20候选诊断不完整。** run125 schema v6不能独立重建全部未选候选的cosine、active-set、safe/unsafe和limiter；selected结果不能替代候选集合审计。
- **v21 progress seed混入zero-budget identity。** run126 seq736证明零额外预算group仍影响progress方向并可成为limiter；这是direction seed语义问题，不是删除保护约束的依据。

- **无条件24-step return失败。** run157累计dense reward而非精准completion，导致target variance、gradient clipping和basic capture坍缩。
- **terminal replay lane weight=1.0失败。** run159虽提高credit quantity约6.57倍，但24个高残差aux transition支配uniform objective，说明“更多成功replay必然更好”错误。
- **legacy all-alive post-capture random explore过度破坏协调。** run161 Hamming median4、good-start explore retreat33.42%，其破坏范围是+8→+24 persistence断点的最早可干预层。
- **transaction fail-loud错误不能转化为skip。** run154/155及后续duplicate-event/zero-gradient错误均阻断训练；正确处理是统一origin、唯一target消费和原子回滚，而非放宽exact/boundary。

## 9. 已排除或被证伪的判断

- **“日志声明trusted control已启用”不等于Runner实际使用trusted人口。** run86前的trainer/runner错接已证明，必须从selected control ratio到early-stop/window的实际读写链验证。
- **“平均mini-batch ratio”等于真实control/loss ratio被证伪。** run89前的`mean(n_i/d_i)`会使小/大population等权；当前必须使用总numerator/denominator。即使控制字段一致，仍须核对其来自真实loss人口。
- **“update级signed mass为零”不等于每个Adam step都是有效对照。** run93把双边全buffer/selected cohort切成单边optimizer step；run94/95说明pair baseline、label和optimizer transaction必须使用同一最终统计单位。
- **“pair质量守恒或one-sided zero”不等于行为成功。** run95在42个transaction满足守恒，最终reward仍0.104、capture2.1、win0.2。
- **“pair原子partition只影响pair监督”被证伪。** run95固定pair第0分区还改变该transaction的base factor PPO样本人口和Adam更新语义；pair保护不能悄然改变主factor训练预算。
- **“capture数量高”不等于高质量胜利。** run90在200k capture3.1但win0.4，失败episode仍有43.8% capture；run95 120k的短期capture/win高点也未保持。

- **“图结构合法即可带来性能提升”被证伪。** 早期 invalid factor=0、图覆盖正常，但短程 reward 仍低；合法性是必要运行条件，不是充分性能条件。
- **“更多三阶 factor 即代表更好的协同”被 run26 证伪。** 高 o3 与高 triplet fraction 没有带来有效捕获收益。
- **“order3 ratio 处于目标区间即可代表训练有效”被多轮比较否定。** run30/run37/run38 的结构指标不能替代 reward、capture、win 的同步改善。
- **“best reward 可代表整体训练质量”被 run30 与 run38 的 final/last5 差异否定。** 需要同时检查后期趋势与滑动指标。
- **“trusted clamp 降低说明 PPO 稳定性改善”被 run38 否定。** trust weighting 可以机械降低加权指标；必须同时看 raw clamp 和性能结果。
- **“early stop 已接线即有性能收益”被 run37 否定。** 真正启用 guard 后性能仍可能退化。
- **“图机制进入训练日志即可说明对策略产生影响”被 scorer 指标反驳。** run38 `adv_triplet_score_multiplier_mean≈0.998`、marginal 接近负值，说明记录/接线不等于有效排序或收益。
- **“单一结构指标改善即可证明算法性能”未获支持。** run30 尚未满足所有阶段性行为和稳定性条件，run37/38 更低；结构指标不能覆盖 reward、capture、win 和稳定性证据。
- **“完整 2M 的 final 高或 win 高即可超过 QMIX 2M”被 run49 否定。** run49 final=6.296、last5=5.751、last10=5.230、win=0.9；其 win 达到 0.8 以上，但 reward 与稳定性门槛均未超过 QMIX 2M，必须联合判定。
- **“capture-to-win 或 pair pursuit 字段非零即说明信用分配有效”被 run50/run51 否定。** run50 的该 credit 末段为零且性能不足；run51 的 pair/delayed/future 指标大面积非零却显著退化。启用、触发、性能有效必须分层判断。
- **“future exact/partial/matched 的高比例能证明成功责任归因”未获支持。** run51 的 matched/exact/partial 比例很高而 win 和 capture 低，说明时间相关/重叠记录不能替代结果指标。
- **“理论上episode outcome已中心化，factor展开后自然仍中心化”被证伪。** 重复capture/triplet展开会按标签数量重新加权；run57前的定义不能仅凭episode公式判定正确。
- **“真实capture factor必为order3”被run58证伪。** 真实事件参与者数为2时应只精确表示为order2；不得补造第三人。
- **“identity进入buffer就等于identity影响训练”被run58证伪。** matched=0时outcome/local delta全零；必须逐层检查active/candidate匹配与loss。
- **“同一稀有episode跨update重复补样能代表持续监督”被support v1问题证伪。** slot-generation/真实episode必须去重，同update epoch复用与跨update重复是不同概念。
- **“正负类齐全即可保证centered训练”被cohort审计否定。** baseline必须与实际optimizer cohort一致；full-buffer或base cohort比例不能代替最终训练总体。
- **“代码manifest变化必然导致训练轨迹变化”被run62/run63否定。** 两run十个eval点完全相同，说明修改可能未进入有效张量源。
- **“candidate loss非零即可改善rank/active”被run66--run69否定。** 还必须确认current可求导score、conditional probability、梯度方向、optimizer位移和selection链路。
- **“提高candidate loss尺度即可解决监督”被run69否定。** target-transition normalization修复稀释后，optimizer后方向仍大多错误。
- **“欧氏合并梯度与candidate梯度dot为正即可保证candidate loss下降”被run71--run74否定。** clipping、Adam动量/预条件、状态重建和非线性forward均可改变实际结果。
- **“即时probability方向正确即可形成rank和active转化”被run70--run75否定。** run75正/负即时方向均为100%，rank仍仅1.61%/4.89%，candidate-only约93.07%。
- **“rollback是无副作用的安全兜底”未获支持。** run73/run74 rollback约32%--34%，并伴随有效graph更新受限；必须把事务完整性和学习停滞同时纳入结论。
- **“消除实际位移违反和rollback即可自动恢复run70行为水平”被run75否定。** run75修正后负约束为0且rollback为0，但final、last5、capture和o3仍低于run70；剩余问题不能继续用run74同一根因解释。
- **“run68的2M高回报说明其前200k同样强”被其前200k数据否定。** run68前200k final0.344、win0.1、capture2.3，长程结果不能倒推较早阶段表现。
- **“QMIX last10为6.175”被当前CSV口径修正。** 6.175是last5；实际last10约5.559。6.175可保留为历史严格门槛，但不得再写成实际last10。
- **“run100反向由base-factor造成”被v3分解证伪。** 两个关键epoch的base pair dot均同向；graph PPO是共同最大负投影，candidate在89.6k/0也有实质负贡献。
- **“clipping、candidate projection、lifecycle或rollback是run100最早反向原因”被逐阶段日志证伪。** 反向在projection前combined阶段已经出现，clip未介入，raw与final相同。
- **“graph source修复后会立即恢复pair训练”被run103证伪。** graph污染已消失，但positive strict evidence仍为0，说明监督漏斗位于更早的行为/active evidence层。
- **“positive pair evidence为0主要由terminal、participant=2、dynamic slot或canonical join错误造成”未获run104支持。** 四个成功capture的provenance均闭合，真实断点是candidate-only未active。
- **“candidate target方向正确即可改善rank”被run106证伪。** 四个target的margin均正确改善但远小于next-better gap，rank和boundary不变。
- **“两个PPO epoch或replay重复曝光可视为多个独立行为证据”被provenance v1/v3否定。** 独立证据必须使用不同真实generation/event且完整provenance闭合。
- **“全局同时存在positive和negative strict evidence就应形成class-complete”被run107否定。** 两类证据没有在同一recent replay有效时窗重叠。
- **“one-sided zero提前used是run107 class-complete为0的原因”被overlap重建否定。** `positive_blocked_only_by_shared_used_state_count=0`；negative在positive到达前已自然离开窗口。
- **“horizon=4结构重建意味着run108必有有效pair更新”被run108否定。** fully trainable还取决于真实payload、stale/mass、全部epoch与outer transaction control；run108唯一attempt最终完整回滚。
- **“pair-only事务应沿用graph/base standard early-stop”被run108证伪。** pair-only objective把这些loss固定为0，standard control population与其事务成功语义不一致。
- **“preflight会污染正式训练RNG”未获run119及后续轨迹支持。** 独立preflight在正式模型/optimizer/CSV前完成；no-target前缀与历史轨迹一致，当前证据支持RNG隔离。
- **“diagnostics升级会改变no-target普通训练”被run119逐行对照否定。** 没有actionable boundary target时，v14路径未改变loss、gradient、optimizer或正式RNG。
- **“exact score方向错误仍是当前0 crossing主因”被run116--run126反复证伪。** 44/44、26/26、38/38及run126 34/34等target-level证据均显示exact方向正确；断点已后移到margin幅度、deficit与rank。
- **“shared Adam反向仍是当前主要原因”不符合v12以后关键事务。** 旧run100曾真实存在Adam反向，但current-priority、exact displacement与联合revalidation已修复该层；run118--run126关键boundary target没有reverse。
- **“competitor switch是run120--run125 crossing为0的主要原因”被关键事务否定。** 多轮关键日志为competitor switch=0；区域切换不是这些事务的首要阻断。
- **“candidate/lifecycle限制150.4k关键scale”被run121--run125日志否定。** 关键事务candidate=无、lifecycle=无；限制来自member exact/boundary可行域或方向本身。
- **“zero-deficit target仍消耗大额boundary budget”在v15以后被否定。** 它们只保留strict floor；run126暴露的是它们仍污染progress seed，而不是预算未回收。
- **“group completion可代表关键member progress”被run122直接证伪。** group约99.96%与nearest约28.9%同时出现。
- **“增加同一固定方向的refinement即可crossing”被run124证伪。** v19已逼近该方向真实安全边界，seq721 completion仍约2.46%。
- **“safe scale更大即可推出真实性能提高”被run124/run125限制。** 必须继续看actual、original completion、post deficit、crossing、active与行为；scale只是中间量。
- **“160k reward较高即可证明boundary机制成功”被run124的因果链否定。** run124 reward1.603但crossing/active为0，行为结果不能归因于selection闭合。

- **“countdown没到Q或Q没使用它”已排除。** env→runner→replay→sample→production RNN/Q链和functional counterfactual均有直接证据。
- **“frozen prey仍作为dense pursuit target”已排除。** run151源码与真实reward路径确认frozen prey不参与nearest target、distance/proximity、single-wolf或capture/group shaping。
- **“prey capture错误reset player RNN”已排除。** capture本身不清空仍alive player的hidden；真正player leave/rejoin与episode reset按lifecycle处理。
- **“run159继续增加terminal replay次数即可”被run160前的直接对照证伪。** 瓶颈是objective population失衡，不是纯sample数量。
- **“+24失败要求固定同一nearest agent”未获支持。** run162成功窗口identity switch更多，team substitution不是错误。
- **“bounded explore只是提高greedy概率”已排除。** run161/run162 post-capture explore fraction基本相同；改变的是一次explore替换的alive-slot数。

## 10. 当前日志分析规范

1. 先读取该 run 的保存 config、console/stderr、progress/CSV、TensorBoard event、checkpoint 目录和源码状态；若某项不存在，明确记录缺失。
2. 确认实际 steps、seed、训练是否中断、是否恢复 checkpoint/optimizer/replay；不得把脚本的目标步数当成实际完成步数。
3. 搜索 Traceback、ERROR、NaN、Inf、断言失败和重复 eval 点。没有记录不等于没有发生，需说明检查范围。
4. 按固定时间点读取趋势，并同时报告 final、last5、best、win rate、capture events、first success step。重点识别中期峰值后回落、震荡和局部最优。
5. 对动态图同时检查环境事件、active ratio、合法性、coverage、order1/2/3、mean order、rollout/eval gap 与身份保持指标。任何一项未记录时不能代替其他项。
6. 对 credit 和图策略同时看 raw 与 weighted order loss、o3-o2 差、正 advantage fraction、gate/EMA、entropy、graph/factor PPO loss、clamp、epoch、early-stop、trust/recent sampling 指标。
7. 对 value 学习检查 policy/critic/adj loss 与 `q_target_mean`、`q_tot_mean`；只有出现数值异常或趋势矛盾时才可作有限推断，不能仅从单个 loss 值判定根因。
8. 与直接对照 run 比较时必须核对环境、seed、训练步数、是否恢复状态、完整配置和代码版本。若同时更改多个模块，结论应写为关联而非因果。
9. 性能结论必须以 reward、capture events、win rate 和后期稳定性为主；图阶数、gate、clamp、bonus、scorer 是解释性诊断而非成功替代指标。
10. 若日志不足以验证机制是否启用、触发或有效，应在 run 记录中保留“日志不足”，不得根据预期补全。
11. 对 delayed/future/success/graph-return/capture-to-win/pair credit，还须检查 mean、active/positive/negative fraction、exact/partial/matched fraction、raw 与 adjusted o3-o2 差及其与 reward/capture/win 是否同步；大面积非零、接近全覆盖或长期为零都只描述信号状态。
12. 长程 run 的检查点应覆盖目标机制触发、历史退化区间和最终阶段；若采用与历史长程run相同预算，可同时在相同step比较reward、last5、last10、best、win、capture与first success，并专门检查best后回落和末段平台/震荡。
13. outcome分析应分开记录原始episode centered label、event分配、factor gate、confidence缩放、clip后credit、local delta和最终loss；任一层的零均值不能替代下一层验证。
14. identity分析应按真实event互斥拆分 active exact match、candidate-only、candidate缺失、participant阶数不支持、索引/时间错位、非法/重复factor；candidate-only不得并入matched。
15. support分析应区分base class support、augmentation、class complete、credit enabled、identity active、local delta非零和最终loss非零的漏斗；“support enabled”不等于“optimizer得到正负信号”。
16. candidate loss分析必须使用同一batch、同一canonical catalog、同一mask和确定性前向比较optimizer step前后 probability/rank；behavior score只能作off-policy诊断。
17. optimizer保护分析必须分别报告欧氏gradient dot、clipping后dot、Adam原始位移dot、修正后位移dot、真实非线性candidate loss、rollback和下一update保持率。
18. lifecycle分析须以adjacency round而不是epoch/mini-batch/optimizer attempt计时；rollback attempt、成功optimizer step、candidate policy version和cache age是不同计数。
19. 诊断字段必须说明分母和valid flag；无样本时的0不能混入均值、min/max、fraction或version字段。CSV与TensorBoard口径不同时必须保留冲突。
20. 当用户请求中的版本名称与实际console/manifest不一致时，以落盘证据为事实来源。例如run74实际为lifecycle v3，而不是请求文本所称v2。
21. 对 trusted-control 必须逐update重建 raw/trusted numerator、denominator、selected ratio、configured/actual epoch、previous/next window和runtime contract；actual early-stop严格等于`actual_epochs < configured_epochs`，最后epoch命中阈值不能记为截断。
22. 对 recent replay 必须区分requested window、selected episode、selected/trained unique episode generation、chunk数和真实训练chunk；window数值不是多样性的替代证据，generation而不是循环buffer slot才是实例身份。
23. 对pair credit必须在最终optimizer transaction层检查成功/失败evidence、重新中心化、one-sided zero、正负共同缩放、signed mass、target-bearing分母、pair/base gradient、Adam位移、score/rank/active以及随后capture/win；整update或全buffer统计不能替代该层。
24. 对原子transaction还必须报告pair与普通transaction的chunk数、valid transition数和base factor PPO人口；保证pair对照不能以改变base优化预算为代价。
25. per-objective诊断必须按每个真实PPO epoch分别报告，不能用两epoch均值隐藏epoch0 combined冲突或epoch1 Adam惯性；scalar sum、独立gradient sum、projection delta与最终`.grad`必须分别重构。
26. pair-evidence漏斗必须区分episode exposure、唯一generation、唯一event和PPO epoch。重复replay exposure与两个epoch不得计作独立行为证据。
27. candidate score→rank分析必须使用与active selection相同的合法population，记录next-better identity/gap、tie、boundary deficit和pre/post rank；正确score方向不能替代rank crossing。
28. strict pair class-complete分析必须重建evidence first-seen、recent replay最后有效update、positive/negative时窗重叠、used/expired原因和support selection；全局类别计数不能代替同一时窗可训练cohort。
29. bounded pending分析须按adjacency update计算TTL，区分current与pending来源，并检查immutable payload、policy age、transaction-time stale trust、raw/effective signed mass、generation/event single-use和checkpoint状态。
30. pair-only outer transaction只有在所有配置PPO epoch、非零target/gradient、optimizer step与post-contract均成功后才能logical commit；任何中止都必须同时验证参数、optimizer、RNG、lifecycle、pending与sequence回滚。
31. aborted cohort的日志字段必须区分“当epoch objective scope已通过”与“logical transaction未完成”。不得用硬编码0把后续early-stop误写成scope失败。
32. 默认关闭轨迹中性应比较语义状态而非不同trainer实例中optimizer state_dict的内部参数ID；旧PyTorch的ID是实例局部值，不能直接作跨实例集合相等断言。
33. selection机制必须按`target/member → canonical identity/sign → production competitor/context → pre/post signed margin → deficit → rank → crossing → active`下钻；transaction aggregate不能替代member事实。
34. aggregate正确可能掩盖member错误。group required、group actual和completion必须使用同一统计口径；run122的sum aggregate是明确反例。
35. candidate方向的completion必须统一使用原始required作为分母；缩小后的candidate floor completion不得冒充original-required completion。
36. safe scale不能单独表示改进。每个候选同时记录真实progress actual、original completion、post deficit、rank crossing、update norm和最终联合exact结果。
37. 多candidate事务必须证明parameter、optimizer、pending、lifecycle和RNG起点一致，未选candidate完全恢复，只有selected candidate正式commit；候选反序结果应在明确浮点容差内一致。
38. 多方向名称或fraction不能证明几何不同；至少检查direction norm、cosine、active constraint ordinals、safe/unsafe bracket和limiter type/member。
39. rank数值改善不等于crossing。例如11→10但signed boundary仍在错误侧时，crossing仍为0。
40. 没有member-level crossing时，active promotion/eviction及其保持属于“尚未到达”，不得写成“保持失败”。
41. boundary零样本必须以无行或valid=0表达；不得用0 score/margin/rank伪造样本并纳入统计。
42. preflight、fixture、合成测试、production测试和真实训练run必须分开报告；测试通过不能写成真实性能提高。
43. candidate/boundary方向评估必须从同一transaction snapshot开始，并在每个trial scale重新执行production forward；线性dot只能用于构造，不能代替真实提交检查。
44. commit后margin retention只有在canonical identity、sign、roster generation、selection context和competitor可比时才能计算；不同transition/prefix的margin不能直接解释为遗忘。

- 对post-capture行为，必须区分first capture、strict later-distinct、simultaneous offset0、同一prey重复capture与formal win；interval只定义为prey A first capture到prey B first later capture，同一prey重复事件不得计入。
- post-capture action row的offset k对应state offset(k-1)→k；distance progress应排除episode topology shock产生的约±19跳变，或单独标为非clean transition。
- exploration分析必须同时记录base/effective epsilon、joint explore flag、greedy/final action、alive-slot Hamming、replacement count/slot/action、available-action legality和unique joint ratio；branch概率与branch内部破坏范围是不同机制。
- completion credit必须拆分uniform natural与forced auxiliary人口，记录raw/weighted loss、gated transition、target gain、TD、gradient和clip；sample数量不能替代optimizer质量。
- first-capture geometry应报告整个population的remaining-prey nearest-player mean/median/p25/p75及distance band转化，不能只比较成功子集造成selection bias。

## 11. 当前已确认存在的问题

- **当前后期代码链尚未重新建立超过run70/run78或run112的稳定性能。** run116/117完整200k均为reward1.036、win0.5、capture2.8，低于run112 pending-off的reward2.177；run120--run126只到160k或中断，均不能替代完整200k稳定性证据。run124的160k reward1.603也没有crossing/active，不能归因为boundary机制成功。
- **capture质量仍明显不足。** run90的200k失败episode capture比例43.8%、capture episode胜率44.4%，而run70/run78分别约26.5%/24.9%和88.9%/77.8%。run95局部120k达到较好capture/win后迅速回撤；现有证据不能把责任唯一归于pair credit、Q/RNN或dynamic slot。
- **pair监督时窗问题已被bounded pending修复到可commit，但selection转化仍弱。** run117以后True/4事务可完成且逐target exact方向正确；run118--run126进一步证明margin可改善，但member-level crossing始终为0。
- **当前最早未闭合机制层是member-level selection-boundary crossing。** v13已闭合score→margin，v15--v21逐步提高budget利用、progress completion和deficit reduction；截至run126 crossing/promotion/eviction仍为0，下游active retention、matched capture和capture→win尚无该机制因果证据。
- **当前v22只有代码与测试证据。** 它修复run126暴露的zero-budget identity污染progress seed，未改变所有member硬约束；没有新的正式run证明其真实progress、crossing或行为结果。

- **当前代码版本的动态场景性能优势未建立。** run75 final1.788、last5 1.003、win0.5、capture2.8，低于同长度强对照run70；run68的2M强结果不能替代当前版本证据。
- **三阶数量与协同质量脱钩。** 证据：run26 的高三阶比例未带来收益；run38 triplet marginal≈-0.0083、multiplier≈0.998。涉及 `adj_generator.py` 与 `r_sddfg.py` 的 candidate/credit 路径。当前仍存在。
- **训练后期稳定性不足。** run30、run42、run49、run56、run65及run71--run75均出现best与末段分离；run75在120k达到best1.947后160k降到-1.632。涉及图策略、价值学习和稀疏监督保持，不能归为单一模块。
- **capture、win rate 与 reward 不总同步。** 证据：run30 capture=2.9、win=0.5；run37 capture≈2.6、win≈0.1；run38 capture≈2.8、win≈0.4。涉及环境成功转化、策略和评估；精确因果未确认。当前仍存在。
- **PPO stale / clamp 风险仍在。** run75 raw graph/factor clamp均值约0.435/0.219，236/242个窗口触发early stop；当前没有PPO实现错误证据，不能据此直接调参。
- **训练图与评估图分布可能不一致。** 证据：run18 gap=0.162、run23 gap约0.166；run30 gap=0.003 但性能仍不充分。差异并非唯一根因，当前是否持续存在取决于具体 run。 
- **长程结果尚未形成全面基线优势。** run68的last5/last10高于QMIX实际滑动值，但final6.718<6.947、win0.6<0.8；QPLEX 2M结果更高但有效性争议未解决。当前不能写为SDDFG已全面胜出。
- **捕获到胜利的转化并不稳定。** 证据：run42 capture=4.1 但 final win=0.4；run49 final capture=4.5、win=0.9 但 reward 滑动窗口仍低于目标；run50/51 的 capture 与 win、reward 再次共同偏低。当前已有日志不能将问题唯一归到环境、credit 或图拓扑。当前仍存在。
- **outcome/candidate监督覆盖仍低。** pair过宽问题自run52已修复；run75 active matched仅8/115.5、candidate-only约93.07%，support有效窗口31/242且exhausted40/242。涉及event identity、candidate/active selection和support漏斗。
- **candidate即时改善很少转化为rank和active。** run75正/负即时概率方向均为100%，rank仅1.61%/4.89%，candidate-only约93.07%。run70--run75共同证明即时概率正确不能替代rank、active和行为闭环。
- **run74的Adam actual displacement违反已由run75 v4修复。** run75修正后负约束计数为0、rollback为0；因此该问题保留为run74历史根因，不再列为当前已复现机制错误。
- **lifecycle v4正确性改善但行为收益不足。** run75全部242个adj update推进policy version且无rollback，但性能仍低于同长度run70；当前问题是性能和覆盖不足，不能再归因于v3 rollback。
- **严格candidate→active生命周期证据仍不完整。** run75新增age1/5/10保持统计，但event级active match仍不是“某个candidate被后续选为active”的严格逐身份转化率。
- **target-bearing诊断漏记已修复但尚待新run验证。** run75 step155200存在真实target/loss/gradient而旧字段为0；修复只影响日志统计，不能解释run75 reward轨迹。
- **旧run源码manifest不够细。** run75只有aggregate SHA且Git commit/tree不可用，能够标识整体源码状态但不能逐文件追溯server源码。
- **stale/recent-window 风险未完全消失。** 证据：run51 adj graph/factor stale ratio 平均约 0.443/0.260，末值约 0.480/0.305；即使近期窗口和 emergency 参数被启用，性能仍低。当前仍存在。
- **历史证据不完整。** 多数 run 缺少完整 config、manifest、恢复状态、CSV/TensorBoard 摘录；这限制了因果判断。当前仍存在。

- run162虽在单seed 80k显著改善capture、strict distinct、win和final eval，但尚未建立多seed或长程相对QMIX/run68的论文级优势。
- far-start仍是最清晰残余：run162中remaining-prey nearest distance>=8的54/141 first-capture episode没有strict distinct。当前缺少pre-capture per-agent Q、factor identity和action history，尚不能把该相关性安全转化为代码修改。
- run162 early eval并非单调优于run161：20k/40k win均为0，主要收益集中在60--80k并在80k eval达到reward1.588/win0.3/capture2.7；短程高training win不能替代更长后期稳定性验证。
- run127--run146若干transaction中间版本只有summary或失败fixture，逐run源码差异与单变量关系未完整重建；不得把完整终点指标自动归因给某个exact/pending patch。

## 12. 必须继承的工作原则

1. 每次代码修改后，由训练日志、配置和源码状态共同形成可核实记录。
2. 必须先完整读取新训练日志，再决定是否修改代码。
3. 修改必须针对日志中已暴露且有证据支持的问题。
4. 不允许在没有日志依据时连续堆叠新机制。
5. 每轮尽量控制变量，并明确直接对照 run；多变量修改必须承认混杂。
6. 不允许只根据 final reward 判断实验成败。
7. 必须同时检查趋势、稳定性、胜率、关键事件、触发次数、loss、梯度/数值异常和诊断指标。
8. 必须确认改动真实进入训练，且区分“存在、启用、触发、性能有效”。
9. 必须检查是否意外恢复旧 checkpoint、optimizer 或 replay。
10. 每次训练步数必须由目标机制首次触发、触发频率、生命周期、历史退化区间和比较需求决定；200k与2M都只是历史预算，不是固定前置阶段。
11. 失败实验必须保留完整记录，不能被后续版本覆盖或删除。
12. 修改前应先定位根因；不得以持续增加奖励项或结构约束掩盖证据不足的根本问题。
13. 日志不足以支持修改时，必须明确记录证据不足，不得猜测。
14. 每轮报告应记录文件、机制变化、参数、可观测日志和直接对照；长期文档只在用户明确要求阶段总结时更新。
15. 不得提前宣称性能一定会提升。
16. 新任务框开始工作前必须完整阅读本文档；新run结果是否写回本文档由用户的明确阶段总结指令决定。
17. 新结论与旧结论冲突时，必须保留判断变化及其证据，不得抹去旧记录。
18. 代码存在、训练启用、实际触发、进入optimizer和产生正面性能作用必须分别证明。
19. 不得通过调低/调高coef、cap、gamma、学习率、reward scale、priority、EMA或更换seed掩盖已确认的机制错误。
20. 不得重新引入普通正reward pair credit、offset=0或历史floor；不得为缺失identity臆造participant/target。
21. event、episode和candidate监督质量必须守恒，non-target local delta必须为0；active与candidate身份必须互斥。
22. single-outcome归零保护不得用伪造正负样本替代；support复用必须有限、可去重、可审计。
23. loss数学正确不等于gradient和optimizer更新正确；任何candidate机制都要追踪到实际conditional probability、rank和active。
24. rollback、optimizer state sync、policy version和lifecycle round必须作为事务检查；失败attempt不得伪装成成功更新。
25. 服务器旧Python/PyTorch兼容性是有效性条件；预检API与训练环境不一致时，不能把测试失败解释为算法失败。
26. 每轮只选择一个有完整日志和代码证据的主要根因；其他问题可记录或修复独立日志错误，但不得混改多个未经证明的主要机制。
27. 训练比较优先使用相同步数或相同预算的直接对照，并同时检查学习速度、后期窗口、行为与图结构；长程比较保留QMIX、run49、run68以及有效性有争议的QPLEX 2M记录。
28. 每轮只处理因果漏斗中第一个未闭合层；上游未闭合时，下游行为指标不得用于该机制归因。
29. 代码修改后先完成最小静态、fixture/replay、production、rollback和RNG验证；不得以测试通过替代真实run验证。
30. 正式训练长度必须由修改分支最早真实触发、所需ordinary update观察数和evaluation间隔确定；历史200k/400k不是默认值。
31. manifest、saved config、diagnostics/schema、console runtime header和CSV字段必须相互一致；版本声明与实际schema不一致时fail-loud并判run无效或部分有效。
32. fresh/resume、terminal/periodic checkpoint、optimizer state和completion metadata必须显式核对；事务语义改变后旧checkpoint不得继续作为同一因果实验恢复。
33. forced/non-actionable evidence不得进入strict pair或boundary监督；证据generation/event必须single-use，不能通过重复旧loss、旧sample或旧event制造累积进展。
34. 失败run和失败transaction均保留；optimizer.step后失败必须审计参数、optimizer、pending、lifecycle、RNG和sequence是否原子恢复。
35. 无样本、不可用字段和未触发分支不得用default-zero掩盖；不使用宽泛`try/except`跳过事务错误。
36. 性能表必须明确统计窗口和分母。final evaluation、0--N全部evaluation累计、训练episode累计是不同口径，不能混用failed-capture、matched或capture-episode-win。

## 13. 信息缺口与未确认事项

- run127--run146的全部console、transaction schema、逐文件manifest差异和每个失败attempt没有在本次整理中逐一重建；summary存在只证明各自target完成，不能补造单变量transaction收益。
- run150与run151公共行为统计相同，但manifest/transaction修复状态不同；当前只把run151作为统一origin后的正式性能基线，不把轨迹相同解释为源码完全相同。
- run156的180k行为汇总可确认，但其全部pre-capture t-32..t历史位置、per-agent Q与factor序列未落盘；“双目标分工从first capture前何时形成”仍无完整transition级答案。
- run158--run161的terminal gate/lane/weight production人口已验证，但completion TD如何在每个selected factor间分配的完整per-factor gradient仍未长期落盘；run162的强行为改善使该问题不再是当前第一个断点，但归因细节仍未确认。
- run162尚无多seed、180k或更长预算结果；不能从80k single-seed formal wins23推断最终win rate、长期稳定性或相对QMIX/QPLEX的论文优势。
- run162 residual far-start band缺少足以区分pre-capture observability、RNN memory、Q ranking、factor persistence与环境POMDP的最小诊断；不能据此加入oracle prey位置或特定slot保护。
- run76、run77、run79、run80、run88的训练产物、配置、源码差异和结论没有出现在当前可见对话；不得从编号连续性推断其机制或性能。
- run82--run85虽被用户描述为完整训练，但本次整理未重新读取其完整console、CSV、TensorBoard、checkpoint和manifest；candidate v12独立 residual Adam、lifecycle v10、terminal checkpoint和双Adam持久化在这些run中的实际触发次数与性能效果仍缺定量证据。
- run86--run94的大量数值来自本次对话提供的比较表或后续问题描述，当前整理没有重新打开全部落盘目录；除明确标明的run95审计外，不能把它们写为本次独立复核的产物事实。
- run87 2M与run68 2M、run90 600k等长程结果的完整config、是否恢复状态、逐文件manifest和每个阶段的CSV/TensorBoard一致性未在当前材料中逐一验证；已记录的是用户提供汇总。
- 在第5次整理时，run95后的v5批人口平衡修复尚未在真实服务器PyTorch1.3.1执行；该时间状态不能被当时本地PyTorch1.8.1+cu111测试替代。
- run95之后v5/v6已由run96/run97真实训练；但run95当时“尚未验证”的时间状态仍保留于历史章节。run96/run97服务器的精确PyTorch版本没有在本次整理中重新核验。
- run100/102已保留六个pair transaction的combined、Adam、score及per-objective关键值，因此这些特定epoch的graph冲突和Adam惯性可确认；当前仍缺run96/97其他全部pair transaction的同等逐epoch原始序列。
- capture后失败的直接原因、candidate replacement是否被margin/selection/任务价值中的哪一层阻断、static/dynamic slot同步回撤是否源于slot状态、以及Q/RNN是否在高点前遗忘成功策略，均仍是未确认因果关系。

- run1--run17、run19--run20、run24--run25、run27--run29、run31--run36 的许多完整控制台、CSV、TensorBoard、config、checkpoint 恢复信息未在本次对话提供；表中只保留已知事实。
- `run38` 用户曾通过附件提出分析请求，但附件的完整文本与原始日志内容不在当前可见对话中；本文件仅采用已记录的指标摘录。
- run41、run43--run47 的原始 eval CSV、train-adj、TensorBoard、保存 config、完整源码差异和精确最终指标未在当前可见对话中保留；表中只能记录已出现的实验主题或问题描述，不能补造数值。
- run42 的所有阶段性 credit、stale、clamp、order2/order3、q-target/q-tot 原始序列未完整保留；本文件只记录已给出的 2M 汇总结果。
- run49 的完整性能节点已摘录，但其 checkpoint/optimizer/replay 恢复状态、完整 config 哈希、源码 manifest、每个 credit 指标的原始逐点序列未在当前可见信息中完整核验。
- run50/run51 的关键 performance 和部分 train-adj 汇总已记录；二者完整 TensorBoard、所有 loss/gradient、每个 eval 图结构比率及恢复状态仍未确认。
- run52--run75的eval CSV、主要console版本、部分manifest与关键train-adj诊断已核验；run75的TensorBoard已完整读取1164个scalar tag。但并非每个历史run的全部TensorBoard scalar、checkpoint目录、optimizer/replay内部状态都被逐文件保存，缺失项不能补造。
- run75审计后的target-bearing日志修复尚无新训练验证；训练机制按设计不变。
- run52--run75多数console未显示恢复旧状态；run74--run75明确`model_dir`为空，但历史其他run的checkpoint/optimizer/replay恢复状态并非都有逐项独立证据。
- run75完整aggregate SHA为`5f6347acfd51de0c12d25385d0e6fbdd8ebdb8073272f4d4e62733dcb38610f0`；没有逐文件hash artifact，不能从aggregate反推单文件差异。
- Wolfpack 奖励函数、捕获成功终止语义、动作/观测具体维度和训练时的精确恢复逻辑未在本次整理逐行核验。
- QMIX 的 `torch.nan_to_num` 兼容性修复确已使训练继续，但精确修改行、旧/新 PyTorch 版本未确认。
- 各 run 使用的 Git commit 不可确认：历史控制台曾打印 `Git commit: unavailable; tracked tree: unavailable`。
- run75已确认lifecycle v4的Adam realized-displacement投影、nonlinear backtracking、TTL时钟和observation archive进入训练，并确认rollback降为0；它是否在多seed或更长训练中保持相同正确性、是否改善行为性能仍未确认。
- run75已有age1/5/10保持统计，但严格candidate→active逐身份转化率仍未记录；event级active match不能替代该数据。
- QPLEX 2M的汇总结果存在，但历史上“不可靠”的具体原因、排除日志和源码状态未在当前对话保留，属于明确冲突信息。
- run62与run63的eval轨迹完全相同是已确认事实；除graph advantage replay写入问题外，是否还存在其他未记录差异无法确认。
- 历史上曾提出多种机制性设想；除本文件记录为已实现/已检验者外，其他设想不构成当前事实或建议。
- run96--run108均有saved config和关键CSV，但本次整理没有逐一重读每个run的完整console、TensorBoard、checkpoint内部状态与逐文件manifest；表中精确机制结论以已核验CSV、配置、源码和对话中保存的审计结果为边界。
- run101因diagnostics版本错配无效；其任何未完成checkpoint、step或loss均未混入run102及后续分析。
- run108实际训练环境的完整PyTorch/CUDA版本未从本次可见run console中确认。历史上run95确认PyTorch1.3.1，本地/部分preflight曾使用PyTorch1.8.1+cu111；不同时间点不得互相替代。
- run108后pair-only control scope修复已由run109以后真实训练取代原“无新run”状态；run108当时的同步冲突仍保留为历史事实。run109因人口分解错误中断，run110以后可完成；精确每个中间版本的源码差异除diagnostics号与落盘行为外并未全部逐文件重建。
- run111没有对应结果目录；其编号只在历史叙述中出现，不能推断机制、配置或性能。
- run113、run114是完整200k中间run，但当前对话只核对了config、diagnostics、eval、terminal和无阻断错误，未完整重建其每个事务定义和直接单变量关系。
- run116此前未落盘失败attempt的完整stderr、源码状态和终止位置未保留；正式性能只使用已落盘完整attempt。
- run118与run122均为中断run。run118最后完整eval为180k、最后transaction约182.4k；run122最后完整eval140k、149.6k事务rollback后中断。二者均没有正式200k/160k终点指标。
- run120--run126均为单seed 160k机制实验（run122除外）；它们没有完整200k后期稳定性证据，不能与run112/run117的200k final/last5作同长度结论。
- run125用户摘要的failed-capture77.78%和capture-episode-win10%与实际160k final evaluation topology CSV重算66.67%/20%冲突；matched22.22%一致。冲突可能来自不同统计窗口或分母，当前文档同时保留并优先标明落盘文件口径。
- run126实际结果已经由项目文件复核，不再属于“对话无结果”。已确认其完整160k、v21、8候选和crossing=0；当前仍缺同一selection context下commit后两个ordinary update的margin retention序列。
- run126候选CSV记录了每个候选相对full Adam的cosine，但没有完整候选两两cosine matrix、active-gradient rank、progress gradient在保护约束张成空间/可行切空间中的投影norm，因此局部可行几何是否已穷尽仍未确认。
- rank-deficit lifecycle需求仍未确认：现有相邻事务transition/prefix/context不同，不能证明ordinary update系统性抹除已取得margin。
- v22当前代码排除zero-extra-budget identity的progress seed并通过测试，但没有正式训练run；其真实性能、deficit reduction和crossing结果未确认。
- run117--run126均未产生已确认的member-level crossing→active promotion/eviction→两个ordinary update保持链；matched capture和capture→win对该boundary机制的因果收益未确认。
- 当前代码链是否能在完整200k或更长同预算下超过run112 pending-off，以及是否能恢复run70/run78行为水平，均未确认。

## 14. 文档修订记录

### 第1次整理

- 整理来源：本次历史对话整理、历史控制台指标摘录、当前工作区源码与脚本检查（2026-07-16）。
- 新增内容：创建统一历史文档；补入项目边界、动态环境配置、关键目录、当前源码机制、run1--run38 和三类基线的已知记录、历史判断标准、成功/失败经验、已证伪判断、日志规范和信息缺口。
- 补充内容：基于当前源码确认 recent episode window、PPO guard、stale trust、triplet credit 相关文件和训练脚本参数确实存在；未将其误写为性能已验证。
- 修正内容：无已有统一文档，无法进行旧记录修正。
- 合并的重复内容：将反复出现的“结构比例不能代表性能”“run30 为近期较优但未达标”“run37/38 机制接线不等于性能成功”整合到对应 run、失败经验和证伪判断中。
- 保留的冲突信息：无可核实的数值冲突；run36 是否真正启用 early stop 的历史说法以“run37 被明确描述为真正启用”为准，run36 保留为初次检查。
- 未解决的信息缺口：大量 run 的原始日志/配置/恢复状态、run53 以后输出内容、源码改动的时间顺序和奖励/终止精确语义未确认。

### 第2次整理

- 整理来源：本次历史对话整理；run40/run48 Traceback；run42、run49、run50、run51 的历史日志、CSV、config、console 与 train-adj 指标摘录；当前源码检查（2026-07-16）。
- 新增内容：补入 run40--run51 的总表记录，新增 run40/run48、run42、run49、run50、run51 详细记录；补入 2M 对照标准、run49 全程节点、run50/run51 的关键 credit 与性能证据。
- 补充内容：扩展关键文件职责，记录 parser、兼容性、扩展 batch、delayed/future/success/graph-return、capture-to-win、pair/triplet complementary credit 的当前源码接线与实际运行边界。
- 修正内容：将“截至本次对话最新完整结果为 run38”的过时表述修正为：run49 是当前已记录的 SDDFG 长程最高终值，run50/run51 是后续已知 200k 记录；未删除 run38 的历史结论。
- 合并的重复内容：将 capture/win/reward 脱钩、信用指标不能替代性能、长程 best 后回落、parser/预检/批字段接线问题分别并入机制、run、证伪判断和日志规范章节。
- 保留的冲突信息：run43--run47 的原始结果不在当前可见对话；仅保留实验主题或后续问题描述，不将其与 run50/run51 的精确结果混同。run45 与 run46 的编号表述曾出现不一致，统一保留为 grouped 未确认记录。
- 未解决的信息缺口：各 run 的恢复状态、配置哈希、源码 manifest、完整 TensorBoard/CSV、run41/run43--run47 的精确指标，以及 run49/run50/run51 的全部逐点诊断和实际 Git commit 仍未确认。

### 第3次整理

- 整理来源：本次历史对话整理；run52--run74落盘eval/train-adj CSV、控制台日志、源码manifest、历史Traceback、当前源码和各轮用户提供的机制审计要求（2026-07-16）。
- 新增内容：补入run52--run74逐run总表和五个关键演进阶段详细记录；记录pair strict-future修复、centered outcome v2--v5、capture identity、support v1--v3、factor loss normalization v2、candidate loss v1--v4、gradient projection、actual-update guard、Adam state sync和lifecycle v1--v4的时间顺序与证据边界。
- 补充内容：新增run68完整2M、run70强200k、run74逐约束/Adam位移/rollback证据；扩充关键文件职责、实验有效性、日志分析规范、成功/失败经验、已证伪判断和当前问题。
- 修正内容：将run49“当前SDDFG最高终值”修正为历史曾最高、现由run68超过；将QMIX 2M last10从误写的6.175修正为约5.559；将run74实际lifecycle版本按console/manifest修正为v3，并把run74后工作区v4明确标为尚未训练。
- 合并的重复内容：将run56--run65反复讨论的episode中心化、support和loss尺度合并到统一机制链；将run66--run74反复讨论的candidate delta→loss→gradient→Adam位移→lifecycle链合并，保留每个run的判断变化。
- 保留的冲突信息：QPLEX 2M有final8.932等汇总，但历史“不可靠”判断缺少排除证据；两者并存。用户请求曾称run74 lifecycle v2，实际落盘为v3，文档保留这一修订。run62/run63轨迹相同但manifest不同，保留为接线异常证据。
- 未解决的信息缺口：run75未审计；run74后lifecycle v4无训练结果；多数run缺少完整TensorBoard逐scalar、恢复状态逐项证明和Git commit；严格candidate→active生命周期及age1/5/10保持在旧run中不完整。

### 第4次整理

- 整理来源：run75全部落盘CSV、TensorBoard event、config、模型与best metadata、对应console、run74直接对照、训练脚本manifest逻辑、当前源码和本轮测试（2026-07-16）。
- 新增内容：补入run75总表与详细记录；确认run75严格有效、run74为直接对照、282项配置完全相同、前60k eval完全一致；记录lifecycle v4实际位移投影、backtracking、state sync、TTL和age1/5/10 observation archive的运行时证据。
- 补充内容：记录run75的完整十点趋势、final/last5/last10/best/win/capture、图结构、candidate覆盖、loss/梯度、credit稀疏性、PPO clamp、episode lifecycle、CSV/TensorBoard完整性和200k门槛判断。
- 修正内容：将“run74后lifecycle v4尚未训练”更新为“run75已验证实现正确性但未达性能门槛”；将run74 Adam位移违反保留为历史根因，并明确它在run75未复现；按同一事件计数口径将run74 candidate-only精确为约89.57%。
- 本轮代码修改：修复`capture_candidate_identity_lifecycle_target_bearing_update`受旧cache门控导致的漏记；新增无cache真实target回归测试。代码修复已完成，但性能效果尚未经过新run验证。
- 已完成测试：Python语法/import、19项candidate identity合成测试、训练脚本列出的完整SDDFG preflight、动态图验证、shape/mask/replay/梯度检查和`git diff --check`均通过。
- 新增经验：实际位移约束和rollback可以被修复，但这不足以自动恢复run70行为水平；真实target诊断必须独立于旧cache存在与否；不同age retention cohort不能作单调性推断。
- 已排除解释：run75性能不足不能继续归因于run74同条件的Adam位移违反或rollback；step155200的target-bearing=0是日志统计错误，不是candidate loss未进入训练。
- 未解决的信息缺口：aggregate manifest不能逐文件重建server源码；run75无严格candidate→active逐身份转化率；当前日志修复尚无新run字段验证；多seed和长程性能未确认。

### 第5次整理

- 整理来源：本次历史对话整理；用户提供的run81--run94配置/比较指标/错误信息；run95控制台、CSV、TensorBoard、源码审计与本地定向测试摘要。
- 新增内容：将run76--run95纳入总表；新增candidate v10--v12、独立residual Adam、lifecycle v10、双Adam/terminal checkpoint、trusted-control、population-total aggregation、signed pair credit、identity-local PPO、pair-local normalization、optimizer cohort centering、pair-evidence support及atomic transaction的演进记录；补入run95完整性能与transaction守恒事实。
- 补充内容：扩展`base_runner.py`、`adj_buffer.py`、`pair_credit.py`、`adj_training_control.py`和合成测试的职责；把control人口、generation采样、optimizer-step pair质量守恒和base PPO transaction人口加入日志分析规范。
- 修正内容：将历史文档“当前代码状态”从run75更新到工作区run95后v5；明确run95的守恒链虽然成立，性能并未改善；将run83的KeyError/UnboundLocalError、run91的ImportError和`validate_adj_buffer`断言作为历史阻断/测试口径记录，而非性能指标。
- 合并的重复内容：把run91--run95反复讨论的pair信用生成、shared advantage污染、v2分母、sampled cohort、原子transaction和batch人口问题按“生成→最终optimizer transaction→主factor PPO→score/rank/active→行为”的层级合并。
- 保留的冲突信息：run84/85及run86--run94被描述为完成训练，但多数完整落盘证据不在当前对话；文档保留用户提供数值并明确未在本次独立复核。run95当前v5补丁通过本地测试但尚未由服务器PyTorch1.3.1训练验证。
- 未解决的信息缺口：run76--run80/run88缺失；run82--run94的原始产物多数未重新读取；pair梯度到实际score/rank的逐transaction序列、capture失败因果、Q/RNN与slot的最早先导关系均未确认。本文没有写入未验证的后续实验或算法方案。

### 第6次整理

- 整理来源：本次历史对话整理；run96--run108 saved config、eval/progress/train-adj/transaction/pair funnel/candidate identity/provenance/pending CSV；run107独立generation文本、cohort-overlap与horizon JSON；run108源码和测试脚本审计。
- 新增内容：将run96--run108加入总表和详细记录；补入support v6、optimizer diagnostics v2、per-objective diagnostics v3、graph advantage source v2、pair-evidence funnel v1/v2、candidate same-population rank、provenance-complete evidence、strict pair时窗重建、bounded pending production和pair-only outer atomic transaction的演进。
- 补充内容：记录run100六个真实epoch的combined/Adam/score方向，run102 graph/candidate/base/outcome/entropy的关键pair dot，run103--run106的candidate→rank→active断点，run107的独立generation及2/36 strict pair时序，run108唯一pending cohort的payload age、trust、mass、loss、gradient、中止与完整rollback。
- 修正内容：把3.19从“当前v5状态”改为“历史v5状态”；将run95后“尚未训练”的时间性判断与run96/97真实验证分开；把run108 abort row的objective scope=0标为日志硬编码错误；明确pair-only复用standard graph/base early-stop才是唯一attempt未完成2 epoch的直接根因。
- 合并的重复内容：将run101版本错配、provenance schema/fixture错配、default-off optimizer state ID比较和最新服务器旧integration测试分别归入无效run、诊断接线、测试语义和文件同步问题，不与训练性能或算法根因混写。
- 保留的冲突信息：run108后工作区已采用pair-target control scope并更新integration测试，但用户最后服务器Traceback仍运行旧测试函数；两种状态同时保留，以工作区源码和实际Traceback分别标明，未假定服务器已经同步。
- 未解决的信息缺口：修复后的pair-only两epochtransaction尚无新真实训练commit；最新服务器CPU/CUDA/checkpoint完整回归及实际PyTorch/CUDA版本未确认；成功commit后的exact score、rank、active、capture质量和性能影响均无事实证据。本文未写入未来实验、参数或算法建议。

### 第7次整理

- 整理来源：本次历史对话整理；run109--run126 config、console、eval/transaction/pending/boundary/group/candidate CSV；run120--run126 `logs/summary.json`；run126 TensorBoard、manifest、terminal checkpoint与当前源码检查（2026-08-08）。
- 新增内容：将run109--run126加入总表和重要run详细记录；补入v12--v22从逐target exact事务、production boundary、deficit预算、water-filling、identity grouping、progress member、最大安全scale、多方向和八候选到当前deficit-only progress seed的时间顺序。
- 补充内容：加入score→margin→deficit→rank→crossing→active→retention→matched→capture conversion因果漏斗；增加member级、original-required、候选隔离、cosine/active-set、safe/unsafe、limiter和同context retention的日志规范；补入run126完整8候选与性能证据。
- 修正内容：将旧记录“最新完整run为run108、pair-only修复尚无新run”修正为run109以后已有真实训练、run117以后逐target事务已闭合；将当前代码状态由run108后diagnostics v2更新为run126训练时v21和当前工作区v22。修正依据为run109--run126落盘config/console/CSV/summary/manifest/checkpoint。
- 修正内容：用户本轮称run126在当前对话没有实际结果，但本对话前一任务及项目文件已经完成run126全产物审计；文档按文件证据记录其fresh、True/4、v21、160k完整、crossing=0，同时明确来源不是仅凭run编号或计划推测。
- 合并的重复内容：将run115--run126反复出现的aggregate masking、预算碎片化、fixed-direction缩步、候选选择和0 crossing判断按“exact→margin→deficit→rank”合并，保留每个版本改善层和随后暴露的新断点。
- 保留的冲突信息：run125用户摘要failed-capture/capture-episode-win为77.78%/10%，实际160k final evaluation topology CSV按明确分母重算为66.67%/20%；matched均为22.22%。两者同时保留，不静默覆盖。run111无目录，保留为编号/口头误用而非实验。
- 未解决的信息缺口：run113/114逐事务语义、run116未落盘attempt、run126同context margin retention、完整candidate两两cosine/tangent-space rank、首次member crossing、active promotion/retention及其行为因果收益均未确认；v22只有代码与测试证据，没有正式训练结果。

### 第8次整理

- 整理来源：本次同一历史对话；run127--run162 summary/config/manifest/checkpoint；run148--run162 formal episode、capture、post-capture、exploration、Q/TD/loss、terminal lane CSV；run149/run151/run152/run157--run162 production fixture与源码审计。
- 新增内容：将run127--run162加入总表；新增freeze countdown 35/59输入、joint epsilon与post-capture floor、terminal-gated 24-step completion target、terminal replay lane、0.10 weighted auxiliary population和bounded post-capture slot replacement的完整演进。
- 补充内容：记录run154/155及后续strict-pair event/zero-gradient阻断；补入first→strict distinct→<=24→win漏斗、first-capture双目标geometry、topology-clean progress、greedy-final Hamming、replacement slot/action多样性和terminal optimizer人口的新分析口径。
- 修正内容：把“当前工作区v22尚无正式run”的时间状态更新为之后已有transaction修复和完整训练链，但保留run126当时的历史结论；把当前最新完整机制证据更新为run162 fresh 80k，同时明确其single-seed/短预算边界。
- 合并的重复内容：将run147--run153的输入/探索接线、run156--run161的completion credit与run162的persistence修复分别按“production接线→optimizer信号→行为→性能”合并，避免把每轮重复contract检查写成独立机制。
- 保留的失败与反证：unconditional n-step的run157、weight=1.0 lane的run159、transaction中断的run154/155和无summary中间run均未删除；测试通过与performance成功继续严格区分。
- 未解决的信息缺口：run127--run146逐patch源码因果、run156完整pre-capture历史、per-factor completion gradient、run162 far-start根因、多seed与长程论文优势均未确认。本次归档不写入未来参数建议或未执行计划。

### 第9次整理：strict-pair有限零更新改为事务性no-op

- 训练阻断：正式训练在`train_adj_on_batch()`的pair Adam路径中触发`RuntimeError: pair Adam direction guard has no usable update`。旧guard把Adam实际位移、pair梯度、裁剪后的pair方向点积和裁剪梯度范数四项合并为一个必须严格为正的fatal条件；其中有限零位移、裁剪后零方向和数值下溢均可能是本batch没有可提交更新的正常优化结果，并不自动表示参数或事务损坏。
- 根因与减法：移除上述复合hard failure，不再把“没有可接受proposal”等同于“训练状态损坏”。保留现有nonlinear exact candidate/backtracking对正scale提交的score、selection-boundary、candidate、lifecycle契约；这些检查决定candidate能否commit，不要求每个batch一定commit。
- 新语义：新增结构化`PairOptimizerRecoverableNoOpError`，只覆盖finite的zero Adam displacement、zero clipped gradient/dot和后续finite zero candidate/final descent。进入该状态前必须恢复并逐项验证parameter、Adam state/step、lifecycle cache、policy/lifecycle clock、retention archive、selection state以及CPU/CUDA/NumPy/Python RNG；runner将其记为recoverable no-op，不推进transaction sequence，不消费pending evidence/generation，并继续下一batch。
- pending与普通路径：pair-pending outer transaction把同一typed no-op作为有界`NO_USABLE_UPDATE`回滚；standard sample path单独统计`adj_sample_recoverable_noop_chunk_count`，与strict-exact bounded deferral分开，不再由底层、中层和runner重复raise同一可恢复结果。事务诊断契约升级到v41。
- 继续fail-loud：parameter/optimizer/gradient NaN或Inf、非法shape/index/pair identity、validated pair gradient内部不变量失效、负向/reversed pair或candidate descent、transaction origin/selection/lifecycle不一致、rollback无法逐项精确恢复、optimizer reconstruction不一致、以及exact search声称漏掉已验证可行点仍会立即终止训练。
- before/after fixture：production-shaped标准pair batch通过真实Adam后恢复参数，忠实制造finite zero displacement。旧语义稳定落入复合fatal guard；新语义连续10次均返回verified atomic no-op，parameter、Adam moments/step、所有事务cache和RNG逐项不变；第11个正常batch能够继续提交。另有真实Adam后注入NaN的反例仍触发`FloatingPointError`并完成精确回滚。
- 已完成验证：CPU/CUDA完整pair optimizer transaction diagnostics、CPU/CUDA pair-pending production integration、run139 exact-failure fixture、run140 boundary contract fixture、dynamic graph validation、pending launcher contract、Python `py_compile`/`compileall`和`git diff --check`均通过。正式fresh长训练尚未验证本次recoverable no-op在真实生产频率及其长期性能影响。

### 第10次整理：run163审计与首捕前策略诊断闭环

- 实验有效性：run163为seed=1、fresh、target/actual=80000/80000的完整训练；summary.json、带run前缀的config/CSV、TensorBoard event、逐文件source manifest、terminal模型及adj_optimizer_state.pt均存在。terminal checkpoint真实反序列化后确认checkpoint_kind=terminal、training_complete=true、total_env_steps=target_env_steps=80000，没有resume/restore。console及CSV未发现真实Traceback、ERROR、NaN/Inf、checkpoint mismatch、strict-pair中断或rollback。
- 已冻结机制契约：checkpoint确认Q target v4、n_step=24、terminal_gated、terminal replay lane开启且auxiliary transition weight=0.10；policy exploration v4、epsilon 1.0→0.05/228000、post-capture greedy floor=0.25、POST_CAPTURE_EXPLORE_MAX_RANDOM_AGENTS=1。run163没有提供修改这些机制的反证。
- run162→run163不是严格单变量性能对照：config除resolved run id/dir及manifest hash外一致，但source manifest从91062d...变为4960c4...，改变了adj_generator.py、r_sddfg.py、base_runner.py、pair diagnostics/pending/launcher/train入口和pair_pending.py，并新增pair_direction.py。joint episode行在episode index 0--264完全一致，首个行为差异出现在episode 265；post-capture轨迹首差在environment episode 232、episode step 29，train aggregate首差在54.6k。35.2k首个pair事务两run完全一致；52.8k同一cohort的第二epoch score变化由run162的约0.1509048变为run163的约0.1929990，随后轨迹分叉。因此52.8k之前可作精确前缀验证，之后只能比较结果分布，不能把差异单独归因于某一源码机制。
- first capture→distinct→win漏斗：run161/run162/run163 formal episode均为368；capture event为169/204/209，first capture为141/141/148，strict distinct为3/16/15，全部strict distinct interval均<=24，training win为3/23/19。first→strict转化率为2.13%/11.35%/10.14%。run163 interval为21,7,3,12,20,22,8,4,17,16,14,2,14,16,1；另有1个同transition win和3个超过已记录24-step窗口的win。run163相对run162提高first capture和总capture，但strict distinct及training win小幅回落，属于平台波动而非terminal completion链失效。
- 80k evaluation：run161/run162/run163的10个formal eval episode分别得到first capture=8/10/8、capture=16/27/17、strict distinct=3/6/4、<=24 distinct=0/1/2、win=1/3/3；run163 eval reward/win/capture为-1.2860001/0.30/1.70，run162为1.588/0.30/2.70。run163没有提高eval win，且reward/capture回落；不能只用training reward或capture数判定提升。
- bounded exploration长期证据：按post-capture action与前一状态对齐后的eligible/explore/fraction，run161=3001/2210/0.736421，run162=2877/2124/0.738269，run163=2986/2186/0.732083。explore分支greedy→final Hamming mean/median由run161的4.2407/4降到run162的0.8583/1，run163保持0.8582/1，最大值仍为1且replacement slot恒为1；joint-action unique ratio为0.953846/0.927966/0.925892，探索率与动作多样性仍保留。topology-clean全样本explore/greedy一步progress与retreat：run162为+0.00970/25.38%和+0.06958/23.33%，run163为+0.03711/22.45%和+0.09524/20.08%。执行层未出现legality、dead-slot或non-explore greedy mismatch。
- good-start失败样本的限定结果：d0<=4且未strict distinct时，run161 explore/greedy progress与retreat为-0.11619/33.42%和+0.11787/20.53%；run162为-0.05217/23.62%和+0.07965/18.58%；run163为-0.04354/24.63%和+0.02692/21.54%。run163略有局部回落，但仍远好于run161的all-alive joint扰动，不构成再次调整exploration概率或slot数量的证据。
- post-capture persistence（mean d0-dk）：run161在+1,+2,+4,+6,+8,+10,+12,+16,+20,+24为0.092,0.143,0.014,-0.094,0.066,0.098,0.106,-0.223,-0.336,-0.472；run162为-0.107,-0.121,-0.022,0.104,0.008,0.385,0.109,0.126,0.424,0.504；run163为-0.014,-0.137,0.331,0.453,0.391,0.667,0.550,0.876,0.896,0.748。run163在+10以后没有恢复持续回撤，失败子集+24也为0.690，高于run162的0.470；bounded exploration→较少协调破坏→较强persistence链继续成立。
- first-capture geometry成为最早已确认leading indicator：remaining prey最近距离的mean/median及far share（d>=8）从run161的7.681/6/39.7%、run162的7.376/5/38.3%改善到run163的6.966/5/31.8%，但三个run的far-start strict distinct均为0。run163分层为：0--2: 5/26=19.23%、3--4: 6/38=15.79%、5--7: 4/37=10.81%、8+: 0/47=0%。因此当前最早可观测瓶颈已经前移到first capture发生时remaining prey仍处于far-start，而不是first capture后的persistence。
- 策略/Q分类边界：run163 topology-clean greedy progress为正、+10后persistence更强，故未发现整体greedy执行退化；Q loss/Q std/Q grad/RNN grad的mean为3.096/1.717/1.768/0.652，max为11.197/3.750/6.024/3.716，均有限且gradient clip fraction为0，不符合run157/run159式Q崩溃。当前旧日志没有first capture前逐动作Q ranking/message margin/factor identity，因而无法继续把far-start归因到具体agent动作排名、joint Q还是factor attribution；在这一证据缺口闭合前，reward/loss/credit assignment均未被判定为根因。
- pair事务事实：run163有5个非零pair cohort（35.2k、52.8k、63.2k、76.0k、78.4k），每个完成2个epoch，共10个非零transaction行；全部pair gradient finite、rollback=0、partition valid，dropped/deferred chunk均为0。adj_sample_recoverable_noop_chunk_count全程为0，因此run163只证明v41 no-op接线没有干扰正常生产事务，未形成真实训练触发该no-op分支的频率证据。
- 本轮源码修改只闭合已暴露的观测缺口，不改变优化目标或动作采样：rSDDFGPolicy.greedy()保存产生最终anytime greedy joint action时已经计算的message utilities，并返回每个agent的best-vs-runner-up有限margin；少于两个有限合法动作的slot记为NaN。WolfpackRunner为每个first capture增量落盘progress_train_pre_capture_32step.csv，只保留inclusive t-32..t，明确记录action_s_t -> info_s_(t+1)对齐、真实位置/距离、greedy/selected action、joint Q、message margin、factor Q和active factor身份。实现不增加policy/environment forward，不调用任何RNG，不修改terminal credit、bounded exploration、reward、loss或checkpoint contract。launcher增加pre-capture diagnostics schema v1的来源声明。
- before/after验证边界：run163真实文件只有post-capture 0..24窗口，无法反推出缺失的pre-capture Q ranking/factor identity，这正是日志确认的阻断；修改后fixture用生产字段检查33行窗口严格为-32..0、capture-step对齐、距离、factor成员、joint Q/margin序列化，同时现有joint epsilon fixture继续按固定seed重建branch/slot/legal action并通过。该修改已证明trajectory-neutral和格式正确，但未被写成性能提升。
- 已完成测试：PyTorch1.8.1兼容环境下Python compile、joint epsilon/action legality/RNG fixture、15项n-step、6项terminal lane、checkpoint contract/round-trip、pending launcher、CPU/CUDA transaction replay、CPU/CUDA production integration、pair/capture/outcome/candidate/provenance/stale-trust/eval/topology全部launcher preflight、dynamic graph/full-policy gradient及git diff --check全部通过。测试通过仅证明代码与契约正确，不证明run级性能改善。
- 新增成功经验：run162建立的bounded slot=1收益在run163长期统计与persistence上复现；terminal-gated completion链在training/eval仍产生<=24 distinct→win；trajectory-neutral诊断应复用现成Q/factor/action/adj结果，而不是为日志额外forward。
- 新增失败/排除项：run163不是相对run162的单变量性能提升实验；eval reward/capture回落且training win由23降至19，不能归类为继续提升。执行机制、动作合法性、random slot数量、post-capture长期回撤、terminal credit失效和Q数值崩溃均已由本run证据排除为第一根因；far-start内部的具体policy ranking/factor attribution仍未由run163旧日志确认。

### 第11次整理：run164首捕前轨迹审计与双prey reward目标修正

- 实验有效性：run164为seed=1、fresh（`model_dir`为空）、target/actual=80000/80000的完整正式训练；summary、带run前缀config与27份既有CSV、TensorBoard、逐文件manifest、terminal模型、optimizer checkpoint和新增`progress_train_pre_capture_32step.csv`均存在。checkpoint真实反序列化确认terminal/complete/80000，Q target v4、n-step=24、terminal-gated、terminal replay lane与0.10 auxiliary weight，以及exploration v4、epsilon 1.0→0.05/228000、post-capture floor=0.25、bounded random slots=1均未变。console/CSV没有真实Traceback、RuntimeError、AssertionError、ERROR、NaN/Inf、restore/resume、checkpoint mismatch、strict-pair failure或rollback。
- run163→run164是干净诊断实验：manifest只改变`rSDDFGPolicy.py`、`wolfpack_runner.py`、launcher和joint-exploration fixture四个诊断相关文件。归一化run id/path/manifest后，27份共同CSV逐单元完全相同；1456个TensorBoard scalar tags、81523个scalar点及其step/value序列完全相同；adj/Q/RNN模型tensor完全相同，optimizer的对象id按param-group位置归一化后parameter state、Adam moments/step、RNG、pending与retention payload也完全相同。因此80k没有第一行为分叉，新增pre-capture诊断获得了production级trajectory-neutral证据。
- 整体性能与run163完全一致：368个formal episode、148个capture/first-capture episode、209次capture（0.5679/episode）、15个strict distinct且全部interval<=24、19个training win；first→strict为10.14%。80k eval的reward/capture/win为-1.2860001/1.70/0.30。run162对应204 captures、141 first、16 strict、23 win和1.588/2.70/0.30；run164没有形成新增性能提升，只提供了可归因诊断。
- far-start复现：capture-time remaining-prey距离0--2、3--4、5--7、8+分别为26/5 strict/6 wins、38/6/7、37/4/5、47/0/1；转化率19.23%、15.79%、10.81%、0%。strict interval按四层分别为`[2,3,7,16,22]`、`[1,4,8,14,16,20]`、`[12,14,17,21]`、空。far-start仍然是strict distinct的硬失败层。
- geometry窗口边界：新增文件包含148次first capture、4632条transition、offset -32..0；128个episode具有完整t-32起点，其余为短episode截断。far与near的remaining-prey最近距离在窗口首点t-32已经为11.878 vs 6.246（Mann-Whitney p=8.27e-6），到t-24为12.318 vs 5.850、t-16为13.261 vs 4.672、t-8为13.870 vs 4.159、t为13.830 vs 2.625。因分叉早于已记录窗口，run164不能把原始geometry形成时刻伪写成t-32内某一步；对strict success/failure的独立对齐则从t-20开始稳定显著（remaining distance 4.133 vs 8.464，p=0.00883）。
- 双侧空间分工差异：near/far在t-32平均有1.754/0.829只agent对remaining prey比对首捕目标更近，t-16为1.885/0.717，t为1.953/0.660；balanced two-prey coverage cost在t-32为10.667/16.805，t-16为8.115/16.870，t为3.625/14.830。far不是首捕瞬间突然拉远，而是窗口开始前就缺少remaining-side agent，并在窗口内继续维持单目标聚集。
- 最早可干预策略指标：以每步state中距离remaining prey最近的alive agent作为B-side角色，t-31..t-24的8步rolling greedy directional progress，near/mid/far为+0.242/+0.115/+0.012（far-vs-near p=0.0259）；扩展到t-31..t-17为+0.194/+0.122/-0.076（p=0.00505），t-16..t-9为+0.207/+0.297/-0.190（p=0.000326）。同一角色的greedy“朝首捕目标推进且不朝remaining prey推进”比例在t-31..t-17为near 17.9%、far 29.0%，t-16..t-9扩大为16.6% vs 34.2%。因此新leading indicator是`B-side 8-step greedy remaining-prey progress <=0`及其target-over-remaining action share，而不再只是capture-time distance>=8。
- exploration被排除为第一根因：pre-capture explore fraction near/mid/far为0.721/0.725/0.793，Hamming约2.97/2.93/2.87；但far全窗口greedy/selected/实际progress为-0.123/-0.084/-0.067，随机执行平均反而把负greedy略向零拉回。far非探索transition的greedy实际progress仍为-0.181；t-31..t-17 far-vs-near的episode级greedy差异显著而selected差异较弱。由此不能把post-capture bounded exploration复制为本轮修复，真正异常已经存在于greedy ranking。
- message margin不是独立leading break：字段语义是同一agent有限合法动作message utility的top1-top2，因此理论与实测均非负；alive角色无NaN，4484个B-side transition只有1个精确零、无负值。B-side near/mid/far全窗口mean/median/p25/p75为0.02701/0.02009/0.00808/0.03734、0.02809/0.02071/0.00902/0.04030、0.02362/0.01725/0.00741/0.03357；<=0.005比例16.0%/14.4%/16.5%。low-margin quartile与high-margin quartile的未来+4 progress为0.154/0.220、far率34.9%/27.4%、strict率8.3%/11.5%，有弱相关但没有先于t-32 geometry的稳定断点，不能单独修改message。
- greedy consistency排除振荡解释：B-side greedy action switch/reversal/immobile episode均值near为0.563/0.024/0.330，far为0.492/0.033/0.318；far没有更高switch或stay，反而更稳定地选择偏向首捕目标的动作。strict success轨迹reversal为0，failure为0.027，但该差异不能解释far cohort最早形成。
- joint/factor Q分类：far joint Q在t-28/-24/-16/-8/t为0.228/0.248/0.242/0.261/0.266，near为0.272/0.291/0.312/0.319/0.337；按training-step quartile控制后far仍普遍更低，故不存在“明显坏geometry被绝对高估”的简单value bug。但far从t-16到首捕有63.0% episode的joint Q上升，平均+0.0265，同时remaining distance由13.261恶化到13.830，说明即时首捕价值仍能压过双目标geometry。B-side learned-factor contribution在t-16 near/far为0.1030/0.0751，capture-participant-related contribution为0.1496/0.1029，二者共同缩小，没有单一factor异常抬高。
- active factor identity不是更早断点：set-level retention near/mid/far为0.446/0.454/0.495，replacement transition率0.920/0.922/0.838；first replacement median三组均为t-31，slot identity lifetime median均为1且>=4比例约2.2%--2.4%。far factor保持不比near差，geometry分叉也早于窗口中任何可观察replacement，因此“B-side factor过早消失”被排除为当前第一根因。
- 确认的完整链：窗口开始前双侧分工已经不足→B-side角色的greedy长期更偏向即将首捕的prey→remaining distance不能闭合且far joint Q仍随首捕临近小幅上升→first capture形成d>=8→已验证的post-capture persistence只能稳定追击、无法补回过大初距→47个far-start均无strict distinct。message ambiguity、factor replacement和pre-capture exploration不是该链中最早的异常层。
- reward目标根因：run164使用legacy `independent_nearest_alive_prey` potential，每只wolf独立追逐自己的最近alive prey，不要求两只prey同时至少有一只wolf覆盖；这会对全队聚向同一较近prey给正shaping。真实far episode 259、offset -1为non-explore且greedy=executed：旧potential cost从10降到8，team shaping=+0.02；同尺度的balanced alive-prey coverage cost从17升到18，team shaping=-0.01。全窗口共有27条“legacy正、coverage负”transition，其中19条来自13个far episode，8条greedy=executed。该符号反转把greedy ranking异常直接追到training reward，而非理论上的“reward sparse”猜测。
- 本轮只修改一个主要性能语义：production launcher新增`--use_multi_prey_coverage_shaping`，把距离reward从每狼独立nearest-prey potential切换到要求每个alive prey至少分配一只wolf的最小总代价potential；scale仍为0.01，只有reward改变，不向policy observation加入prey身份/位置，不改变环境难度、动作、loss、epsilon、n-step、terminal lane或factor selection。single-prey时该potential退化为原sum-of-agent-distance，capture/topology后的baseline仍重建，active-slot reward均分且dead slots保持0。
- 新reward语义新增checkpoint/summary/manifest fail-loud合同：optimizer checkpoint保存version=1、mode和distance scale；启用coverage时旧checkpoint缺合同直接要求fresh，mode/scale不一致拒绝restore；terminal summary写出`reward_shaping`；launcher console与source manifest记录模式、helper、真实run164 fixture和回归脚本。
- before/after fixture：`scripts/fixtures/run164_pre_capture_reward_conflict.json`固定episode 259 offset -1的真实food/player位置与动作；环境级调用验证旧`update_food_status()`为+0.02、新路径为-0.01、没有capture、active slots均分/dead slots为0、reward有限、NumPy RNG逐项不变，并验证checkpoint缺失/mismatch fail-loud与launcher恰好启用一次新模式。
- 已完成实现测试：Python `py_compile`/`compileall`、reward flag parse、run164环境级fixture、joint exploration/action legality/RNG、15项n-step、6项terminal lane、CPU/CUDA transaction replay、17项pending foundation、9项cohort overlap、CPU/CUDA production integration与完整pair optimizer diagnostics、12项pair credit、12项capture outcome、12项capture identity、21项outcome replay、12项cohort centering、10项confidence、21项factor loss、53项candidate、12项provenance、17项stale trust、5项eval graph、5项topology persistence、dynamic graph/full-policy/CUDA gradient以及`git diff --check`全部通过。本机没有bash可执行文件，未单独执行`bash -n`；launcher source contract测试和正式preflight路径均已覆盖新fixture。测试通过只证明实现与契约正确。
- 当前代码状态：balanced multi-prey coverage reward已启用并通过实现回归；它对first→strict、win、reward和far-start比例的真实性能影响尚未由任何新正式run验证，不能提前记为性能成功。

### 第12次整理：run165 balanced coverage production验证与capture-quorum目标修正

- run165为fresh seed=1完整80k：actual/target均为80000，`model_dir`为空，training complete、terminal checkpoint、optimizer、TensorBoard、summary/config/manifest和核心CSV齐全。Q/terminal/exploration契约保持不变；`use_per=False`，alpha/beta只是未启用参数。run164→run165的主要训练语义变化只有legacy nearest-prey shaping切换为`balanced_alive_prey_coverage`，scale仍为0.01；policy、Q loss、factor、dynamic graph和exploration源码hash未变。
- production结果：368 formal episodes、61 first captures、73 captures（0.198/episode）、2 strict distinct、first→strict 3.28%、2 training wins；最后eval capture/win为0.8/0。相比run164的148/209/15/10.14%/19，basic capture与completion同步明显退化。已捕获子集的time-to-first mean基本不变（119.83→118.03），主要问题是大量episode不再形成首捕。
- 距离分层0--2、3--4、5--7、8+分别为`9/0 strict/0 wins`、`11/1/1`、`12/1/1`、`29/0/0`；far-start占比31.8%→47.5%，distance mean/median/p25/p75从6.97/5/3/10变为8.15/7/4/11。far `t-16..-9` B-side greedy progress从-0.190降为-0.229；target-over-remaining仅34.2%→31.2%，没有转化为remaining-side接近。
- Reward production曝光成立：首个诊断episode的step 51已出现有意义reward差异，formal episode 37（step 7400）首次出现episode级动作/结果分叉，首个已落盘Q batch（9600）亦不同。stable pre-capture balanced team shaping mean/std/mean-absolute/range为0.00262/0.02342/0.01796/`[-0.08,0.08]`；replay normalization std均值0.06114，信号没有被scale或normalization抹除。
- 数值路径有限但方差升高：run165 q-target std均值2.73、TD abs mean 0.682、policy grad norm均值2.54，对应run164为1.72/0.503/1.91；所有训练指标有限，无NaN/Inf或gradient failure。far状态中`P(Q rises | balanced coverage worsens)=52.9%`，与run164的52.6%基本相同；far joint Q从t-16到首捕仍有59.3%上升，reward→value→geometry没有闭合。方差上升是capture稀缺及normalization变化后的下游现象，不是第一根因。
- 新leading indicator：run165 `t-16..-9`中，旧balanced objective的所有最优assignment有62.6%的far窗口仍是`1+(N-1)`，failure为55.5%，strict仅6.25%。实际capture要求同一prey附近至少2只wolf；旧目标只保证每只alive prey有1只wolf。far transition还有26.1%出现“B-side不前进但balanced shaping仍为正”，capture-target侧总代价改善仍可补偿remaining-side退化。balanced cost quartile与strict/win关系弱，本轮分类为F（reward objective不贴合capture quorum），而非PER、exploration、factor、Q loss或scale问题。
- 本轮只修改reward objective：新增`capture_quorum_balanced_alive_prey_coverage_cost`，按字典序先最大化达到2-wolf quorum的prey数，再最大化一般coverage prey数，最后最小化总距离；2/3/4+ wolves分别形成2+0、2+1、至少2+2。production environment已切换该reward-only helper；observation、RNG、scale、loss、factor和exploration不变。
- 真实before/after fixture来自run165 episode 139 offset -20的non-explore far transition：B-side greedy action使remaining progress=-1；旧目标cost 43并列最优，新quorum目标cost 50，而progress=+1的actions 1/2以cost 48成为最优。新增`run165_capture_quorum_reward_conflict.json`固定该trajectory。
- Reward checkpoint contract升为version 2、mode=`capture_quorum_balanced_alive_prey_coverage`。同时确认并修复大小写缺陷：实际配置使用`env_name=wolfpack`，旧合同只识别`Wolfpack`，因此run165 checkpoint真实写入version 0；当前代码会对该旧checkpoint fail-loud并要求fresh。该缺陷不影响run165 fresh训练过程，但旧checkpoint不能安全resume。
- run165 production中368条transaction rollback=0；92条pending abort/rollback=0，recoverable no-op=0，因此只能记为本run未触发。reward fixture/RNG/checkpoint、py_compile/compileall、15项n-step、6项terminal replay、bounded exploration/action legality/dead slot、CPU/CUDA transaction、pair pending、dynamic graph/CUDA gradient及git diff check通过。本机无bash，`bash -n`未执行；launcher source contract/preflight测试通过。
- 当前代码状态：capture-quorum balanced coverage已实现并通过fixture/regression；它对basic capture、far-start、strict/win和Q方差的真实性能影响尚未由新正式run验证，不能记为性能成功。

### 第13次整理：run166 capture-quorum reward验证与pre-capture可见性断点

- run166为fresh seed=1完整80k正式训练：actual/target均为80000，`model_dir`为空，training complete且保存terminal模型、optimizer checkpoint、TensorBoard、summary/config、逐文件manifest和全部核心CSV。checkpoint与summary共同确认reward contract v2、`capture_quorum_balanced_alive_prey_coverage`、scale=0.01；Q target v4、n-step=24、terminal-gated、terminal replay lane、auxiliary weight=0.10；exploration v4、post-capture floor=0.25、bounded random slots=1均未变。reward normalization真实启用；`use_per=False`，alpha/beta只作为未启用参数存在。
- run165→run166的manifest差异只涉及reward helper、Wolfpack environment、reward checkpoint contract、launcher、reward fixture及新增run165真实fixture；policy、Q/factor/graph/exploration源码hash不变。首个已落盘trajectory reward在episode step 1已经不同；首个episode级action trace分叉在formal episode 37、step 7400；首个落盘Q batch在step 9600不同。该顺序符合reward先改变target、随后改变Q/action的单变量链，没有发现其它训练语义混杂。
- basic capture从run165的大幅坍缩中部分恢复：368 formal episode中有99个first-capture episode、138次capture（0.375/formal）、8个strict distinct且8/8 interval均<=24、9个training win；first→strict为8.08%。run164/run165/run166的first/captures/strict/wins依次为148/209/15/19、61/73/2/2、99/138/8/9。run166最后eval capture/win为1.4/0.2，best eval win为0.3@40k；reward contract不同，跨run不以eval reward绝对值作主判据。first-capture time mean/median为117.67/119，99次首捕participant均为2。
- run166 first-capture remaining-prey距离0--2、3--4、5--7、8+分别为15/5 strict/5 wins、24/2/3、20/1/1、40/0/0；转化率33.33%、8.33%、5.00%、0%。far-start由run165的47.5%降至40.4%，但仍高于run164的31.8%；distance mean/median/p25/p75为7.95/6/3/12.5。capture-quorum reward改善了run165 geometry，但尚未恢复run164水平，far-start仍是strict hard-failure band。
- reward objective真实改变assignment可行集：run166 `t-16..t-9` 784个窗口状态按当前objective重建，roster 4全部为2+2（247/247），roster 5全部为2+3（160/160），roster 6为2+4 171、3+3 34、并列10；旧1+coverage反事实在相同roster-4状态中只有55个2+2、177个1+3和15个并列。far 320个状态中当前objective为2+2 96、2+3 61、2+4 78、3+3 6、并列2、1+2 71、0+2 6，证明N>=4时旧1+(N-1)不可捕获目标已被消除。这里的assignment是reward最优匹配标签，不等同于policy长期身份分配。
- 核心行为同步改善但未闭合：far `t-16..t-9` B-side最近agent greedy progress从run164/run165的-0.190/-0.229变为-0.0219，target-over-remaining share从34.2%/31.2%降为22.2%；near/mid/far greedy progress为+0.282/+0.079/-0.0219，没有破坏near。可是far的两名最近quorum成员平均greedy progress仍为-0.0438，第二名为-0.050，双双正向仅10%；roster 4的第二名为-0.153、roster 5为-0.127，说明即使objective强制2+2，第二个remaining-side成员仍未形成稳定推进。
- capture-ready真实geometry给出新的最早断点：strict轨迹remaining-prey最近/第二近距离在t-16为3.625/6.875、t-8为3.125/6.250、t为2.500/4.375；far轨迹为13.450/17.975、14.400/18.475、14.300/18.350。按policy vector observation的精确L1<=8视野重建，far在t-32只有25.6%至少一名wolf可见remaining prey、10.3%有两名可见，到t-8降为7.5%/0%；strict在t-32为75%/25%，t-16已经100%/87.5%，t时100%/100%。因此run166的t-32窗口开始时两组已经分叉，第二quorum成员失去remaining-prey可见性发生在旧窗口之前，不能把根因伪写成t-16内某个reward/Q/factor动作。
- message/factor/exploration不是当前最早独立断点：far第二B-side message margin mean/median/p25/p75为0.02367/0.01761/0.00672/0.03247，与near的0.02402/0.01835/0.00853/0.03378接近；far B-pair/target-pair learned-factor覆盖为85.9%/87.2%，near为80.1%/83.0%，不存在B-side factor缺失。far B-pair factor-Q 0.0361低于target-pair 0.0417，但更像已形成几何/价值偏差的结果。pre-capture selected progress与greedy均未形成稳定双成员推进，现有32步数据又缺失最后可见transition，尚不能在greedy ranking与executed exploration之间作唯一归因。
- reward→Q已有弱改善但仍不强：run166 far `P(Q rises | capture-quorum cost worsens)=50.0%`，`P(Q rises | cost improves)=53.28%`，比run165近乎无区分略好，但差距仅3.28个百分点。Q/TD/gradient稳定：24个真实train batch的q-target std、TD abs mean、policy grad norm、Q grad norm均值为1.720/0.510/1.811/1.716，接近run164且显著低于run165的2.731/0.682/2.539/1.956；terminal占位行的空值不作为production NaN。raw pre-capture team shaping mean/std/mean-absolute/range约为-0.00091/0.03398/0.01963/`[-0.33,0.23]`，reward-normalization std均值0.0706，signal既未消失也未数值爆炸。
- 分阶段结果揭示后期平台：0--20k、20--40k、40--60k、60--80k的first为5/16/27/51，far share为80.0%/43.8%/29.6%/41.2%，strict为0/2/5/1，wins为0/2/6/1。40--60k是当前最佳阶段，60k后basic capture继续增加但far与strict转化回落；这不是Q/gradient爆炸，而是第二quorum成员frontier没有随训练继续稳定。
- 本轮分类为A（capture-quorum reward方向有效但只部分闭合）：assignment可行性、B-side greedy、far-start、first capture、strict和win相对run165同步改善，因此不继续叠加reward。已确认的第一性能根因是old diagnostic horizon不足以覆盖remaining-prey最后可见transition；高概率的行为瓶颈是第二quorum成员在t-32前离开remaining-prey可观察frontier后，后续局部policy无法维持2-agent pursuit。最后可见时究竟先由greedy ranking、exploration还是factor identity触发，run166仍未确认。
- 本轮未修改production reward/loss/factor/exploration。真实far失败transition的frontier-gap reward反事实虽能解除4/241个current-v2 action tie，但在241个第二成员nonprogress transition上，当前v2有70.1%至少一个progress action更优，frontier反事实只有58.1%；因此该reward叠加被数据否决，没有进入代码。
- 唯一代码修改是trajectory-neutral诊断：Wolfpack environment在既有`info`中新增每只alive prey按精确vector observation L1视野得到的`food_visible_player_slots`；pre-capture schema升为v2并保留原`t-32..t` CSV，同时新增`progress_train_pre_capture_prefix.csv`，从formal episode起点保存到first capture的完整前缀。字段复用已计算的position/action/joint-Q/message-margin/factor-Q/active-factor，不增加policy/environment forward，不消费RNG，不改变policy observation、reward、replay、loss、checkpoint或action execution。
- before/after fixture保持旧33行兼容窗口严格为-32..0，并用40步真实形态输入验证新prefix为40行、offset -39..0；同时检查可见slot序列化、observer count、capture-step对齐、joint Q/factor/margin、action legality、dead slot和RNG不变。该修改只证明诊断实现正确，性能尚未由新run验证。
- run166 production有368个adj transaction，rollback、actual-direction guard、target/boundary guard和recoverable no-op均为0；console preflight触发并通过recoverable no-op fixture，但production没有触发，不能写成真实训练频率验证。console/CSV无真实Traceback、RuntimeError、AssertionError、ERROR、Inf、checkpoint mismatch或restore/resume。
- 已完成测试：PyTorch1.8.1+cu111环境下`py_compile`/`compileall`、capture-quorum reward/RNG/checkpoint fixture、15项n-step、6项terminal replay、joint exploration/bounded slot/action legality/dead slot/full-prefix fixture、pending launcher、CPU/CUDA transaction replay、17项pending foundation、CPU/CUDA production integration、完整pair optimizer diagnostics（含recoverable no-op continuation与non-finite fail-loud）、dynamic graph/factor-Q/full-policy/CUDA gradient及`git diff --check`全部通过。测试通过只证明实现和训练合同正确，不证明性能提升。
- 当前代码状态：capture-quorum reward保留为production训练语义；新增完整pre-capture prefix与精确visibility诊断已实现并通过fixture/regression。诊断的trajectory-neutral性和“最后可见transition”的真实性能归因尚未由新正式run验证，不能提前记录为性能成功。
