import copy
import pickle as pkl
import numpy as np
import gym
from gym import spaces
import time
import os
from .assets.Agent import DQNAgent
from numpy.random import RandomState
from utils.wolfpack_reward import (
    WOLFPACK_DISTANCE_SHAPING_SCALE,
    capture_quorum_balanced_alive_prey_coverage_cost,
    coverage_potential_rewards,
    terminal_win_reward_components,
)


"""
Generator：用元胞自动机生成随机障碍地图（可通关检查联通性）。
- 关键参数：
  - deathLimit / birthLimit 控制“细胞死亡/出生”的阈值，影响障碍密度。
  - probStartAlive 初始为 True 的概率（True 代表障碍）。
- 核心流程：
  1) initialiseMap：按概率初始化布尔网格。
  2) doSimulationStep：根据邻居数量做一次迭代（类似生命游戏规则）。
  3) simulate：反复生成，直到 flood fill 判断“可通达区域连通”。
"""
class Generator(object):
    def __init__(self, size, deathLimit=4, birthLimit=3, seed=0):
        self.x_size = size[0]
        self.y_size = size[1]
        # booleanMap[y][x]：True 表示“障碍/填充”，False 表示“可走”
        self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
        self.probStartAlive = 0.82  # 初始为 True 的概率（越大障碍越多）
        self.deathLimit = deathLimit
        self.birthLimit = birthLimit
        self.seed = seed
        self.prng = RandomState(seed)
        self.copy = None

    """方法作用：按概率随机初始化地图。
        步骤：
          1) 遍历每个格点；
          2) 以 probStartAlive 的概率置为 True（障碍）。
    """
    def initialiseMap(self):
        for x in range(self.x_size):
            for y in range(self.y_size):
                if self.prng.random.uniform() < self.probStartAlive:
                    self.booleanMap[y][x] = True

    """方法作用：对整张图应用一次元胞自动机规则，更新障碍/空白。
        步骤：
          1) 计算每个格点的“活邻居数”（True 的邻居数量）；
          2) 若当前为 True(障碍)：邻居数 < deathLimit 则变 False，否则保持 True；
          3) 若当前为 False(空)：邻居数 > birthLimit 则变 True，否则保持 False。
     """
    def doSimulationStep(self):
        newMap = [[False] * k for k in [self.x_size] * self.y_size]
        for x in range(self.x_size):
            for y in range(self.y_size):
                alive = self.countAliveNeighbours(x, y)
                if self.booleanMap[y][x]:
                    if alive < self.deathLimit:
                        newMap[y][x] = False
                    else:
                        newMap[y][x] = True
                else:
                    if alive > self.birthLimit:
                        newMap[y][x] = True
                    else:
                        newMap[y][x] = False
        self.booleanMap = newMap

    """方法作用：统计 (x,y) 周围 8 邻域中为 True 的数量；边界外视为 True（促使边缘更“封”）。
        步骤：
          1) 遍历偏移 (-1..1, -1..1) 排除自身；
          2) 越界计为 1；未越界则读 booleanMap 计数。
    """
    def countAliveNeighbours(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbour_x = x + i
                neighbour_y = y + j
                if not ((i == 0) and (j == 0)):
                    if neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= self.x_size or neighbour_y >= self.y_size:
                        count = count + 1
                    elif self.booleanMap[neighbour_y][neighbour_x]:
                        count = count + 1
        return count

    """方法作用：反复随机生成并迭代，直至通过 flood fill 验证“空白区连通”。
        步骤：
          1) 重置并随机初始化；
          2) 进行 numSteps 次 doSimulationStep；
          3) 用 doFloodfill 检查是否所有 False 区域连成一片（可行走区域连通）；
          4) 若不连通则重新来过。
    """
    def simulate(self, numSteps):
        done = False
        while not done:
            self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
            self.initialiseMap()
            for kk in range(numSteps):
                self.doSimulationStep()

            if self.doFloodfill(self.booleanMap):
                done = True

    """方法作用：从第一个空白点开始 flood fill，若能覆盖所有空白点则连通。
         步骤：
              1) 深拷贝地图到 self.copy；
              2) 找到第一个 False 的起点；
              3) floodfill 标记所有连通的 False；
              4) 若仍有 False 未被覆盖 => 不连通，返回 False；否则 True。
    """
    def doFloodfill(self, newMap):
        self.copy = copy.deepcopy(newMap)
        foundX, foundY = -1, -1
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                if not self.copy[i][j]:
                    foundX = i
                    foundY = j
                    flag = True
                    break
            if flag:
                break
        self.floodfill(foundX, foundY)
        done = True
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                if not self.copy[i][j]:
                    done = False
                    flag = True
                    break
            if flag:
                break
        return done

    """方法作用：队列式 flood fill，把连通的 False 全部置 True（仅在 doFloodfill 内部使用）。
           步骤：
             1) 从起点入队；
             2) 反复出队，若位置为 False 就置 True；
             3) 将四邻域（上下左右）中为 False 的继续入队。
    """
    def floodfill(self, x, y):
        queue = []
        queue.append((x, y))
        while len(queue) != 0:
            a = queue[0][0]
            b = queue[0][1]

            del queue[0]
            if not self.copy[a][b]:
                self.copy[a][b] = True

            if (not a + 1 >= len(self.copy)) and (not self.copy[a + 1][b]):
                queue.append((a + 1, b))
                self.copy[a + 1][b] = True
            if (not (a - 1 < 0)) and (not self.copy[a - 1][b]):
                queue.append((a - 1, b))
                self.copy[a - 1][b] = True
            if (not b + 1 >= len(self.copy[0])) and (not self.copy[a][b + 1]):
                queue.append((a, b + 1))
                self.copy[a][b + 1] = True
            if (not b - 1 < 0) and (not self.copy[a][b - 1]):
                queue.append((a, b - 1))
                self.copy[a][b - 1] = True

"""环境作用：WolfPack（狼群围捕）开放队伍规模版本（回合内可加/减玩家）。
    关键特性：
      - 离散动作空间（共 7 个动作：前/右/后/左/停/左转/右转，其中实现把左转记为5，右转为6）。
      - 观测支持 vector / partial_obs / full_rgb（此处主要使用 vector）。
      - OpenScheduler 决定每个时间步是否“增减”玩家，实现回合内 N 变化。
      - 食物(prey)由预训练的 DQNAgent 控制，可被“协作围捕”后冻结并在冷却后复活,因此回合在最大时间步时终止。
      valid_indices 是“当前在场玩家的真实索引列表”（例如 [0,2,3] 表示 0、2、3 号玩家此刻在场；1 号玩家此刻缺席）。
      采用固定长度的观测槽位（max_player_num），一个内部索引表来把“真实玩家集合”与这些槽位对应起来
      - 奖励：
          * 靠近食物有微小正向 shaping（上一步距离-本步距离）*0.01；
          * 单狼贴近食物但未形成围捕 => close_penalty 惩罚,鼓励协作而不是抢占位置；
          * 多狼共同在 coopRadius 范围内夹击,该只食物记 -1 并冻结一段时间后再复活 => 正奖励，食物 -1 并冻结。
"""
class WolfpackPenaltyOpen(gym.Env):
    metadata = {'render.modes': ['human']}

    def _team_reward_components(self, individual_rewards, success_now):
        return terminal_win_reward_components(
            individual_rewards=individual_rewards,
            success_now=success_now,
            episode_success_before=self.episode_success,
            reward_win=self.reward_win,
        )

    def _make_food_agent(self, idx):
        agent = DQNAgent(
            agent_id=idx,
            args={
                "with_gpu": self.prey_with_gpu,
                "max_seq_length": 5,
                "seed": int(self.seed) + 100000 + int(idx),
            },
            obs_type="partial_obs",
            mode="test",
        )

        dirname = os.path.dirname(__file__)
        filename = os.path.join(
            dirname,
            "assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)
        )
        agent.load_parameters(filename)
        return agent

    def __init__(self, grid_height=20, grid_width=20, num_agents=3, max_food_num=2,
                 sight_sideways=8, sight_radius=8, max_time_steps=200,
                 coop_radius=1, groupMultiplier=2, food_freeze_rate=0,
                 add_rate=0.05, del_rate=0.05, seed=None,
                 max_player_num=5, obs_type="vector", with_random_grid=False, random_grid_dir=None,
                 prey_with_gpu=False, close_penalty=0.1,
                 intra_episode_dynamic=False, shock_steps="", shock_remove_num=0,
                 shock_join_delay=10, shock_join_num=0, shock_recover_delay=30,
                 dynamic_min_agents=2, continue_after_success=False,
                 reward_win=0.0, use_multi_prey_coverage_shaping=False):

        # ====== 基本网格/观测与动作配置 ======
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_type = obs_type
        self.close_penalty = close_penalty
        self.num_players = num_agents  # 当前在场玩家数
        self.max_player_num = max_player_num  # 最大可容纳玩家数（固定观测维度用）
        self.max_food_num = max_food_num

        self.ma_obs_type = obs_type
        self.N_DISCRETE_ACTIONS = 7
        self.n_actions = self.N_DISCRETE_ACTIONS

        self.intra_episode_dynamic = bool(intra_episode_dynamic)
        self.shock_steps = shock_steps
        self.shock_remove_num = int(shock_remove_num)
        self.shock_join_delay = int(shock_join_delay)
        self.shock_join_num = int(shock_join_num)
        self.shock_recover_delay = int(shock_recover_delay)
        self.dynamic_min_agents = int(dynamic_min_agents)
        self.continue_after_success = bool(continue_after_success)
        self.reward_win = float(reward_win)
        if not np.isfinite(self.reward_win) or self.reward_win < 0.0:
            raise ValueError("reward_win must be finite and non-negative")
        self.use_multi_prey_coverage_shaping = bool(
            use_multi_prey_coverage_shaping
        )

        # ====== 回合内动态事件参数校验 ======
        # 所有张量仍使用 max_player_num 作为固定容量；这里只改变槽位是否有效。
        if self.num_players > self.max_player_num:
            raise ValueError("num_agents cannot exceed max_player_num")
        if self.intra_episode_dynamic:
            if not 1 <= self.dynamic_min_agents <= self.num_players:
                raise ValueError("dynamic_min_agents must be in [1, num_agents]")
            if self.shock_recover_delay <= 0:
                raise ValueError("shock_recover_delay must be > 0")
            if self.shock_remove_num <= 0:
                raise ValueError("shock_remove_num must be > 0")
            if self.shock_join_delay < 0 or self.shock_join_num < 0:
                raise ValueError("shock join parameters must be >= 0")
            if self.obs_type != "vector":
                raise ValueError(
                    "intra_episode_dynamic currently requires obs_type='vector'"
                )

        self.action_space = [spaces.Discrete(self.n_actions) for _ in range(self.max_player_num)]
        # DDFG 一般会用 get_env_info()/get_avail_actions()，此处 action_space 仅作描述用途
        self.other_player_acts = None

        # ====== 观测空间（vector 版本）======
        if obs_type == "vector":
            # 单个 agent 的部分可观测向量构成（固定维度，slot 对齐，便于 DDFG 训练）
            # The simultaneous-prey objective depends on each prey's remaining
            # freeze window, so expose one normalized countdown per prey slot.
            self.unit_obs_dim = 2 + 4 + 2 * (self.max_player_num - 1) + 7 * self.max_food_num + 4 + 1

            self.observation_space = [[self.unit_obs_dim] for _ in range(self.max_player_num)]

        # 全局状态维度（固定长度 state_dim，share_obs 会复制 N 份）
        self.state_dim = 7 * self.max_player_num + 8 * self.max_food_num + 1
        
        self.share_observation_space = [[self.state_dim] for _ in range(self.max_player_num)]

        # ====== 随机/调度器/可视化状态 ======
        self.add_rate = add_rate  # 每步“增加玩家”的几何分布参数
        self.del_rate = del_rate  # 每步“删除玩家”的几何分布参数
        self.masking = []  # （预留）掩码
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        self.prng = RandomState(self.seed)
        # OpenScheduler：回合内增减玩家的“到达-离开”过程
        self.scheduler = OpenScheduler(self.num_players, self.add_rate, self.del_rate, self.max_player_num, 2,
                                       seed=self.seed)

        # 视野与 padded RGB 网格（为 partial_obs / full_rgb 服务）
        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.pads = max(self.sight_sideways, self.sight_radius)

        # Padded RGB matrix for preys that receive RGB inputs
        self.RGB_padded_grid = [[[0, 0, 255] for b in range(2 * self.pads + self.grid_width)]
                                for a in range(2 * self.pads + self.grid_height)]

        # Grid to save the locations of agents, preys and obstacles
        # 主 grid：0=墙/不可走未填；1=空地；2=玩家；3=食物
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if "full_rgb" in self.obs_type:
            # RGB grid without padding in case agents use RGB grids as input
            self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)]
                             for a in range(self.grid_height)]

        # ====== 初始玩家/食物数量与槽位管理 ======
        self.init_num_players = num_agents
        # valid_indices：长度=max_player_num；值=“该槽位对应的真实 players 列表下标”，-1 代表空槽
        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx
        self.prev_valid_indices = None

        # ====== 回合长度、食物冷却、缓存 ======
        self.max_time_steps = max_time_steps
        self.food_freeze_rate = food_freeze_rate
        self.other_player_obses = None
        self.all_player_acts = None
        self.other_player_acts = None

        # ====== 地图生成 ======
        self.with_random_grid = with_random_grid
        self.random_grid_dir = random_grid_dir

        if not self.random_grid_dir is None:
            self.levelMap = self.load_map(self.random_grid_dir)
        elif not self.with_random_grid:  # 默认全可走
            self.levelMap = [[False] * k for k in [self.grid_width] * self.grid_height]
        else:  # 自动机生成随机地图并连通性检查
            app = Generator((self.grid_width, self.grid_height), 7, 8)
            app.initialiseMap()
            app.simulate(2)
            self.levelMap = app.booleanMap

        # ====== 一堆运行时状态缓存 ======
        self.visualizer = None
        self.obstacleCoord = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if i]
        self.possibleCoordinates = None
        self.player_positions = None
        self.player_orientation = None
        self.food_positions = None
        self.food_alive_statuses = None
        self.food_frozen_time = None
        self.food_orientation = None
        self.player_points = None
        self.food_points = None
        self.a_to_idx = None
        self.idx_to_a = None
        self.prev_dist_to_food = None
        self.prev_team_coverage_cost = None

        self.player_obs_type = [obs_type for _ in range(self.num_players)]
        self.food_obs_type = ["partial_obs" for _ in range(self.max_food_num)]
        self.food_obses = None
        self.prey_with_gpu = prey_with_gpu

        # _make_food_agent 内部已经完成参数加载。
        self.remaining_timesteps = max_time_steps
        self.food_list = [self._make_food_agent(idx) for idx in range(self.max_food_num)]

        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.coopRadius = coop_radius              # 围捕时“近邻”半径（曼哈顿距离）
        self.groupMultiplier = groupMultiplier     # 协作奖励系数

    """保存随机生成的 levelMap。"""
    def save_map(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.levelMap, f)

    def load_map(self, filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    """方法作用：重置一轮游戏并返回所有潜在槽位的观测（长度=max_player_num）。
           步骤：
             1) 重置随机种子、调度器、网格与 RGB 缓存；
             2) 生成可走坐标列表 possibleCoordinates；
             3) 随机采样玩家、食物初始位置与朝向；
             4) 初始化奖励、冷却、计时器等状态；
             5) 构建每个槽位的 vector 观测（不存在的槽位会输出全 -1，末尾 exist_flag=-1）。
    """
    def reset(self):
        # 可视化/随机种子/调度器重置
        self.visualizer = None
        self.seed += 1250
        self.prng = RandomState(self.seed)

        self.num_players = self.init_num_players
        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx
        self.prev_valid_indices = None

        # ====== 初始化同一回合的动态成员状态 ======
        self.episode_step = 0
        self.episode_success = False
        self.pending_recovery = []

        # UID 仅用于统计身份，不输入策略网络
        self.slot_uids = [-1] * self.max_player_num
        for slot in range(self.init_num_players):
            self.slot_uids[slot] = slot
        self.next_agent_uid = self.init_num_players

        if self.intra_episode_dynamic:
            self.event_scheduler = IntraEpisodeEventScheduler(
                shock_steps=self.shock_steps,
                remove_num=self.shock_remove_num,
                join_delay=self.shock_join_delay,
                join_num=self.shock_join_num,
            )
            self.scheduler = None
        else:
            # 原随机回合内调度器完全保留
            self.event_scheduler = None
            self.scheduler = OpenScheduler(
                self.num_players,
                self.add_rate,
                self.del_rate,
                self.max_player_num,
                2,
                seed=self.seed,
            )

        # 重置网格、RGB
        self.RGB_padded_grid = [[[0, 0, 255] for _ in range(2 * self.pads + self.grid_width)]
                                for a in range(2 * self.pads + self.grid_height)]

        # Reset grid locations
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if "full_rgb" in self.obs_type:
            # Reset RGB Grid
            self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]

        # 可走坐标（levelMap 为 False 的位置可走）
        self.possibleCoordinates = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if
                                    not i]

        # 预计算可走格子集合（用于快速判断是否可移动）
        self._possibleCoordinates_set = set(self.possibleCoordinates)

        # 采样玩家位置（不重复）
        player_loc_idx = self.prng.choice(
            range(len(self.possibleCoordinates)), self.num_players, replace=False
        ).tolist()
        self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]

        # Reset initial player orientation and points
        self.player_orientation = [0 for a in range(self.num_players)]
        self.player_points = [0 for a in range(self.max_player_num)]

        # 采样食物位置（不能和玩家重叠）
        coordinates_no_player = [a for a in self.possibleCoordinates if a not in self.player_positions]
        food_loc_idx = self.prng.choice(range(len(coordinates_no_player)), self.max_food_num, replace=False).tolist()

        # Reset all food attributes
        self.food_obses = None
        self.food_positions = [coordinates_no_player[a] for a in food_loc_idx]
        self.food_alive_statuses = [True for a in range(self.max_food_num)]
        self.food_frozen_time = [0 for a in range(self.max_food_num)]
        self.food_points = [0 for a in range(self.max_food_num)]

        self.food_orientation = [0 for a in range(self.max_food_num)]

        # 将元素写回 grid / RGB
        for coord in self.possibleCoordinates:
            self.grid[coord[0]][coord[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [0, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [0, 0, 0]
        for coord in self.player_positions:
            self.grid[coord[0]][coord[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [255, 255, 255]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 255, 255]
        for coord in self.food_positions:
            self.grid[coord[0]][coord[1]] = 3
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [255, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 0, 0]

        # 初始化距离塑形基线。新机制只改变训练 reward，不改变 observation。
        self._reset_distance_shaping_baseline()

        self.remaining_timesteps = self.max_time_steps

        # 计算初始观测
        self.player_obs_type = [self.obs_type for _ in range(self.num_players)]
        player_obses = [self.observation_computation(self.ma_obs_type, agent_id=id) for id in
                        range(self.max_player_num)]
        food_obses = [self.observation_computation(obs_type,
                                                   agent_type="food", agent_id=id) for id, obs_type in
                      enumerate(self.food_obs_type)]
        self.food_obses = food_obses

        # 每回合重新实例化 prey，避免其序列状态跨 episode 泄漏；参数在工厂方法内加载。
        self.food_list = [self._make_food_agent(idx) for idx in range(self.max_food_num)]
        obs = np.asarray(player_obses, dtype=np.float32)
        share_obs = np.tile(self.get_state().astype(np.float32), (self.max_player_num, 1))
        avail_actions = self.get_avail_actions().astype(np.float32)
        return obs, share_obs, avail_actions

    """方法作用：让“冷却结束”的已死食物随机复活到不冲突的空地。
        步骤：
            1) 找出可复活的食物索引（alive_status=False 且 frozen_time<=0）；
            2) 在不含玩家与存活食物的位置里随机采样新坐标；
            3) 标记这些食物为存活并更新其位置；
            4) 更新 prev_dist_to_food（用于塑形奖励基线）。
    """
    def revive(self):
        """让“冷却结束”的已死食物随机复活到不冲突的空地。
        重要约定（为保持逻辑一致）：
          - 冻结期（alive_status=False）的 prey 在逻辑上视为“不存在”：
            * 不参与观测；
            * 不参与距离塑形；
            * 不占用格子（复活时也不排除其旧坐标）。
        """
        # 只把 alive 的 prey 当作占位
        alive_food_positions = [pos for i, pos in enumerate(self.food_positions) if self.food_alive_statuses[i]]
        coordinates_no_player = [a for a in self.possibleCoordinates
                                 if (a not in self.player_positions) and (a not in alive_food_positions)]

        revived_idxes = [i for i in range(self.max_food_num)
                         if (not self.food_alive_statuses[i]) and (self.food_frozen_time[i] <= 0)]

        if revived_idxes and coordinates_no_player:
            pick_num = min(len(revived_idxes), len(coordinates_no_player))
            idxes = self.prng.choice(range(len(coordinates_no_player)), pick_num, replace=False).tolist()
            coords = [coordinates_no_player[a] for a in idxes]
            for j, idx in enumerate(revived_idxes[:pick_num]):
                self.food_alive_statuses[idx] = True
                self.food_positions[idx] = coords[j]
                self.food_orientation[idx] = 0

        # 复活会改变 alive-prey 集合；禁止把该不连续变化记作 shaping。
        self._reset_distance_shaping_baseline()

    def update_status(self):
        """推进 prey 的冻结计时器（<=0 时可在 revive() 中复活）。"""
        for idx in range(len(self.food_alive_statuses)):
            if not self.food_alive_statuses[idx]:
                self.food_frozen_time[idx] = max(0, self.food_frozen_time[idx] - 1)

    def calculate_new_position(self, collectiveAct, prev_player_position, prev_player_orientation):
        zipped_data = list(zip(collectiveAct, prev_player_position, prev_player_orientation))
        result = [self.calculate_indiv_position(a, (b, c), d) for (a, (b, c), d) in zipped_data]
        return result

    """方法作用：单个实体根据动作和朝向更新位置/朝向（带越界/碰墙检测）。
        动作编码：
          0: 前进；1: 右移；2: 后退；3: 左移；4: 不动；5: 左转；(else): 右转
        步骤：
          1) 按当前朝向把动作映射到相对位移或朝向变化；
          2) 若目标位置在可走集合内，则更新；否则停留；
          3) 返回 (nx, ny, new_orientation)。
    """
    def calculate_indiv_position(self, action, pair, orientation):
        """根据动作计算下一步的位置与朝向（绝对移动，不依赖 orientation）。

        动作（离散 id）：
          0: 上移(绝对方向 Up)
          1: 右移(绝对方向 Right)
          2: 下移(绝对方向 Down)
          3: 左移(绝对方向 Left)
          4: 原地不动(Stay)
          5: 左转（只改变朝向，用于部分可观测视野方向）
          6: 右转（只改变朝向，用于部分可观测视野方向）

        说明：
          - 移动是否有效由 self._possibleCoordinates_set 判定；无效则回滚。
          - step() 输入动作为 one-hot/logits，这里保留兜底解码。
        """
        x, y = pair
        next_x, next_y = x, y

        # 兼容：one-hot / ndarray -> argmax（全 0 则 stay）
        try:
            if isinstance(action, (list, tuple, np.ndarray)):
                a = np.asarray(action).reshape(-1)
                action = int(np.argmax(a)) if np.any(a) else 4
            else:
                action = int(action)
        except Exception:
            action = 4

        # 绝对方向移动：0上 1右 2下 3左
        if action == 0:
            next_x, next_y = x - 1, y
        elif action == 1:
            next_x, next_y = x, y + 1
        elif action == 2:
            next_x, next_y = x + 1, y
        elif action == 3:
            next_x, next_y = x, y - 1
        elif action == 4:
            return (x, y, orientation)

        # 左转 / 右转：仅改变朝向（用于 partial_obs 的视野方向）
        if action == 5:
            return (x, y, (orientation - 1) % 4)
        if action == 6:
            return (x, y, (orientation + 1) % 4)

        # 如果是移动动作：检查是否撞墙/越界/障碍
        if (next_x, next_y) in self._possibleCoordinates_set:
            return (next_x, next_y, orientation)
        return (x, y, orientation)

    def update_food_status(self):
        """协作捕获 + 单狼惩罚 + 距离塑形（只对 alive prey 计算距离）。
        奖励构成（slot 级别）：
          1) 距离塑形：启用 coverage 时，先最大化具有捕获 quorum 的
             alive prey 数，再最大化被覆盖 prey 数，最后最小化总代价；
             否则保留 legacy per-wolf nearest prey；
          2) 单狼惩罚：若只有 1 只狼在 coopRadius 内贴近 prey，则该狼 -close_penalty；
          3) 协作奖励：若有 k>=2 只狼在 coopRadius 内，则捕获 prey：
             - 捕获事件的总奖励固定为 groupMultiplier（不随 k 增长）；
             - 每只参与围捕的 wolf 获得 groupMultiplier / k；
             - 同一步每个 prey 最多捕获一次（按 prey 遍历一次），避免重复加分。
        """
        self.food_points = [0 for _ in range(self.max_food_num)]
        self.last_capture_count = 0
        self.last_capture_events = []

        # 1) 距离塑形（slot）。Coverage potential 保持 sum-of-agent-distance
        # 的量纲，并在 active slots 间均分；trainer 使用其 team sum。
        cur_dist_to_food = [None] * self.max_player_num
        for rid in range(self.num_players):
            slot = self.valid_indices.index(rid)
            px, py = self.player_positions[rid]
            cur_dist_to_food[slot] = self._min_dist_to_alive_food(px, py)

        if self.use_multi_prey_coverage_shaping:
            current_coverage_cost = self._current_team_coverage_cost()
            if self.prev_team_coverage_cost is None:
                raise RuntimeError(
                    "Wolfpack coverage shaping baseline is not initialized"
                )
            active_slots = [
                slot for slot, rid in enumerate(self.valid_indices)
                if rid >= 0
            ]
            self.player_points = coverage_potential_rewards(
                previous_cost=self.prev_team_coverage_cost,
                current_cost=current_coverage_cost,
                active_slots=active_slots,
                max_player_num=self.max_player_num,
                scale=WOLFPACK_DISTANCE_SHAPING_SCALE,
            ).tolist()
            self.prev_team_coverage_cost = current_coverage_cost
        else:
            shaped = []
            for prev_d, cur_d in zip(
                    self.prev_dist_to_food, cur_dist_to_food):
                if prev_d is None or cur_d is None:
                    shaped.append(0.0)
                else:
                    shaped.append(
                        WOLFPACK_DISTANCE_SHAPING_SCALE * (prev_d - cur_d)
                    )
            self.player_points = shaped
        self.prev_dist_to_food = cur_dist_to_food

        # 2) 捕获/惩罚（按 prey 遍历，每个 prey 每步最多处理一次）
        real_to_slot = {rid: self.valid_indices.index(rid) for rid in range(self.num_players)}

        for fid in range(self.max_food_num):
            if not self.food_alive_statuses[fid]:
                continue

            fx, fy = self.food_positions[fid]

            close_real_ids = []
            for rid, (px, py) in enumerate(self.player_positions):
                if abs(px - fx) + abs(py - fy) <= self.coopRadius:
                    close_real_ids.append(rid)

            k = len(close_real_ids)

            # ==========================================
            # 2. 核心捕猎与惩罚逻辑 (课程学习阶段控制)
            # ==========================================
            if k >= 2:
                participant_slots = sorted(
                    int(real_to_slot[rid]) for rid in close_real_ids
                )
                self.last_capture_events.append({
                    "event_id": int(
                        self.episode_step * self.max_food_num + fid
                    ),
                    "target_id": int(fid),
                    "participant_slots": participant_slots,
                    "capture_position": [int(fx), int(fy)],
                })
                # 捕获 prey：标记死亡并冻结
                self.food_alive_statuses[fid] = False
                self.food_frozen_time[fid] = self.food_freeze_rate
                self.food_points[fid] -= 1
                self.last_capture_count += 1

                # 从 grid/rgb 上清除（冻结期视为“不存在”）
                self.grid[fx][fy] = 1
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[fx][fy] = [0, 0, 0]
                self.RGB_padded_grid[fx + self.pads][fy + self.pads] = [0, 0, 0]

                # 总奖励固定 groupMultiplier，均分给 k 个 close wolves
                per_agent = float(self.groupMultiplier) / float(k)
                for rid in close_real_ids:
                    self.player_points[real_to_slot[rid]] += per_agent

            elif k == 1:
                # 单狼贴近惩罚
                rid = close_real_ids[0]
                self.player_points[real_to_slot[rid]] -= float(self.close_penalty)

        # 3) 只绘制 alive prey（冻结期不显示）
        for idx, (fx, fy) in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[fx][fy] = 3
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[fx][fy] = [255, 0, 0]
                self.RGB_padded_grid[fx + self.pads][fy + self.pads] = [255, 0, 0]

    def update_state(self, hunter_collective_action, food_collective_action):
        self.other_player_acts = hunter_collective_action
        self.remaining_timesteps -= 1
        self.update_status()
        self.revive()

        # dead prey（冻结期）视为不存在：其动作强制为 stay，避免“隐形 prey 移动”
        food_collective_action = [a if self.food_alive_statuses[i] else 4
                                  for i, a in enumerate(food_collective_action)]

        # 位置与朝向更新（玩家与食物）
        prev_player_position = self.player_positions
        prev_player_orientation = self.player_orientation
        prev_food_position = self.food_positions
        prev_food_orientation = self.food_orientation
        # Notes notes
        update_results_player = self.calculate_new_position(hunter_collective_action, prev_player_position,
                                                            prev_player_orientation)
        post_player_position = [(a, b) for (a, b, c) in update_results_player]
        post_player_orientation = [c for (a, b, c) in update_results_player]
        self.player_orientation = post_player_orientation

        update_results_food = self.calculate_new_position(food_collective_action, prev_food_position,
                                                          prev_food_orientation)
        post_food_position = [(a, b) for (a, b, c) in update_results_food]
        post_food_orientation = [c for (a, b, c) in update_results_food]
        self.food_orientation = post_food_orientation

        # ==== 同格冲突处理：任意时刻，玩家/食物不能落在同一格 ====
        prev_positions = [None] * (len(prev_player_position) + len(prev_food_position))
        post_positions = [None] * (len(prev_player_position) + len(prev_food_position))
        types = [None] * (len(prev_player_position) + len(prev_food_position))
        food_status = [False] * (len(prev_player_position) + len(prev_food_position))

        for a in range(len(prev_player_position)):
            prev_positions[a] = prev_player_position[a]
            post_positions[a] = post_player_position[a]
            types[a] = "player"

        for a in range(len(prev_food_position)):
            prev_positions[a + len(prev_player_position)] = prev_food_position[a]
            post_positions[a + len(post_player_position)] = post_food_position[a]
            types[a + len(post_player_position)] = "food"
            if self.food_alive_statuses[a]:
                food_status[a + len(post_player_position)] = True

        # 找到多重落点（长度>1 的 group），把这些实体回滚到原位，直到没有冲突
        a, seen, result = post_positions, {}, {}
        for idx, item in enumerate(a):
            next_pass = True
            if types[idx] == "food" and not food_status[idx]:
                next_pass = False

            if next_pass:
                if item not in seen:
                    result[item] = [idx]
                    seen[item] = types[idx]
                else:
                    result[item].append(idx)

        groupings = list(result.values())
        doubles = [t for t in groupings if len(t) > 1]
        while len(doubles) > 0:
            res = set([item for sublist in doubles for item in sublist])
            for ii in range(len(post_positions)):
                if ii in res:
                    post_positions[ii] = prev_positions[ii]

            a, seen, result = post_positions, {}, {}
            for idx, item in enumerate(a):
                next_pass = True
                if types[idx] == "food" and not food_status[idx]:
                    next_pass = False

                if next_pass:
                    if item not in seen:
                        result[item] = [idx]
                        seen[item] = types[idx]
                    else:
                        result[item].append(idx)

            groupings = list(result.values())
            doubles = [t for t in groupings if len(t) > 1]

        # ====== 刷新网格显示（先清旧、后写新）======
        for a in self.food_positions:
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.food_positions = post_positions[len(post_player_position):]

        for idx, a in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[a[0]][a[1]] = 3
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[a[0]][a[1]] = [255, 0, 0]
                self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 0, 0]

        for a in self.player_positions:
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.player_positions = post_positions[:len(post_player_position)]

        for idx, a in enumerate(self.player_positions):
            self.grid[a[0]][a[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]

        # 计算奖励 & 生死状态
        self.update_food_status()

    """方法作用：生成某个“潜在槽位”的观测。
        vector 模式：
          - 先写自己 (x,y)，再按槽位顺序写其他玩家 (x,y)，再把所有食物的信息拼上（位置+朝向 one-hot），最后附 exist_flag。
          - 若该槽位无效（valid_indices[agent_id] == -1），整行置 -1，exist_flag=-1。
        partial_obs / full_rgb 模式（用于 DQN prey 或可视化）：
          - 根据朝向裁切一个扇区/矩形窗口返回。
    """
    def observation_computation(self, obs_type, agent_type="player", agent_id=0):
        if obs_type == "vector":
            # ==============================
            # 部分可观测 vector 观测（用于 DDFG）
            # ==============================
            # 仅对 player 生效：prey 默认使用 self.food_obs_type（通常是 partial_obs / full_rgb）
            if agent_type != "player":
                return np.zeros((self.unit_obs_dim,), dtype=np.float32)

            return self._observation_player_vector_partial(agent_id).astype(np.float32)

        elif obs_type == "partial_obs":
            # 以朝向为参考，从 padded RGB 中裁一块可见区域
            if agent_type == "player":
                orientation = self.player_orientation[agent_id]
                pos_0, pos_1 = self.player_positions[agent_id][0], self.player_positions[agent_id][1]
            else:
                orientation = self.food_orientation[agent_id]
                pos_0, pos_1 = self.food_positions[agent_id][0], self.food_positions[agent_id][1]

            pos_0 = pos_0 + self.pads
            pos_1 = pos_1 + self.pads
            obs_grid = np.asarray(self.RGB_padded_grid)

            if orientation == 0:  # 朝上：取上方扇区
                partial_ob = obs_grid[pos_0 - self.sight_radius:pos_0 + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]


            elif orientation == 1:  # 朝右
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1:pos_1 + self.sight_radius + 1]

                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = partial_ob[::-1]

            elif orientation == 2:  # 朝下
                partial_ob = obs_grid[pos_0:pos_0 + self.sight_radius + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
                partial_ob = np.fliplr(partial_ob)
                partial_ob = partial_ob[::-1]

            elif orientation == 3:  # 朝左
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1 - self.sight_radius:pos_1 + 1]
                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = np.fliplr(partial_ob)

            return partial_ob


    def step(self, action):
        # s_t 的成员集合。本步动作和奖励必须与事件发生前的成员对齐。
        active_masks_before = self._active_mask()

        # 1) 解码动作：slot -> 离散 id
        if isinstance(action, (list, tuple)):
            action_arr = np.asarray(action)
        else:
            action_arr = action

        hunter_collective_action = []
        for slot in range(self.max_player_num):
            try:
                a = action_arr[slot]
            except Exception:
                a = 4
            hunter_collective_action.append(self._decode_one_hot_action(a))

        # 2) 将“按槽位”的动作重排为“按真实玩家索引”的动作
        restructured_action = []
        for rid in range(self.num_players):
            slot = self.valid_indices.index(rid)
            restructured_action.append(hunter_collective_action[slot])

        # 3) prey 动作（dead prey 冻结期当不存在：动作 stay）
        food_collective_action = []
        for fid, prey in enumerate(self.food_list):
            if self.food_alive_statuses[fid]:
                food_collective_action.append(prey.act(self.food_obses[fid], epsilon=0.1))
            else:
                food_collective_action.append(4)

        # 4) 推进世界（内部会更新 self.player_points 等）
        self.update_state(restructured_action, food_collective_action)

        # 5) 保存 s_t,a_t 产生的 reward，然后改变下一状态的成员集合。
        reward_slot = np.asarray(self.player_points, dtype=np.float32).reshape(self.max_player_num, 1)
        individual_rewards = reward_slot * active_masks_before

        self.episode_step += 1
        self.prev_valid_indices = self.valid_indices.copy()

        if self.intra_episode_dynamic:
            # 脚本化事件：同一 episode 内突发退出、新成员加入、原成员恢复。
            event_info = self._apply_scripted_events()
        else:
            # 向下兼容：保留原随机 OpenScheduler 路径。
            before_event_mask = self._active_mask()
            deleted, new_types = self.scheduler.open_process()
            self.del_agent(deleted)
            self.add_agent(new_types)
            after_event_mask = self._active_mask()

            event_info = {
                "left_slots": np.where(
                    (before_event_mask[:, 0] == 1)
                    & (after_event_mask[:, 0] == 0)
                )[0].astype(np.int64).tolist(),
                "joined_slots": np.where(
                    (before_event_mask[:, 0] == 0)
                    & (after_event_mask[:, 0] == 1)
                )[0].astype(np.int64).tolist(),
                "recovered_slots": [],
            }

        # 成员变化或捕获会改变 potential 的定义域；从下一状态重建基线，
        # 避免 topology/capture discontinuity 伪装成距离 shaping。
        self._reset_distance_shaping_baseline()

        # 6) 下一观测（玩家侧）
        obs = np.asarray([self.observation_computation(self.obs_type, agent_id=i)
                          for i in range(self.max_player_num)], dtype=np.float32)

        # prey 观测（环境内部用，不返回）
        food_returns = ([self.observation_computation(obs_type, agent_type="food", agent_id=i)
                         for i, obs_type in enumerate(self.food_obs_type)],
                        [0 for _ in range(self.max_food_num)],
                        [self.remaining_timesteps == 0 for _ in range(self.max_food_num)],
                        {})
        self.food_obses = food_returns[0]

        # 7) done / masks / avail / share_obs
        success = (not any(self.food_alive_statuses))  # 所有 prey 当前都被捕获/冻结
        (
            base_team_reward,
            terminal_win_reward,
            team_reward_scalar,
            first_success_now,
        ) = self._team_reward_components(individual_rewards, success)
        self.episode_success = self.episode_success or success
        time_limit = (self.remaining_timesteps <= 0)
        # 动态恢复实验不能在 prey 暂时全部冻结时提前截断，否则后续恢复事件不会发生。
        episode_done = time_limit or (success and not self.continue_after_success)
        done = np.full((self.max_player_num, 1), episode_done, dtype=bool)

        active_masks = self._active_mask()

        avail_actions = self.get_avail_actions().astype(np.float32)
        share_obs = np.tile(self.get_state().astype(np.float32), (self.max_player_num, 1))

        # ========== team reward：把所有智能体 reward 加起来 ==========
        # 团队奖励是 transition-level 标量，必须广播到所有固定槽位。
        # Trainer 当前读取 slot 0 的共享奖励；若按 next active mask 屏蔽，slot 0
        # 恰好退出时会错误地把整个团队奖励变为 0。
        team_reward = np.full(
            (self.max_player_num, 1),
            team_reward_scalar,
            dtype=np.float32,
        )

        # --------- make infos consistent with SMAC/MPE/Prey style ---------
        # 评估/日志常用：是否成功围捕（所有 prey 均被捕获）
        won = bool(self.episode_success)

        topology_changed = bool(
            event_info["left_slots"]
            or event_info["joined_slots"]
            or event_info["recovered_slots"]
        )
        common_info = {
            "won": won,
            "valid_indices": self.valid_indices.copy(),
            "slot_uids": self.slot_uids.copy(),
            "num_players": int(self.num_players),
            "active_masks": active_masks.copy(),
            "individual_rewards": individual_rewards.copy(),
            "base_team_reward": base_team_reward,
            "terminal_win_reward": terminal_win_reward,
            "team_reward": team_reward_scalar,
            "capture_count": int(self.last_capture_count),
            "capture_events": [
                {
                    "event_id": int(event["event_id"]),
                    "target_id": int(event["target_id"]),
                    "participant_slots": list(event["participant_slots"]),
                    "capture_position": list(event["capture_position"]),
                }
                for event in self.last_capture_events
            ],
            "success_now": bool(success),
            "first_success_now": first_success_now,
            "topology_changed": topology_changed,
            "left_slots": list(event_info["left_slots"]),
            "joined_slots": list(event_info["joined_slots"]),
            "recovered_slots": list(event_info["recovered_slots"]),
            "left_count": len(event_info["left_slots"]),
            "joined_count": len(event_info["joined_slots"]),
            "recovered_count": len(event_info["recovered_slots"]),
            "pending_recovery_count": len(self.pending_recovery),
            "episode_step": int(self.episode_step),
            # Trajectory-neutral task-state provenance for post-capture
            # diagnostics.  Keep fixed prey-slot order and fixed player-slot
            # order so the runner can measure whether the behaviour switches
            # to the still-unfrozen prey during the 24-step window without an
            # extra environment or policy forward pass.
            "food_alive_statuses": [
                bool(value) for value in self.food_alive_statuses
            ],
            "food_freeze_remaining": [
                float(self._normalized_food_freeze_remaining(fid))
                for fid in range(self.max_food_num)
            ],
            "food_positions": [
                [int(position[0]), int(position[1])]
                for position in self.food_positions
            ],
            "player_slot_positions": [
                (
                    [
                        int(self.player_positions[self.valid_indices[slot]][0]),
                        int(self.player_positions[self.valid_indices[slot]][1]),
                    ]
                    if int(self.valid_indices[slot]) >= 0 else None
                )
                for slot in range(self.max_player_num)
            ],
            "food_visible_player_slots": [
                [
                    int(slot)
                    for slot in range(self.max_player_num)
                    if (
                        bool(self.food_alive_statuses[fid])
                        and int(self.valid_indices[slot]) >= 0
                        and (
                            abs(
                                int(self.player_positions[
                                    self.valid_indices[slot]
                                ][0])
                                - int(self.food_positions[fid][0])
                            )
                            + abs(
                                int(self.player_positions[
                                    self.valid_indices[slot]
                                ][1])
                                - int(self.food_positions[fid][1])
                            )
                        ) <= int(self.sight_radius)
                    )
                ]
                for fid in range(self.max_food_num)
            ],
        }
        infos = [dict(common_info) for _ in range(self.max_player_num)]

        self._validate_dynamic_outputs(obs, share_obs, active_masks, avail_actions)
        return obs, share_obs, team_reward, done, infos, avail_actions

    def _onehot4(self, ori: int):
        """将朝向(0/1/2/3)转为 one-hot(4)。"""
        try:
            o = int(ori)
        except Exception:
            o = 0
        return [1.0 if i == o else 0.0 for i in range(4)]

    def _decode_one_hot_action(self, a):
        """把 one-hot / logits / scalar 解码成离散动作 id。

        约定：全 0 时视为 stay(4)，避免空槽位引入随机动作。
        """
        if isinstance(a, (int, np.integer)):
            return int(a)
        arr = np.asarray(a).reshape(-1)
        if arr.size == 0:
            return 4
        if not np.any(arr):
            return 4
        return int(np.argmax(arr))

    def _min_dist_to_alive_food(self, px: int, py: int):
        """到最近 alive prey 的 L1 距离；若没有 alive prey，则返回 0。"""
        alive = [self.food_positions[i] for i in range(self.max_food_num) if self.food_alive_statuses[i]]
        if not alive:
            return 0.0
        return float(min([abs(px - fx) + abs(py - fy) for fx, fy in alive]))

    def _current_team_coverage_cost(self):
        """Return the reward-only capture-feasible coverage potential."""
        alive_food_positions = [
            self.food_positions[fid]
            for fid in range(self.max_food_num)
            if self.food_alive_statuses[fid]
        ]
        return capture_quorum_balanced_alive_prey_coverage_cost(
            player_positions=self.player_positions,
            food_positions=alive_food_positions,
        )

    def _reset_distance_shaping_baseline(self):
        """Reset both legacy and coverage baselines without consuming RNG."""
        self.prev_dist_to_food = [None] * self.max_player_num
        for rid in range(self.num_players):
            slot = self.valid_indices.index(rid)
            px, py = self.player_positions[rid]
            self.prev_dist_to_food[slot] = self._min_dist_to_alive_food(px, py)
        self.prev_team_coverage_cost = self._current_team_coverage_cost()

    def _observation_player_vector_partial(self, agent_id: int):
        """玩家（wolf）的部分可观测 vector 观测。"""
        rid = self.valid_indices[agent_id]
        if rid < 0:
            return np.asarray([-1.0] * self.unit_obs_dim, dtype=np.float32)

        px, py = self.player_positions[rid]
        pori = self.player_orientation[rid]

        obs = []
        obs.extend([float(px), float(py)])
        obs.extend(self._onehot4(pori))

        # teammates
        others = []
        for slot in range(self.max_player_num):
            if slot == agent_id:
                continue
            orid = self.valid_indices[slot]
            if orid >= 0:
                ox, oy = self.player_positions[orid]
                dx, dy = ox - px, oy - py
                dist = abs(dx) + abs(dy)
                if dist <= self.sight_radius:
                    others.append((dist, dx, dy))
        others.sort(key=lambda x: x[0])

        for i in range(self.max_player_num - 1):
            if i < len(others):
                obs.extend([float(others[i][1]), float(others[i][2])])
            else:
                obs.extend([-1.0, -1.0])

        # prey slots
        for fid in range(self.max_food_num):
            freeze_remaining = self._normalized_food_freeze_remaining(fid)
            if self.food_alive_statuses[fid]:
                fx, fy = self.food_positions[fid]
                dx, dy = fx - px, fy - py
                dist = abs(dx) + abs(dy)
                if dist <= self.sight_radius:
                    obs.extend([float(dx), float(dy)])
                    obs.extend(self._onehot4(self.food_orientation[fid]))
                else:
                    obs.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
            else:
                obs.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
            obs.append(freeze_remaining)

        # obstacles N,E,S,W
        nbrs = [(px - 1, py), (px, py + 1), (px + 1, py), (px, py - 1)]
        for nx, ny in nbrs:
            obs.append(0.0 if (nx, ny) in self._possibleCoordinates_set else 1.0)

        obs.append(1.0)

        if len(obs) != self.unit_obs_dim:
            raise RuntimeError(f"unit_obs_dim mismatch: got {len(obs)} expected {self.unit_obs_dim}")
        return np.asarray(obs, dtype=np.float32)

    def get_state(self):
        """全局状态向量（固定长度 state_dim）。"""
        s = []
        for slot in range(self.max_player_num):
            rid = self.valid_indices[slot]
            if rid >= 0:
                x, y = self.player_positions[rid]
                ori = self.player_orientation[rid]
                s.extend([float(x), float(y)])
                s.extend(self._onehot4(ori))
                s.append(1.0)
            else:
                s.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for fid in range(self.max_food_num):
            if self.food_alive_statuses[fid]:
                x, y = self.food_positions[fid]
                ori = self.food_orientation[fid]
                s.extend([float(x), float(y)])
                s.extend(self._onehot4(ori))
                s.append(1.0)
            else:
                s.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            s.append(self._normalized_food_freeze_remaining(fid))

        s.append(float(self.remaining_timesteps) / float(self.max_time_steps))

        out = np.asarray(s, dtype=np.float32)
        if out.size != self.state_dim:
            raise RuntimeError(f"state_dim mismatch: got {out.size} expected {self.state_dim}")
        return out

    def _normalized_food_freeze_remaining(self, food_id: int):
        """Return the task-relevant freeze window on a stable [0, 1] scale."""
        if self.food_alive_statuses[food_id] or self.food_freeze_rate <= 0:
            return 0.0
        remaining = float(self.food_frozen_time[food_id])
        return float(np.clip(remaining / float(self.food_freeze_rate), 0.0, 1.0))

    def get_avail_actions(self):
        """返回 (N, n_actions) 的可用动作掩码。
        规则：
          - 空槽位（valid_indices=-1）：只允许 stay(4)
          - 有效槽位：对移动动作(0~3)做“撞墙/障碍/越界”无效化；stay(4)/转向(5/6)默认有效
        """
        avail = np.ones((self.max_player_num, self.n_actions), dtype=np.float32)

        for slot in range(self.max_player_num):
            rid = self.valid_indices[slot]

            # 空槽位：仅允许 stay(4)
            if rid < 0:
                avail[slot, :] = 0.0
                if 0 <= 4 < self.n_actions:
                    avail[slot, 4] = 1.0
                continue

            # 有效槽位：根据当前位置判断移动是否撞墙/越界/障碍
            x, y = self.player_positions[rid]

            # 绝对方向移动：0上 1右 2下 3左
            candidates = {
                0: (x - 1, y),
                1: (x, y + 1),
                2: (x + 1, y),
                3: (x, y - 1),
            }

            for a_id, (nx, ny) in candidates.items():
                if a_id >= self.n_actions:
                    continue
                if (nx, ny) not in self._possibleCoordinates_set:
                    avail[slot, a_id] = 0.0

            # stay(4) / turn(5/6) 默认保持可用（无需处理）

        return avail

    def get_env_info(self):
        """给 runner/算法侧的环境描述信息。"""
        return {
            "n_agents": int(self.max_player_num),
            "n_actions": int(self.n_actions),
            "state_shape": int(self.state_dim),
            "obs_shape": int(self.unit_obs_dim),
            "episode_limit": int(self.max_time_steps),
        }

    def render(self, mode='human', close=False):
        if self.visualizer is None:
            self.visualizer = Visualizer(self.grid, self.grid_height, self.grid_width)

        self.visualizer.grid = self.grid
        self.visualizer.render()

    def _active_mask(self):
        """返回固定槽位掩码：1=当前在场，0=退出/尚未加入。"""
        return np.asarray(
            [[1.0 if rid >= 0 else 0.0] for rid in self.valid_indices],
            dtype=np.float32,
        )

    def _remove_agents_for_shock(self, requested_num):
        """在单个 episode 内一次性移除多个成员，并保存恢复快照。

        快照保存原 slot、逻辑 UID、位置、朝向和观测类型。真实玩家 rid
        会在删除后重新编号，因此不能把 rid 用作跨时间身份。
        """
        max_removable = max(0, self.num_players - self.dynamic_min_agents)
        remove_num = min(int(requested_num), max_removable)
        if remove_num <= 0:
            return []

        active_slots = [
            slot for slot, rid in enumerate(self.valid_indices) if rid >= 0
        ]
        selected_slots = [
            int(slot) for slot in self.prng.choice(
                active_slots, remove_num, replace=False
            ).tolist()
        ]

        recovery_records = []
        real_ids = []
        for slot in selected_slots:
            rid = self.valid_indices[slot]
            recovery_records.append({
                "slot": int(slot),
                "uid": int(self.slot_uids[slot]),
                "position": tuple(self.player_positions[rid]),
                "orientation": int(self.player_orientation[rid]),
                "obs_type": self.player_obs_type[rid],
                "due_step": int(self.episode_step + self.shock_recover_delay),
            })
            real_ids.append(rid)

        self.pending_recovery.extend(recovery_records)
        # 必须倒序删除，避免 list.pop() 改变后续 rid。
        self.del_agent(sorted(real_ids, reverse=True))
        return selected_slots

    def _restore_one_member(self, record):
        """恢复一个离线成员；优先恢复原位置，否则选择当前空地。"""
        slot = int(record["slot"])
        if slot < 0 or slot >= self.max_player_num:
            return False
        if self.valid_indices[slot] != -1:
            # 恢复槽位被占用时不丢弃记录，下一个环境步继续重试。
            return False

        alive_food_positions = {
            pos for i, pos in enumerate(self.food_positions)
            if self.food_alive_statuses[i]
        }
        free_positions = sorted(
            set(self.possibleCoordinates)
            - set(self.player_positions)
            - alive_food_positions
        )
        if not free_positions:
            return False

        preferred_position = tuple(record["position"])
        if preferred_position in free_positions:
            position = preferred_position
        else:
            position = free_positions[int(self.prng.choice(len(free_positions)))]

        rid = len(self.player_positions)
        self.player_positions.append(position)
        self.player_orientation.append(int(record["orientation"]))
        self.player_obs_type.append(record["obs_type"])
        self.valid_indices[slot] = rid
        self.slot_uids[slot] = int(record["uid"])
        self.player_points[slot] = 0.0
        self.num_players += 1

        self.grid[position[0]][position[1]] = 2
        if "full_rgb" in self.obs_type:
            self.RGB_grid[position[0]][position[1]] = [255, 255, 255]
        self.RGB_padded_grid[
            position[0] + self.pads
        ][position[1] + self.pads] = [255, 255, 255]
        return True

    def _restore_due_members(self):
        """恢复到期成员；无法恢复的记录保留到下一步重试。"""
        recovered_slots = []
        still_pending = []

        for record in self.pending_recovery:
            if int(record["due_step"]) > self.episode_step:
                still_pending.append(record)
                continue

            if self._restore_one_member(record):
                recovered_slots.append(int(record["slot"]))
            else:
                still_pending.append(record)

        self.pending_recovery = still_pending
        return recovered_slots

    def _apply_scripted_events(self):
        """应用本步的退出、恢复、新加入事件。

        恢复优先于新加入；尚未到期的恢复槽位会被预留，避免新成员抢占。
        """
        request = self.event_scheduler.events_at(self.episode_step)
        left_slots = self._remove_agents_for_shock(request["remove_num"])
        recovered_slots = self._restore_due_members()

        reserved_slots = {
            int(record["slot"]) for record in self.pending_recovery
        }
        allowed_join_slots = [
            slot for slot, rid in enumerate(self.valid_indices)
            if rid == -1 and slot not in reserved_slots
        ]
        joined_slots = self.add_agent(
            ["vector"] * int(request["join_num"]),
            allowed_slots=allowed_join_slots,
        )

        return {
            "left_slots": left_slots,
            "joined_slots": joined_slots,
            "recovered_slots": recovered_slots,
        }

    def _validate_dynamic_outputs(self, obs, share_obs, active_masks, avail_actions):
        """动态模式下检查固定容量和空槽位编码，尽早暴露张量错位。"""
        if not self.intra_episode_dynamic:
            return

        if obs.shape[0] != self.max_player_num:
            raise RuntimeError(f"obs slot mismatch: {obs.shape}")
        if share_obs.shape[0] != self.max_player_num:
            raise RuntimeError(f"share_obs slot mismatch: {share_obs.shape}")
        if active_masks.shape != (self.max_player_num, 1):
            raise RuntimeError(f"active mask mismatch: {active_masks.shape}")
        if avail_actions.shape[0] != self.max_player_num:
            raise RuntimeError(f"avail_actions slot mismatch: {avail_actions.shape}")

        inactive = active_masks[:, 0] == 0
        if np.any(inactive):
            if not np.all(obs[inactive] == -1.0):
                raise RuntimeError("inactive slot observation must be all -1")
            if not np.all(avail_actions[inactive, 4] == 1.0):
                raise RuntimeError("inactive slot must allow stay action")
            if not np.all(np.sum(avail_actions[inactive], axis=-1) == 1.0):
                raise RuntimeError("inactive slot must only allow stay action")

    """方法作用：增加若干玩家（OpenScheduler 触发后调用）。
        步骤：
            1) 随机从空地挑选不冲突的坐标；
            2) 更新位置/朝向/网格显示；
            3) 在 valid_indices 里为新来的玩家分配空槽位（值=当前最大真实索引+1）。
        新成员不能占用等待原成员恢复的槽位；恢复成员使用原 slot 和原 UID。
    """
    def add_agent(self, new_types, allowed_slots=None):
        if not new_types:
            return []

        empty_slots = [
            slot for slot, rid in enumerate(self.valid_indices)
            if rid == -1
        ]

        if allowed_slots is not None:
            allowed = set(allowed_slots)
            empty_slots = [slot for slot in empty_slots if slot in allowed]

        alive_food_positions = [
            pos for i, pos in enumerate(self.food_positions)
            if self.food_alive_statuses[i]
        ]
        available_pos = sorted(
            set(self.possibleCoordinates)
            - set(self.player_positions)
            - set(alive_food_positions)
        )

        add_num = min(len(new_types), len(empty_slots), len(available_pos))
        if add_num <= 0:
            return []

        self.prng.shuffle(empty_slots)
        pos_indices = self.prng.choice(
            len(available_pos), add_num, replace=False
        ).tolist()

        added_slots = []

        for i in range(add_num):
            slot = empty_slots[i]
            pos = available_pos[pos_indices[i]]
            rid = len(self.player_positions)

            self.player_positions.append(pos)
            self.player_orientation.append(0)
            self.player_obs_type.append(new_types[i])
            self.valid_indices[slot] = rid
            self.num_players += 1

            self.slot_uids[slot] = self.next_agent_uid
            self.next_agent_uid += 1
            self.player_points[slot] = 0.0

            self.grid[pos[0]][pos[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[pos[0]][pos[1]] = [255, 255, 255]
            self.RGB_padded_grid[
                pos[0] + self.pads
                ][pos[1] + self.pads] = [255, 255, 255]

            added_slots.append(slot)

        return added_slots

    def del_agent(self, agent_id):
        """删除若干玩家（OpenScheduler 触发后调用）。

        多索引删除必须从大到小删除，避免 pop 导致索引偏移。
        """
        if agent_id is None:
            return
        if isinstance(agent_id, int):
            agent_id = [agent_id]

        deleted_slots = []
        for idx in sorted(agent_id, reverse=True):
            if idx < 0 or idx >= len(self.player_positions):
                continue

            # 删除前记录 rid 对应的固定 slot；rid 随 list.pop() 会重新编号。
            slot = self.valid_indices.index(idx)
            a = self.player_positions[idx]
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]

            self.player_positions.pop(idx)
            self.player_orientation.pop(idx)
            if idx < len(self.player_obs_type):
                self.player_obs_type.pop(idx)
            self.slot_uids[slot] = -1
            self.num_players -= 1
            deleted_slots.append(slot)

            # 更新槽位映射：>idx 的真实 id 需要 -1；=idx 的槽位置空
            for slot_i, v in enumerate(self.valid_indices):
                if v > idx:
                    self.valid_indices[slot_i] -= 1
                elif v == idx:
                    self.valid_indices[slot_i] = -1

        return deleted_slots

"""开放调度器：以几何分布随机决定“删除/添加”玩家的数量与时机。
    关键参数：
      - add_rate / remove_rate：每步触发添加/删除的概率；
      - max_available_agents：可容纳的最大玩家数（不会超过这个上限）；
      - min_alive_time：玩家加入后至少存活多少步才有机会被删除（防止抖动）。
    接口：
      - open_process()：返回 (deleted_idxs, new_obs_type_list)，供环境执行增删。
"""
class OpenScheduler(object):

    def __init__(self, num_agents, add_rate, remove_rate, max_available_agents, min_available_agents=2, min_alive_time=25, seed=0):
        # cap 修正：max_available_agents 不能小于初始 num_agents
        if max_available_agents is None:
            max_available_agents = num_agents
        max_available_agents = int(max(max_available_agents, num_agents))

        self.available_agents = int(num_agents)
        self.max_available_agents = int(max_available_agents)
        self.min_available_agents = int(min_available_agents)

        self.alive_time = [0] * self.available_agents
        self.min_alive_time = min_alive_time
        self.seed = seed
        self.prng = RandomState(self.seed)
        self.timestep = 0
        # Use geometric distribution to sample remove or del
        self.geometric_add_rate = add_rate
        self.geometric_remove_rate = remove_rate
    def add_agents(self, agent_nums):
        new_obs_type = []
        for _ in range(agent_nums):
            new_obs_type.append("vector")#登记新玩家的观测类型（可扩展为不同类型）。
            self.available_agents += 1
            self.alive_time.append(0) #为新玩家建立一条“存活时长计数器”，初始为 0
        return new_obs_type

    """删除指定下标的玩家（从 alive_time 中移除并递减 available_agents）。"""
    def del_agents(self, agent_idxs):
        agent_idxs_sorted = agent_idxs.copy()
        agent_idxs_sorted.sort(reverse=True)
        for idxes in agent_idxs_sorted:
            del self.alive_time[idxes]
        self.available_agents -= len(agent_idxs_sorted)
        return agent_idxs_sorted # list[int]），要被移除的相对索引

    """采样要删除的玩家索引：
          仅选择“存活时间 > min_alive_time”的玩家，删除数量 ∈ {0,1,2}（受概率与可删人数约束）。
    """
    def agent_removal_sampler(self):
        # 只有在线时间 > min_alive_time 的玩家才有资格被“抽走”
        eligible_idxs = [idx for idx, alive_dur in enumerate(self.alive_time) if alive_dur > self.min_alive_time]
        max_removable = max(0, self.available_agents - self.min_available_agents)
        removed_amount = min(1, max_removable, len(eligible_idxs))

        removed_indices = []
        if removed_amount != 0:
            # 在 eligible_idxs 里无放回随机挑 removed_amount 个索引并返回。
            removed_indices = self.prng.choice(len(eligible_idxs), removed_amount, replace=False).tolist()
            removed_indices = [eligible_idxs[k] for k in removed_indices]
        return removed_indices

    """方法作用：推进一个调度步，可能删除/添加玩家。
        步骤：
          1) alive_time 全 +1；
          2) 以 remove_rate 决定是否删除；若删除 → agent_removal_sampler 采样并 del_agents；
          3) 以 add_rate 决定是否添加；若添加且未达到上限，则添加 1 名玩家；
          4) 返回 (deleted_idxs, new_obs_type_list)。
    """
    def open_process(self):
        """推进一个调度步，可能删除/添加玩家。
        修正点：
          - 若 cap <= 当前在线人数，则本步禁止 add（避免负上限导致异常）。
        """
        self.alive_time = [x + 1 for x in self.alive_time]
        self.timestep += 1

        remove = (self.prng.uniform() < self.geometric_remove_rate)
        add = (self.prng.uniform() < self.geometric_add_rate)

        deleted_idxs = []
        if remove:
            removed_indices = self.agent_removal_sampler()
            if removed_indices:
                deleted_idxs = self.del_agents(removed_indices)
                # 本步若发生删除，则禁止再添加（删除优先）
                return deleted_idxs, []

        # 走到这里说明本步没有发生删除（要么 remove=False，要么没选出可删的）
        new_obs_type = []
        space = max(0, self.max_available_agents - self.available_agents)
        if add and space > 0:
            agent_nums = min(1, space)
            if agent_nums > 0:
                new_obs_type = self.add_agents(agent_nums)

        return deleted_idxs, new_obs_type

"""保留原 OpenScheduler，新增可复现的多人突发事件。"""
class IntraEpisodeEventScheduler:
    """只生成事件请求，不直接修改环境内部玩家数组。"""

    def __init__(
        self,
        shock_steps,
        remove_num,
        join_delay,
        join_num,
    ):
        if isinstance(shock_steps, str):
            shock_steps = [
                int(x.strip())
                for x in shock_steps.split(",")
                if x.strip()
            ]

        self.shock_steps = sorted(set(shock_steps or []))
        self.remove_num = int(remove_num)
        self.join_delay = int(join_delay)
        self.join_num = int(join_num)

        self.join_events = {}
        for step in self.shock_steps:
            join_step = step + self.join_delay
            self.join_events[join_step] = (
                self.join_events.get(join_step, 0) + self.join_num
            )

    def events_at(self, episode_step):
        return {
            "remove_num": (
                self.remove_num
                if episode_step in self.shock_steps
                else 0
            ),
            "join_num": self.join_events.get(episode_step, 0),
        }

"""基于 pygame 的简易网格渲染器：蓝色背景、黑色空地、白色玩家、红色食物。"""
class Visualizer(object):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    WIDTH = 20
    HEIGHT = 20

    MARGIN = 0

    # Create a 2 dimensional array. A two dimensional
    # array is simply a list of lists.

    def __init__(self, grid, grid_height=20, grid_width=20):
        import pygame
        self.pygame = pygame
        self.grid_height, self.grid_width = grid_height, grid_width
        self.grid = grid
        self.WINDOW_SIZE = [self.grid_height * self.HEIGHT, self.grid_width * self.WIDTH]

        self.pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        self.pygame.display.set_caption("Wolfpack")
        self.clock = pygame.time.Clock()

    """渲染一帧网格；若关闭窗口则退出 pygame。"""
    def render(self):
        done = False
        for event in self.pygame.event.get():  # User did something
            if event.type == self.pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)

        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLUE
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                elif self.grid[row][column] == 3:
                    color = self.RED
                self.pygame.draw.rect(self.screen,
                                      color,
                                      [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                       (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                       self.WIDTH,
                                       self.HEIGHT])

        self.clock.tick(60)
        self.pygame.display.flip()

        if done:
            self.pygame.quit()
