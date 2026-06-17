# import copy
# import random
# import pickle as pkl
# import numpy as np
# import threading
# import gym
# from gym import spaces
# import time
# import sys, os
# from .assets.Agent import DQNAgent
# from numpy.random import RandomState
#
#
# """
# Generator：用cellular automata algorithm生成随机障碍地图（可通关检查联通性）。
# - 关键参数：
#   - deathLimit / birthLimit 控制“细胞死亡/出生”的阈值，影响障碍密度。
#   - probStartAlive 初始为 True 的概率（True 代表障碍）。
# - 核心流程：
#   1) initialiseMap：按概率初始化布尔网格。
#   2) doSimulationStep：根据邻居数量做一次迭代（类似生命游戏规则）。
#   3) simulate：反复生成，直到 flood fill 判断“可通达区域连通”。
# """
# class Generator(object):
#     def __init__(self, size, deathLimit=4, birthLimit=3, seed=0):
#         self.x_size = size[0]
#         self.y_size = size[1]
#         # booleanMap[y][x]：True 表示“障碍/填充”，False 表示“可走”
#         self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
#         self.probStartAlive = 0.82  # 初始为 True 的概率（越大障碍越多）
#         self.deathLimit = deathLimit
#         self.birthLimit = birthLimit
#         self.seed = seed
#         self.prng = RandomState(seed)
#         self.copy = None
#
#     """方法作用：按概率随机初始化地图。
#         步骤：
#           1) 遍历每个格点；
#           2) 以 probStartAlive 的概率置为 True（障碍）。
#     """
#     def initialiseMap(self):
#         for x in range(self.x_size):
#             for y in range(self.y_size):
#                 if self.prng.random.uniform() < self.probStartAlive:
#                     self.booleanMap[y][x] = True
#
#     """方法作用：对整张图应用一次元胞自动机规则，更新障碍/空白。
#         步骤：
#           1) 计算每个格点的“活邻居数”（True 的邻居数量）；
#           2) 若当前为 True(障碍)：邻居数 < deathLimit 则变 False，否则保持 True；
#           3) 若当前为 False(空)：邻居数 > birthLimit 则变 True，否则保持 False。
#      """
#     def doSimulationStep(self):
#         newMap = [[False] * k for k in [self.x_size] * self.y_size]
#         for x in range(self.x_size):
#             for y in range(self.y_size):
#                 alive = self.countAliveNeighbours(x, y)
#                 if self.booleanMap[y][x]:
#                     if alive < self.deathLimit:
#                         newMap[y][x] = False
#                     else:
#                         newMap[y][x] = True
#                 else:
#                     if alive > self.birthLimit:
#                         newMap[y][x] = True
#                     else:
#                         newMap[y][x] = False
#         self.booleanMap = newMap
#
#     """方法作用：统计 (x,y) 周围 8 邻域中为 True 的数量；边界外视为 True（促使边缘更“封”）。
#         步骤：
#           1) 遍历偏移 (-1..1, -1..1) 排除自身；
#           2) 越界计为 1；未越界则读 booleanMap 计数。
#     """
#     def countAliveNeighbours(self, x, y):
#         count = 0
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 neighbour_x = x + i
#                 neighbour_y = y + j
#                 if not ((i == 0) and (j == 0)):
#                     if neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= self.x_size or neighbour_y >= self.y_size:
#                         count = count + 1
#                     elif self.booleanMap[neighbour_y][neighbour_x]:
#                         count = count + 1
#         return count
#
#     """方法作用：反复随机生成并迭代，直至通过 flood fill 验证“空白区连通”。
#         步骤：
#           1) 重置并随机初始化；
#           2) 进行 numSteps 次 doSimulationStep；
#           3) 用 doFloodfill 检查是否所有 False 区域连成一片（可行走区域连通）；
#           4) 若不连通则重新来过。
#     """
#     def simulate(self, numSteps):
#         done = False
#         while not done:
#             self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
#             self.initialiseMap()
#             for kk in range(numSteps):
#                 self.doSimulationStep()
#
#             if self.doFloodfill(self.booleanMap):
#                 done = True
#
#     """方法作用：从第一个空白点开始 flood fill，若能覆盖所有空白点则连通。
#          步骤：
#               1) 深拷贝地图到 self.copy；
#               2) 找到第一个 False 的起点；
#               3) floodfill 标记所有连通的 False；
#               4) 若仍有 False 未被覆盖 => 不连通，返回 False；否则 True。
#     """
#     def doFloodfill(self, newMap):
#         self.copy = copy.deepcopy(newMap)
#         foundX, foundY = -1, -1
#         for i in range(len(self.copy)):
#             flag = False
#             for j in range(len(self.copy[i])):
#                 if not self.copy[i][j]:
#                     foundX = i
#                     foundY = j
#                     flag = True
#                     break
#             if flag:
#                 break
#         self.floodfill(foundX, foundY)
#         done = True
#         for i in range(len(self.copy)):
#             flag = False
#             for j in range(len(self.copy[i])):
#                 if not self.copy[i][j]:
#                     done = False
#                     flag = True
#                     break
#             if flag:
#                 break
#         return done
#
#     """方法作用：队列式 flood fill，把连通的 False 全部置 True（仅在 doFloodfill 内部使用）。
#            步骤：
#              1) 从起点入队；
#              2) 反复出队，若位置为 False 就置 True；
#              3) 将四邻域（上下左右）中为 False 的继续入队。
#     """
#     def floodfill(self, x, y):
#         queue = []
#         queue.append((x, y))
#         while len(queue) != 0:
#             a = queue[0][0]
#             b = queue[0][1]
#
#             del queue[0]
#             if not self.copy[a][b]:
#                 self.copy[a][b] = True
#
#             if (not a + 1 >= len(self.copy)) and (not self.copy[a + 1][b]):
#                 queue.append((a + 1, b))
#                 self.copy[a + 1][b] = True
#             if (not (a - 1 < 0)) and (not self.copy[a - 1][b]):
#                 queue.append((a - 1, b))
#                 self.copy[a - 1][b] = True
#             if (not b + 1 >= len(self.copy[0])) and (not self.copy[a][b + 1]):
#                 queue.append((a, b + 1))
#                 self.copy[a][b + 1] = True
#             if (not b - 1 < 0) and (not self.copy[a][b - 1]):
#                 queue.append((a, b - 1))
#                 self.copy[a][b - 1] = True
#
# """环境作用：WolfPack（狼群围捕）开放队伍规模版本（回合内可加/减玩家）。
#     关键特性：
#       - 离散动作空间（共 6 个动作：前/右/后/左/停/左转/右转，其中实现把左转记为5，右转为其他分支）。
#       - 观测支持 vector / partial_obs / full_rgb（此处主要使用 vector）。
#       - OpenScheduler 决定每个时间步是否“增减”玩家，实现回合内 N 变化。
#       - 食物(prey)由预训练的 DQNAgent 控制，可被“协作围捕”后冻结并在冷却后复活,因此回合在最大时间步时终止。
#       valid_indices 是“当前在场玩家的真实索引列表”（例如 [0,2,3] 表示 0、2、3 号玩家此刻在场；1 号玩家此刻缺席）。
#       采用固定长度的观测槽位（max_player_num），一个内部索引表来把“真实玩家集合”与这些槽位对应起来
#       - 奖励：
#           * 靠近食物有微小正向 shaping（上一步距离-本步距离）*0.01；
#           * 单狼贴近食物但未形成围捕 => close_penalty 惩罚,鼓励协作而不是抢占位置；
#           * 多狼共同在 coopRadius 范围内夹击,该只食物记 -1 并冻结一段时间后再复活 => groupMultiplier * 参与人数 的正奖励，食物 -1 并冻结。
# """
# class WolfpackPenaltyOpen(gym.Env):
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self, grid_height=20, grid_width=20, num_players=5, max_food_num=2,
#                  sight_sideways=8, sight_radius=8, max_time_steps=200,
#                  coop_radius=1, groupMultiplier=2, food_freeze_rate=0,
#                  add_rate=0.05, del_rate=0.05, seed=None,
#                  max_player_num=5, implicit_max_player_num=3,
#                  obs_type="vector", with_random_grid=False, random_grid_dir=None,
#                  prey_with_gpu=False, close_penalty=0.5):
#
#         # ====== 基本网格/观测与动作配置 ======
#         self.grid_height = grid_height
#         self.grid_width = grid_width
#         self.obs_type = obs_type
#         self.close_penalty = close_penalty
#         self.num_players = num_players                     # 当前在场玩家数
#         self.max_player_num = max_player_num               # 最大可容纳玩家数（固定观测维度用）
#         self.implicit_max_player_num = implicit_max_player_num  # 初始“可用”玩家容量（scheduler 用）
#         self.ma_obs_type = obs_type
#         self.N_DISCRETE_ACTIONS = 6
#         # 为了支持变 N，这里给每个 potential slot 一个 Discrete(6) 的动作空间（外部会只取有效索引）
#         self.action_space = [gym.spaces.Discrete(6) for _ in range(max_player_num)]
#         self.other_player_acts = None
#
#         # ====== 观测空间（vector 版本）======
#         if obs_type == "vector":
#             # 单个 agent 的观测向量构成：
#             # [自己(x,y), 其他玩家(x,y)* (max_player_num-1), 食物位置*(2*max_food_num), 食物朝向one-hot*(4*max_food_num), exist_flag]
#             box_high = []
#             for idx in range(self.max_player_num):
#                 box_high.append(self.grid_height - 1)  # x 上界
#                 box_high.append(self.grid_width - 1)   # y 上界
#             for idx in range(6 * max_food_num):
#                 if idx % 6 == 0:
#                     box_high.append(self.grid_height - 1)
#                 elif idx % 6 == 1:
#                     box_high.append(self.grid_width - 1)
#                 else:
#                     box_high.append(1)                 # 朝向 one-hot 的上界 or 标志位
#             box_high.append(1)                          # 最尾的 exist_flag（1 or -1）
#
#             # 下界全部置 0，exist_flag 的下界置 -1（无效 agent 行会全 -1）
#             box_low = [0] * len(box_high)
#             box_low[-1] = -1
#
#             # 为所有 potential slots 复制相同的 Box 定义
#             final_box_high = [box_high.copy() for _ in range(self.max_player_num)]
#             final_box_low = [box_low.copy() for _ in range(self.max_player_num)]
#
#             self.observation_space = [
#                 spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float64)
#                 for low, high in zip(final_box_low, final_box_high)
#             ]
#
#         # ====== 随机/调度器/可视化状态 ======
#         self.add_rate = add_rate  # 每步“增加玩家”的几何分布参数
#         self.del_rate = del_rate  # 每步“删除玩家”的几何分布参数
#         self.masking = []  # （预留）掩码
#         if seed is None:
#             seed = int(time.time())
#         self.seed = seed
#         self.randomizer = random
#         self.prng = RandomState(self.seed)
#         # OpenScheduler：回合内增减玩家的“到达-离开”过程
#         self.scheduler = OpenScheduler(self.num_players, self.add_rate, self.del_rate,
#                                        self.implicit_max_player_num, seed=self.seed)
#
#         # 视野与 padded RGB 网格（为 partial_obs / full_rgb 服务）
#         self.sight_sideways = sight_sideways
#         self.sight_radius = sight_radius
#         self.pads = max(self.sight_sideways, self.sight_radius)
#
#         # Padded RGB matrix for preys that receive RGB inputs
#         self.RGB_padded_grid = [[[0, 0, 255] for b in range(2 * self.pads + self.grid_width)]
#                                 for a in range(2 * self.pads + self.grid_height)]
#
#         # Grid to save the locations of agents, preys and obstacles
#         # 主 grid：0=墙/不可走未填；1=空地；2=玩家；3=食物
#         self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
#
#         if "full_rgb" in self.obs_type:
#             # RGB grid without padding in case agents use RGB grids as input
#             self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)]
#                              for a in range(self.grid_height)]
#
#         # ====== 初始玩家/食物数量与槽位管理 ======
#         self.init_num_players = num_players
#         # valid_indices：长度=max_player_num；值=“该槽位对应的真实 players 列表下标”，-1 代表空槽
#         self.valid_indices = [-1] * self.max_player_num
#         for idx in range(self.init_num_players):
#             self.valid_indices[idx] = idx
#         self.prev_valid_indices = None
#         self.max_food_num = max_food_num
#
#         # ====== 回合长度、食物冷却、缓存 ======
#         self.max_time_steps = max_time_steps
#         self.food_freeze_rate = food_freeze_rate
#         self.other_player_obses = None
#         self.all_player_acts = None
#         self.other_player_acts = None
#
#         # ====== 地图生成 ======
#         self.with_random_grid = with_random_grid
#         self.random_grid_dir = random_grid_dir
#
#         if not self.random_grid_dir is None:
#             self.levelMap = self.load_map(self.random_grid_dir)
#         elif not self.with_random_grid: # 默认全可走
#             self.levelMap = [[False] * k for k in [self.grid_width] * self.grid_height]
#         else: # 自动机生成随机地图并连通性检查
#             app = Generator((self.grid_width, self.grid_height), 7, 8)
#             app.initialiseMap()
#             app.simulate(2)
#             self.levelMap = app.booleanMap
#
#         # ====== 一堆运行时状态缓存 ======
#         self.visualizer = None
#         self.obstacleCoord = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if i]
#         self.possibleCoordinates = None
#         self.player_positions = None
#         self.player_orientation = None
#         self.food_positions = None
#         self.food_alive_statuses = None
#         self.food_frozen_time = None
#         self.food_orientation = None
#         self.player_points = None
#         self.food_points = None
#         self.a_to_idx = None
#         self.idx_to_a = None
#         self.prev_dist_to_food = None
#
#         self.player_obs_type = [obs_type for _ in range(self.num_players)]
#         self.food_obs_type = ["partial_obs" for _ in range(self.max_food_num)]
#         self.food_obses = None
#         self.prey_with_gpu = prey_with_gpu
#
#
#         # ====== 食物（猎物）智能体加载（预训练 DQN） ======
#         self.remaining_timesteps = max_time_steps
#         self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
#                                    obs_type="partial_obs")
#                           for a in range(self.max_food_num)]
#
#         dirname = os.path.dirname(__file__)
#         for idx, agent in enumerate(self.food_list):
#             filename = os.path.join(dirname,
#                                     ("assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
#             agent.load_parameters(filename)
#
#         self.sight_sideways = sight_sideways
#         self.sight_radius = sight_radius
#         self.coopRadius = coop_radius              # 围捕时“近邻”半径（曼哈顿距离）
#         self.groupMultiplier = groupMultiplier     # 协作奖励系数
#
#     """保存随机生成的 levelMap。"""
#     def save_map(self, filename):
#         with open(filename, 'wb') as f:
#             pkl.dump(self.levelMap, f)
#
#     def load_map(self, filename):
#         with open(filename, 'rb') as f:
#             self.levelMap = pkl.load(f)
#
#     """方法作用：重置一轮游戏并返回所有潜在槽位的观测（长度=max_player_num）。
#            步骤：
#              1) 重置随机种子、调度器、网格与 RGB 缓存；
#              2) 生成可走坐标列表 possibleCoordinates；
#              3) 随机采样玩家、食物初始位置与朝向；
#              4) 初始化奖励、冷却、计时器等状态；
#              5) 构建每个槽位的 vector 观测（不存在的槽位会输出全 -1，末尾 exist_flag=-1）。
#     """
#     def reset(self):
#         # 可视化/随机种子/调度器重置
#         self.visualizer = None
#         self.seed += 1250
#         self.prng = RandomState(self.seed)
#
#         self.num_players = self.init_num_players
#         self.valid_indices = [-1] * self.max_player_num
#         for idx in range(self.init_num_players):
#             self.valid_indices[idx] = idx
#         self.prev_valid_indices = None
#
#         self.scheduler = OpenScheduler(self.num_players, self.add_rate, self.del_rate,
#                                        self.implicit_max_player_num, seed=self.seed)
#
#         # 重置网格、RGB
#         self.RGB_padded_grid = [[[0, 0, 255] for _ in range(2 * self.pads + self.grid_width)]
#                                 for a in range(2 * self.pads + self.grid_height)]
#
#         # Reset grid locations
#         self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
#
#         if "full_rgb" in self.obs_type:
#             # Reset RGB Grid
#             self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]
#
#         # 可走坐标（levelMap 为 False 的位置可走）
#         self.possibleCoordinates = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if
#                                     not i]
#
#         # 采样玩家位置（不重复）
#         player_loc_idx = self.prng.choice(
#             range(len(self.possibleCoordinates)), self.num_players, replace=False
#         ).tolist()
#         # player_loc_idx = random.sample(range(len(self.possibleCoordinates)), self.num_players)
#         self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]
#
#         # Reset initial player orientation and points
#         self.player_orientation = [0 for a in range(self.num_players)]
#         self.player_points = [0 for a in range(self.max_player_num)]
#
#         # 采样食物位置（不能和玩家重叠）
#         coordinates_no_player = [a for a in self.possibleCoordinates if a not in self.player_positions]
#         food_loc_idx = self.prng.choice(range(len(coordinates_no_player)), self.max_food_num, replace=False).tolist()
#
#         # Reset all food attributes
#         self.food_obses = None
#         self.food_positions = [coordinates_no_player[a] for a in food_loc_idx]
#         self.food_alive_statuses = [True for a in range(self.max_food_num)]
#         self.food_frozen_time = [0 for a in range(self.max_food_num)]
#         self.food_points = [0 for a in range(self.max_food_num)]
#
#         self.food_orientation = [0 for a in range(self.max_food_num)]
#
#         # 将元素写回 grid / RGB
#         for coord in self.possibleCoordinates:
#             self.grid[coord[0]][coord[1]] = 1
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[coord[0]][coord[1]] = [0, 0, 0]
#             self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [0, 0, 0]
#         for coord in self.player_positions:
#             self.grid[coord[0]][coord[1]] = 2
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[coord[0]][coord[1]] = [255, 255, 255]
#             self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 255, 255]
#         for coord in self.food_positions:
#             self.grid[coord[0]][coord[1]] = 3
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[coord[0]][coord[1]] = [255, 0, 0]
#             self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 0, 0]
#
#
#         # 初始化“距离食物”的塑形基线（仅为有效槽位计算）
#         self.prev_dist_to_food = [None] * self.max_player_num
#         for id in range(max(self.valid_indices)+1):
#             px, py = self.player_positions[id][0], self.player_positions[id][1]
#             self.prev_dist_to_food[self.valid_indices.index(id)] = \
#                 min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])
#         # self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
#         #                           for (px, py) in self.player_positions]
#         self.remaining_timesteps = self.max_time_steps
#
#         # 计算初始观测
#         self.player_obs_type = [self.obs_type for _ in range(self.num_players)]
#         player_obses = [self.observation_computation(self.ma_obs_type, agent_id=id) for id in
#                         range(self.max_player_num)]
#         food_obses = [self.observation_computation(obs_type,
#                                                    agent_type="food", agent_id=id) for id, obs_type in
#                       enumerate(self.food_obs_type)]
#         self.food_obses = food_obses
#
#         # 重新实例化并加载食物策略参数（避免并行进程下的引用问题）
#         self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
#                                    obs_type="partial_obs")
#                           for a in range(self.max_food_num)]
#         dirname = os.path.dirname(__file__)
#         for idx, agent in enumerate(self.food_list):
#             filename = os.path.join(dirname,
#                                     ("assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
#             agent.load_parameters(filename)
#         return player_obses
#
#     """方法作用：让“冷却结束”的已死食物随机复活到不冲突的空地。
#         步骤：
#             1) 找出可复活的食物索引（alive_status=False 且 frozen_time<=0）；
#             2) 在不含玩家与存活食物的位置里随机采样新坐标；
#             3) 标记这些食物为存活并更新其位置；
#             4) 更新 prev_dist_to_food（用于塑形奖励基线）。
#     """
#     def revive(self):
#         # find possible locations to revive dead prey
#         coordinates_no_player = [a for a in self.possibleCoordinates if
#                                  a not in self.player_positions and a not in self.food_positions]
#         revived_idxes = []
#         for idx, food in enumerate(self.food_positions):
#             if self.food_frozen_time[idx] <= 0 and not self.food_alive_statuses[idx]:
#                 revived_idxes.append(idx)
#
#         if len(revived_idxes) > 0:
#             idxes = []
#             for k in range(len(revived_idxes)):
#                 idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
#                 # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
#                 while idx in idxes:
#                     idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
#                     # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
#                 idxes.append(idx)
#             coords = [coordinates_no_player[idx] for idx in idxes]
#
#             coord_idx = 0
#             for idx in revived_idxes:
#                 self.food_alive_statuses[idx] = True
#                 self.food_positions[idx] = coords[coord_idx]
#                 coord_idx += 1
#
#         # self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
#         #                           for (px, py) in self.player_positions]
#
#         self.prev_dist_to_food = [None] * self.max_player_num
#         for id in range(max(self.valid_indices)+1):
#             px, py = self.player_positions[id][0], self.player_positions[id][1]
#             self.prev_dist_to_food[self.valid_indices.index(id)] = \
#                 min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])
#
#     """方法作用：推进所有“已死食物”的冷却时间（每步 -1）。"""
#     def update_status(self):
#         for idx in range(len(self.food_alive_statuses)):
#             if not self.food_alive_statuses[idx]:
#                 self.food_frozen_time[idx] -= 1
#
#     """方法作用：批量计算一组实体的下一位置与朝向（包装器）。
#         步骤：
#         1) zip 三个列表（动作 / 位置 / 朝向）；
#         2) 对每个实体调用 calculate_indiv_position；
#         3) 返回 [(nx,ny,ori), ...]。
#     """
#     def calculate_new_position(self, collectiveAct, prev_player_position, prev_player_orientation):
#         zipped_data = list(zip(collectiveAct, prev_player_position, prev_player_orientation))
#         result = [self.calculate_indiv_position(a, (b, c), d) for (a, (b, c), d) in zipped_data]
#         return result
#
#     """方法作用：单个实体根据动作和朝向更新位置/朝向（带越界/碰墙检测）。
#         动作编码：
#           0: 前进；1: 右移；2: 后退；3: 左移；4: 不动；5: 左转；(else): 右转
#         步骤：
#           1) 按当前朝向把动作映射到相对位移或朝向变化；
#           2) 若目标位置在可走集合内，则更新；否则停留；
#           3) 返回 (nx, ny, new_orientation)。
#     """
#     def calculate_indiv_position(self, action, pair, orientation):
#         x = pair[0]
#         y = pair[1]
#         next_x = x
#         next_y = y
#
#         # go forward
#         if action == 0:  # 前进
#             # Facing upwards
#             if orientation == 0:
#                 next_x -= 1
#             # Facing right
#             elif orientation == 1:
#                 next_y += 1
#             # Facing downwards
#             elif orientation == 2:
#                 next_x += 1
#             else:
#                 next_y -= 1
#
#             if (next_x, next_y) in set(self.possibleCoordinates):
#                 return (next_x, next_y, orientation)
#             else:
#                 return (x, y, orientation)
#
#         # Step right
#         elif action == 1:  # 右移（相对朝向）
#             # Facing upwards
#             if orientation == 0:
#                 next_y += 1
#             # Facing right
#             elif orientation == 1:
#                 next_x += 1
#             # Facing downwards
#             elif orientation == 2:
#                 next_y -= 1
#             else:
#                 next_x -= 1
#
#             if (next_x, next_y) in set(self.possibleCoordinates):
#                 return (next_x, next_y, orientation)
#             else:
#                 return (x, y, orientation)
#
#         # Step back
#         elif action == 2:  # 后退
#             # Facing upwards
#             if orientation == 0:
#                 next_x += 1
#             # Facing right
#             elif orientation == 1:
#                 next_y -= 1
#             # Facing downwards
#             elif orientation == 2:
#                 next_x -= 1
#             else:
#                 next_y += 1
#
#             if (next_x, next_y) in set(self.possibleCoordinates):
#                 return (next_x, next_y, orientation)
#             else:
#                 return (x, y, orientation)
#
#         # Step left
#         elif action == 3:  # 左移（相对朝向）
#             # Facing upwards
#             if orientation == 0:
#                 next_y -= 1
#             # Facing right
#             elif orientation == 1:
#                 next_x -= 1
#             # Facing downwards
#             elif orientation == 2:
#                 next_y += 1
#             else:
#                 next_x += 1
#
#             if (next_x, next_y) in set(self.possibleCoordinates):
#                 return (next_x, next_y, orientation)
#             else:
#                 return (x, y, orientation)
#
#         # stay still
#         elif action == 4:  # 不动
#             return (x, y, orientation)
#
#         # rotate left
#         elif action == 5:  # 左转
#             new_orientation = 0
#             if orientation == 0:
#                 new_orientation = 3
#             elif orientation == 1:
#                 new_orientation = 0
#             elif orientation == 2:
#                 new_orientation = 1
#             else:
#                 new_orientation = 2
#
#             return (x, y, new_orientation)
#
#         # rotate right
#         else:  # 右转
#             new_orientation = 0
#             if orientation == 0:
#                 new_orientation = 1
#             elif orientation == 1:
#                 new_orientation = 2
#             elif orientation == 2:
#                 new_orientation = 3
#             else:
#                 new_orientation = 0
#
#             return (x, y, new_orientation)
#
#     """方法作用：根据玩家/食物位置关系更新奖励与食物存活状态（围捕/贴近惩罚/塑形）。
#           步骤：
#             1) 计算每名“有效玩家”的当前最近食物曼哈顿距离，与上一步比较给 0.01*(prev-cur)；
#             2) 对每名玩家的四邻（上下左右）若碰到食物：
#                - 统计 coopRadius 范围内靠近此食物的玩家数 close；
#                - 若 close>1：触发围捕 -> 玩家加 groupMultiplier*close，食物扣 1、标记死亡、开始冷却；
#                - 否则：单狼贴近 -> 玩家扣 close_penalty；
#             3) 把仍存活食物重绘回网格（RGB 同步）。
#           """
#     def update_food_status(self):
#         self.food_points = [0 for a in range(self.max_food_num)]
#
#         enumFood = list(enumerate(self.food_positions))
#         food_locations = [(food[0], food[1]) for idx, food in enumFood if self.food_alive_statuses[idx]]
#         food_id = [idx for idx, food in enumFood if self.food_alive_statuses[idx]]
#
#         player_locations = self.player_positions
#         set_of_food_location = set(food_locations)
#
#         # 当前距离（仅对有效槽位）
#         cur_dist_to_food = [None] * self.max_player_num
#         for id in range(max(self.valid_indices)+1):
#             px, py = self.player_positions[id][0], self.player_positions[id][1]
#             cur_dist_to_food[self.valid_indices.index(id)] = \
#                 min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])
#
#         # cur_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in food_locations])
#         #                     for (px, py) in self.player_positions]
#         # 距离塑形奖励（不存在/新加入的槽位为 0）
#         self.player_points = [0.01 * (prev_dist - cur_dist) if (prev_dist != None and cur_dist != None) else 0.0
#                               for prev_dist, cur_dist in zip(self.prev_dist_to_food, cur_dist_to_food)]
#         self.prev_dist_to_food = cur_dist_to_food
#
#         self.food_points = [0 for _ in range(self.max_food_num)]
#
#         # 围捕与贴近惩罚
#         player_id_counter = 0
#         for player_loc in player_locations:
#             real_agent_id = self.valid_indices.index(player_id_counter)
#             player_vicinities = [((player_loc[0] + a[0]), (player_loc[1] + a[1])) for a in
#                                  [(0, 1), (0, -1), (1, 0), (-1, 0)]]
#             for player_vic in player_vicinities:
#                 if player_vic in set_of_food_location:
#                     center = player_vic
#                     enumerated = enumerate(player_locations)
#                     close = [x for (x, (a, b)) in enumerated if abs(a - center[0]) + abs(b - center[1])
#                              <= self.coopRadius]
#                     if len(close) > 1:
#                         # 成功围捕
#                         self.player_points[real_agent_id] += self.groupMultiplier * len(close)
#                         food_index = food_locations.index(center)
#                         self.food_points[food_id[food_index]] += -1
#                         self.food_alive_statuses[food_id[food_index]] = False
#                         self.food_frozen_time[food_id[food_index]] = self.food_freeze_rate
#
#                         # 擦除该食物在网格上的显示
#                         self.grid[center[0]][center[1]] = 1
#                         if "full_rgb" in self.obs_type:
#                             self.RGB_grid[center[0]][center[1]] = [0, 0, 0]
#                         self.RGB_padded_grid[center[0] + self.pads][center[1] + self.pads] \
#                             = [0, 0, 0]
#
#                     else: # 单狼贴近惩罚
#                         self.player_points[real_agent_id] -= self.close_penalty
#
#             player_id_counter += 1
#
#         # 重绘仍存活的食物
#         for idx, food in enumerate(self.food_positions):
#             if self.food_alive_statuses[idx]:
#                 self.grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = 3
#                 if "full_rgb" in self.obs_type:
#                     self.RGB_grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = [255, 0, 0]
#                 self.RGB_padded_grid[self.food_positions[idx][0] + self.pads][self.food_positions[idx][1] + self.pads] \
#                     = [255, 0, 0]
#
#     """方法作用：推进一个时间步（玩家/食物动作 → 位置更新 → 碰撞处理 → 绘制 → 计算奖励）。
#            步骤：
#              1) 记录玩家动作，时间步 -1；推进食物冷却并尝试复活；
#              2) 按动作与朝向分别计算玩家与食物更新后的位置/朝向；
#              3) 解决“同一格冲突”：如多个实体想进同一格，则全部回滚到各自前一格；
#              4) 刷新 grid 与 RGB 缓存；
#              5) 调用 update_food_status() 计算围捕/塑形/惩罚与食物生死变更。
#            """
#     def update_state(self, hunter_collective_action, food_collective_action):
#         self.other_player_acts = hunter_collective_action
#         self.remaining_timesteps -= 1
#         self.update_status()
#         self.revive()
#
#         # 位置与朝向更新（玩家与食物）
#         prev_player_position = self.player_positions
#         prev_player_orientation = self.player_orientation
#         prev_food_position = self.food_positions
#         prev_food_orientation = self.food_orientation
#         #Notes notes
#         update_results_player = self.calculate_new_position(hunter_collective_action, prev_player_position,
#                                                             prev_player_orientation)
#         post_player_position = [(a, b) for (a, b, c) in update_results_player]
#         post_player_orientation = [c for (a, b, c) in update_results_player]
#         self.player_orientation = post_player_orientation
#
#         update_results_food = self.calculate_new_position(food_collective_action, prev_food_position,
#                                                           prev_food_orientation)
#         post_food_position = [(a, b) for (a, b, c) in update_results_food]
#         post_food_orientation = [c for (a, b, c) in update_results_food]
#         self.food_orientation = post_food_orientation
#
#         # ==== 同格冲突处理：任意时刻，玩家/食物不能落在同一格 ====
#         prev_positions = [None] * (len(prev_player_position) + len(prev_food_position))
#         post_positions = [None] * (len(prev_player_position) + len(prev_food_position))
#         types = [None] * (len(prev_player_position) + len(prev_food_position))
#         food_status = [False] * (len(prev_player_position) + len(prev_food_position))
#
#         for a in range(len(prev_player_position)):
#             prev_positions[a] = prev_player_position[a]
#             post_positions[a] = post_player_position[a]
#             types[a] = "player"
#
#         for a in range(len(prev_food_position)):
#             prev_positions[a + len(prev_player_position)] = prev_food_position[a]
#             post_positions[a + len(post_player_position)] = post_food_position[a]
#             types[a + len(post_player_position)] = "food"
#             if self.food_alive_statuses[a]:
#                 food_status[a + len(post_player_position)] = True
#
#         # 找到多重落点（长度>1 的 group），把这些实体回滚到原位，直到没有冲突
#         a, seen, result = post_positions, {}, {}
#         for idx, item in enumerate(a):
#             next_pass = True
#             if types[idx] == "food" and not food_status[idx]:
#                 next_pass = False
#
#             if next_pass:
#                 if item not in seen:
#                     result[item] = [idx]
#                     seen[item] = types[idx]
#                 else:
#                     result[item].append(idx)
#
#         groupings = list(result.values())
#         doubles = [t for t in groupings if len(t) > 1]
#         while len(doubles) > 0:
#             res = set([item for sublist in doubles for item in sublist])
#             for ii in range(len(post_positions)):
#                 if ii in res:
#                     post_positions[ii] = prev_positions[ii]
#
#             a, seen, result = post_positions, {}, {}
#             for idx, item in enumerate(a):
#                 next_pass = True
#                 if types[idx] == "food" and not food_status[idx]:
#                     next_pass = False
#
#                 if next_pass:
#                     if item not in seen:
#                         result[item] = [idx]
#                         seen[item] = types[idx]
#                     else:
#                         result[item].append(idx)
#
#             groupings = list(result.values())
#             doubles = [t for t in groupings if len(t) > 1]
#
#         # ====== 刷新网格显示（先清旧、后写新）======
#         for a in self.food_positions:
#             self.grid[a[0]][a[1]] = 1
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
#             self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
#         self.food_positions = post_positions[len(post_player_position):]
#
#         for idx, a in enumerate(self.food_positions):
#             if self.food_alive_statuses[idx]:
#                 self.grid[a[0]][a[1]] = 3
#                 if "full_rgb" in self.obs_type:
#                     self.RGB_grid[a[0]][a[1]] = [255, 0, 0]
#                 self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 0, 0]
#
#         for a in self.player_positions:
#             self.grid[a[0]][a[1]] = 1
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
#             self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
#         self.player_positions = post_positions[:len(post_player_position)]
#
#         for idx, a in enumerate(self.player_positions):
#             self.grid[a[0]][a[1]] = 2
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
#             self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
#
#         # 计算奖励 & 生死状态
#         self.update_food_status()
#
#     """方法作用：生成某个“潜在槽位”的观测。
#         vector 模式：
#           - 先写自己 (x,y)，再按槽位顺序写其他玩家 (x,y)，再把所有食物的信息拼上（位置+朝向 one-hot），最后附 exist_flag。
#           - 若该槽位无效（valid_indices[agent_id] == -1），整行置 -1，exist_flag=-1。
#         partial_obs / full_rgb 模式（用于 DQN prey 或可视化）：
#           - 根据朝向裁切一个扇区/矩形窗口返回。
#     """
#     def observation_computation(self, obs_type, agent_type="player", agent_id=0):
#         if obs_type == "vector":
#             observations = []
#             player_locs = [-1] * (2 * self.max_player_num)
#             exact_a_loc = self.valid_indices[agent_id]
#             if exact_a_loc >= 0:
#                 player_locs[0] = self.player_positions[exact_a_loc][0]
#                 player_locs[1] = self.player_positions[exact_a_loc][1]
#
#             pointer = 1
#             for other_agent_id in range(self.max_player_num):
#                 if other_agent_id != agent_id:
#                     exact_a_loc = self.valid_indices[other_agent_id]
#                     if exact_a_loc >= 0:
#                         player_locs[2*pointer] = self.player_positions[exact_a_loc][0]
#                         player_locs[2*pointer+1] = self.player_positions[exact_a_loc][1]
#                     pointer += 1
#
#             food_locs = [x for a in self.food_positions for x in list(a)]
#             observations.extend(player_locs)
#             observations.extend(food_locs)
#
#             # 食物朝向 one-hot
#             for orientation in self.food_orientation:
#                 or_vector = [0] * 4
#                 or_vector[orientation] = 1
#                 observations.extend(or_vector)
#
#             # 槽位有效性：-1 表无效，1 表有效（并且若无效整行置 -1，方便上游掩码）
#             if observations[0] == -1 :
#                 observations.append(-1)
#                 observations = [-1 for _ in range(len(observations))]
#             else:
#                 observations.append(1)
#
#             return np.asarray(observations)
#
#         elif obs_type == "partial_obs":
#             # 以朝向为参考，从 padded RGB 中裁一块可见区域
#             if agent_type == "player":
#                 orientation = self.player_orientation[agent_id]
#                 pos_0, pos_1 = self.player_positions[agent_id][0], self.player_positions[agent_id][1]
#             else:
#                 orientation = self.food_orientation[agent_id]
#                 pos_0, pos_1 = self.food_positions[agent_id][0], self.food_positions[agent_id][1]
#
#             pos_0 = pos_0 + self.pads
#             pos_1 = pos_1 + self.pads
#             obs_grid = np.asarray(self.RGB_padded_grid)
#
#             if orientation == 0:# 朝上：取上方扇区
#                 partial_ob = obs_grid[pos_0 - self.sight_radius:pos_0 + 1,
#                              pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
#
#
#             elif orientation == 1:# 朝右
#                 partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
#                              pos_1:pos_1 + self.sight_radius + 1]
#
#                 partial_ob = partial_ob.transpose((1, 0, 2))
#                 partial_ob = partial_ob[::-1]
#
#             elif orientation == 2: # 朝下
#                 partial_ob = obs_grid[pos_0:pos_0 + self.sight_radius + 1,
#                              pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
#                 partial_ob = np.fliplr(partial_ob)
#                 partial_ob = partial_ob[::-1]
#
#             elif orientation == 3: # 朝左
#                 partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
#                              pos_1 - self.sight_radius:pos_1 + 1]
#                 partial_ob = partial_ob.transpose((1, 0, 2))
#                 partial_ob = np.fliplr(partial_ob)
#
#             return partial_ob
#
#     """方法作用：环境一步交互（支持开放队伍规模）。
#             入参：
#               - action：长度=max_player_num 的列表，但只有 valid_indices>=0 的槽位有效；
#             步骤：
#               1) 将 action 重排为实际有效玩家顺序（restructured_action）；
#               2) 让食物智能体根据其观测 self.food_obses 选择动作；
#               3) 调用 update_state 进行世界推进与奖励更新；
#               4) 调用 scheduler.open_process() 决定是否删除/添加玩家，并更新 valid_indices；
#               5) 计算并缓存新的 food_obses（供下一步 prey 使用）；
#               6) 返回 player 端的 (obs, reward, done, info)（obs 为所有槽位）。
#             """
#     def step(self, action):
#         hunter_collective_action = list(action)
#         # 把“按槽位顺序”的动作，转成“按真实玩家索引顺序”的动作
#         restructured_action = [hunter_collective_action[self.valid_indices.index(idx)] for idx in range(
#                 max(self.valid_indices)+1
#         )]
#         # 让食物使用其 DQN 策略
#         food_collective_action = [prey.act(obs, epsilon=0.1) for prey, obs in zip(self.food_list, self.food_obses)]
#         # 推进世界
#         self.update_state(restructured_action, food_collective_action)
#
#         # 开放进程：可能删/加 agent（回合内变 N 的核心）
#         deleted, new_types = self.scheduler.open_process()
#         self.prev_valid_indices = self.valid_indices.copy()
#         self.del_agent(deleted)
#         self.add_agent(new_types)
#
#         # 生成玩家端返回（obs/rew/done/info）
#         player_returns = (np.asarray([self.observation_computation(self.obs_type, agent_id=id)
#                                  for id in range(self.max_player_num)]),
#                           self.player_points, [self.remaining_timesteps == 0 for a in range(len(self.player_points))],
#                           {})
#
#         #[self.remaining_timesteps == 0 for a in range(len(self.player_points))]
#         # 同步更新食物的观测（环境内部用，不返回）
#         food_returns = ([self.observation_computation(obs_type, agent_type="food", agent_id=id)
#                          for id, obs_type in enumerate(self.food_obs_type)],
#                         self.food_points, [self.remaining_timesteps == 0
#                                            for a in range(self.max_food_num)])
#
#         self.food_obses = food_returns[0]
#
#         return player_returns
#
#     """方法作用：用 pygame 可视化当前 grid。"""
#     def render(self, mode='human', close=False):
#         if self.visualizer is None:
#             self.visualizer = Visualizer(self.grid, self.grid_height, self.grid_width)
#
#         self.visualizer.grid = self.grid
#         self.visualizer.render()
#         self.visualizer.render()
#
#     """方法作用：增加若干玩家（OpenScheduler 触发后调用）。
#         步骤：
#             1) 随机从空地挑选不冲突的坐标；
#             2) 更新位置/朝向/网格显示；
#             3) 在 valid_indices 里为新来的玩家分配空槽位（值=当前最大真实索引+1）。
#     """
#     def add_agent(self, new_types):
#         def_orientation = 0
#         available_pos = list(set(self.possibleCoordinates) - set(self.player_positions) -
#                              set(self.food_positions))
#         pos_idxes = self.prng.choice(len(available_pos), len(new_types), replace=False).tolist()
#         added_pos = [available_pos[a] for a in pos_idxes]
#         orientation = [def_orientation for _ in range(len(added_pos))]
#         self.player_orientation.extend(orientation)
#         self.player_positions.extend(added_pos)
#         for a in added_pos:
#             self.grid[a[0]][a[1]] = 2
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
#             self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
#             self.num_players += 1
#             offset = max(self.valid_indices) + 1
#             possible_indices = [idx_s for idx_s, a in enumerate(self.valid_indices) if a == -1]
#             idx_val = self.prng.choice(len(possible_indices), 1).tolist()[0]
#             self.valid_indices[possible_indices[idx_val]] = offset
#
#     """方法作用：删除若干玩家（OpenScheduler 触发后调用）。
#         步骤：
#         1) 擦除其在网格的显示，弹出位置/朝向；
#         2) num_players -= 1；
#         3) 更新 valid_indices：>idx 的都左移一位（-1），等于 idx 的置 -1（空槽）。
#     """
#     def del_agent(self, agent_id):
#         for idx in agent_id:
#             a = self.player_positions[idx]
#             self.grid[a[0]][a[1]] = 1
#             if "full_rgb" in self.obs_type:
#                 self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
#             self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
#             self.player_positions.pop(idx)
#             self.player_orientation.pop(idx)
#             self.num_players -= 1
#             for idx_val, a in enumerate(self.valid_indices):
#                 if a > idx:
#                     self.valid_indices[idx_val] -= 1
#                 if a == idx:
#                     self.valid_indices[idx_val] = -1
#
# """开放调度器：以几何分布随机决定“删除/添加”玩家的数量与时机。
#     关键参数：
#       - add_rate / remove_rate：每步触发添加/删除的概率；
#       - max_available_agents：可容纳的最大玩家数（不会超过这个上限）；
#       - min_alive_time：玩家加入后至少存活多少步才有机会被删除（防止抖动）。
#     接口：
#       - open_process()：返回 (deleted_idxs, new_obs_type_list)，供环境执行增删。
# """
# class OpenScheduler(object):
#     def __init__(self, num_agents, add_rate, remove_rate, max_available_agents, min_alive_time=25, seed=0):
#         self.available_agents = num_agents
#         self.max_available_agents = max_available_agents
#         self.alive_time = [0] * self.available_agents
#         self.min_alive_time = min_alive_time
#         self.seed = seed
#         self.prng = RandomState(self.seed)
#         self.timestep = 0
#         # Use geometric distribution to sample remove or del
#         self.geometric_add_rate = add_rate
#         self.geometric_remove_rate = remove_rate
#
#     """添加 agent_nums 个玩家，返回它们的观测类型列表（这里简单返回 'vector'）。"""
#     def add_agents(self, agent_nums):
#         new_obs_type = []
#         for _ in range(agent_nums):
#             new_obs_type.append("vector")#登记新玩家的观测类型（可扩展为不同类型）。
#             self.available_agents += 1
#             self.alive_time.append(0) #为新玩家建立一条“存活时长计数器”，初始为 0
#         return new_obs_type
#
#     """删除指定下标的玩家（从 alive_time 中移除并递减 available_agents）。"""
#     def del_agents(self, agent_idxs):
#         agent_idxs_sorted = agent_idxs.copy()
#         agent_idxs_sorted.sort(reverse=True)
#         for idxes in agent_idxs_sorted:
#             del self.alive_time[idxes]
#         self.available_agents -= len(agent_idxs_sorted)
#         return agent_idxs_sorted # list[int]），要被移除的相对索引
#
#     """采样要删除的玩家索引：
#           仅选择“存活时间 > min_alive_time”的玩家，删除数量 ∈ {0,1,2}（受概率与可删人数约束）。
#     """
#     def agent_removal_sampler(self):
#         # 只有在线时间 > min_alive_time 的玩家才有资格被“抽走”
#         eligible_idxs = [idx for idx, alive_dur in enumerate(self.alive_time) if alive_dur > self.min_alive_time]
#         # 采一个“计划删除人数”的随机量（上限 2，以 0.7 的概率取1，以 0.3 的概率取 2）
#         removed_amount = min(min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1, self.available_agents-1),
#                              len(eligible_idxs))
#         removed_indices = []
#         if not removed_amount == 0:
#             #否则在 eligible_idxs 里无放回随机挑 removed_amount 个索引并返回
#             removed_indices = self.prng.choice(len(eligible_idxs), removed_amount, replace=False).tolist()
#             removed_indices = [eligible_idxs[k] for k in removed_indices]
#         return removed_indices
#
#     """方法作用：推进一个调度步，可能删除/添加玩家。
#         步骤：
#           1) alive_time 全 +1；
#           2) 以 remove_rate 决定是否删除；若删除 → agent_removal_sampler 采样并 del_agents；
#           3) 以 add_rate 决定是否添加；若添加 → 以 {1,2} 概率采样添加数量，受上限约束；
#           4) 返回 (deleted_idxs, new_obs_type_list)。
#     """
#     def open_process(self):
#         self.alive_time = [x+1 for x in self.alive_time] # 1) 全员寿命 +1
#         self.timestep+=1
#
#         remove = False
#         if self.prng.uniform() < self.geometric_remove_rate:#以伯努利触发一次“删除流程”
#             remove = True
#
#         add = False
#         if self.prng.uniform() < self.geometric_add_rate: #伯努利触发新增
#             add = True
#         deleted_idxs = []
#
#         if remove:
#             removed_indices = self.agent_removal_sampler() # 采样删除对象
#             deleted_idxs = self.del_agents(removed_indices) # 从调度器状态删除
#
#         new_obs_type = []
#         if add:
#             agent_nums = min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1,
#                                 self.max_available_agents - self.available_agents)
#             new_obs_type = self.add_agents(agent_nums)
#         return deleted_idxs, new_obs_type
#
# """基于 pygame 的简易网格渲染器：蓝色背景、黑色空地、白色玩家、红色食物。"""
# class Visualizer(object):
#     BLACK = (0, 0, 0)
#     WHITE = (255, 255, 255)
#     GREEN = (0, 255, 0)
#     RED = (255, 0, 0)
#     BLUE = (0, 0, 255)
#
#     WIDTH = 20
#     HEIGHT = 20
#
#     MARGIN = 0
#
#     # Create a 2 dimensional array. A two dimensional
#     # array is simply a list of lists.
#
#     def __init__(self, grid, grid_height=20, grid_width=20):
#         import pygame
#         self.pygame = pygame
#         self.grid_height, self.grid_width = grid_height, grid_width
#         self.grid = grid
#         self.WINDOW_SIZE = [self.grid_height * self.HEIGHT, self.grid_width * self.WIDTH]
#
#         self.pygame.init()
#         self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
#         self.pygame.display.set_caption("Wolfpack")
#         self.clock = pygame.time.Clock()
#
#     """渲染一帧网格；若关闭窗口则退出 pygame。"""
#     def render(self):
#         done = False
#         for event in self.pygame.event.get():  # User did something
#             if event.type == self.pygame.QUIT:  # If user clicked close
#                 done = True  # Flag that we are done so we exit this loop
#
#         self.screen.fill(self.BLACK)
#
#         for row in range(self.grid_height):
#             for column in range(self.grid_width):
#                 color = self.BLUE
#                 if self.grid[row][column] == 1:
#                     color = self.BLACK
#                 elif self.grid[row][column] == 2:
#                     color = self.WHITE
#                 elif self.grid[row][column] == 3:
#                     color = self.RED
#                 self.pygame.draw.rect(self.screen,
#                                       color,
#                                       [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
#                                        (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
#                                        self.WIDTH,
#                                        self.HEIGHT])
#
#         self.clock.tick(60)
#         self.pygame.display.flip()
#
#         if done:
#             self.pygame.quit()

import copy
import random
import pickle as pkl
import numpy as np
import threading
import gym
from gym import spaces
import time
import sys, os
from .assets.Agent import DQNAgent
from numpy.random import RandomState


"""
Generator：用cellular automata algorithm生成随机障碍地图（可通关检查联通性）。
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
      - 离散动作空间（共 6 个动作：前/右/后/左/停/左转/右转，其中实现把左转记为5，右转为其他分支）。
      - 观测支持 vector / partial_obs / full_rgb（此处主要使用 vector）。
      - OpenScheduler 决定每个时间步是否“增减”玩家，实现回合内 N 变化。
      - 食物(prey)由预训练的 DQNAgent 控制，可被“协作围捕”后冻结并在冷却后复活,因此回合在最大时间步时终止。
      valid_indices 是“当前在场玩家的真实索引列表”（例如 [0,2,3] 表示 0、2、3 号玩家此刻在场；1 号玩家此刻缺席）。
      采用固定长度的观测槽位（max_player_num），一个内部索引表来把“真实玩家集合”与这些槽位对应起来
      - 奖励：
          * 靠近食物有微小正向 shaping（上一步距离-本步距离）*0.01；
          * 单狼贴近食物但未形成围捕 => close_penalty 惩罚,鼓励协作而不是抢占位置；
          * 多狼共同在 coopRadius 范围内夹击,该只食物记 -1 并冻结一段时间后再复活 => groupMultiplier * 参与人数 的正奖励，食物 -1 并冻结。
"""
class WolfpackPenaltyOpen(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_height=20, grid_width=20, num_agents=5, max_food_num=2,
                 sight_sideways=8, sight_radius=8, max_time_steps=200,
                 coop_radius=1, groupMultiplier=2, food_freeze_rate=0,
                 add_rate=0.05, del_rate=0.05, seed=None,
                 max_player_num=5, implicit_max_player_num=3,
                 obs_type="vector", with_random_grid=False, random_grid_dir=None,
                 prey_with_gpu=False, close_penalty=0.1):

        # ====== 基本网格/观测与动作配置 ======
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_type = obs_type
        self.close_penalty = close_penalty
        self.num_agents = num_agents                     # 当前在场玩家数
        self.max_player_num = max_player_num               # 最大可容纳玩家数（固定观测维度用）
        self.implicit_max_player_num = implicit_max_player_num  # 初始“可用”玩家容量（scheduler 用）
        self.ma_obs_type = obs_type
        self.N_DISCRETE_ACTIONS = 6
        # 为了支持变 N，这里给每个 potential slot 一个 Discrete(6) 的动作空间（外部会只取有效索引）
        self.action_space = [gym.spaces.Discrete(6) for _ in range(max_player_num)]
        self.other_player_acts = None

        # ====== 初始玩家/食物数量与槽位管理 ======
        self.init_num_players = num_agents
        # valid_indices：长度=max_player_num；值=“该槽位对应的真实 players 列表下标”，-1 代表空槽
        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx
        self.prev_valid_indices = None
        self.max_food_num = max_food_num

        # ====== 观测空间（vector 版本）======
        if obs_type == "vector":
            # 单个 agent 的观测向量构成：
            # [自己(x,y), 其他玩家(x,y)* (max_player_num-1), 食物位置*(2*max_food_num), 食物朝向one-hot*(4*max_food_num), exist_flag]
            box_high = []
            for idx in range(self.max_player_num):
                box_high.append(self.grid_height - 1)  # x 上界
                box_high.append(self.grid_width - 1)   # y 上界
            for idx in range(6 * max_food_num):
                if idx % 6 == 0:
                    box_high.append(self.grid_height - 1)
                elif idx % 6 == 1:
                    box_high.append(self.grid_width - 1)
                else:
                    box_high.append(1)                 # 朝向 one-hot 的上界 or 标志位
            box_high.append(1)                          # 最尾的 exist_flag（1 or -1）

            # 下界全部置 0，exist_flag 的下界置 -1（无效 agent 行会全 -1）
            box_low = [0] * len(box_high)
            box_low[-1] = -1

            # 为所有 potential slots 复制相同的 Box 定义
            final_box_high = [box_high.copy() for _ in range(self.max_player_num)]
            final_box_low = [box_low.copy() for _ in range(self.max_player_num)]

            self.observation_space = [
                spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float64)
                for low, high in zip(final_box_low, final_box_high)
            ]

            # =================== 新增代码开始 ===================

            # 1. 定义全局状态的维度
            # 根据我们在 get_state() 中的设计：
            # 玩家信息: max_player_num * 3 (x, y, exist_flag)
            # 食物位置: max_food_num * 2 (x, y)
            # 食物朝向: max_food_num * 4 (One-hot)
            state_dim = (self.max_player_num * 3) + (self.max_food_num * 6)

            # 2. 定义 share_observation_space
            # 这是一个列表，长度为智能体数量，每个元素是一个 Box 对象描述 State 的形状
            self.share_observation_space = [
                spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
                for _ in range(self.max_player_num)
            ]

            # =================== 新增代码结束 ===================

        # ====== 随机/调度器/可视化状态 ======
        self.add_rate = add_rate  # 每步“增加玩家”的几何分布参数
        self.del_rate = del_rate  # 每步“删除玩家”的几何分布参数
        self.masking = []  # （预留）掩码
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        self.randomizer = random
        self.prng = RandomState(self.seed)
        # OpenScheduler：回合内增减玩家的“到达-离开”过程
        self.scheduler = OpenScheduler(self.num_agents, self.add_rate, self.del_rate, self.implicit_max_player_num,
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
        elif not self.with_random_grid: # 默认全可走
            self.levelMap = [[False] * k for k in [self.grid_width] * self.grid_height]
        else: # 自动机生成随机地图并连通性检查
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

        self.player_obs_type = [obs_type for _ in range(self.num_agents)]
        self.food_obs_type = ["partial_obs" for _ in range(self.max_food_num)]
        self.food_obses = None
        self.prey_with_gpu = prey_with_gpu


        # ====== 食物（猎物）智能体加载（预训练 DQN） ======
        self.remaining_timesteps = max_time_steps
        self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
                                   obs_type="partial_obs")
                          for a in range(self.max_food_num)]

        dirname = os.path.dirname(__file__)
        for idx, agent in enumerate(self.food_list):
            filename = os.path.join(dirname,
                                    ("assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
            agent.load_parameters(filename)

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
            self.levelMap = pkl.load(f)

    def get_state(self):
        """
        构造全局状态 (Global State)。
        包含：所有槽位的玩家坐标 + 所有食物的坐标 + 所有食物的朝向。
        注意：这里使用绝对坐标，不进行相对视角的转换。
        """
        state = []

        # A. 所有玩家的位置 (按槽位索引顺序 0 ~ max_player_num-1)
        # 如果槽位无效或玩家不存在，用 -1 填充，如果槽位有效则填坐标
        for i in range(self.max_player_num):
            real_idx = self.valid_indices[i]
            if real_idx != -1:  # 槽位上有真实玩家
                pos = self.player_positions[real_idx]
                state.extend([pos[0], pos[1]])  # x, y
                state.append(1)  # 存在标记
            else:
                state.extend([-1, -1])  # 无效位置
                state.append(0)  # 不存在标记

        # B. 所有食物的位置 (flat list)
        food_locs = [x for a in self.food_positions for x in list(a)]
        state.extend(food_locs)

        # C. 所有食物的朝向 (One-hot)
        for orientation in self.food_orientation:
            or_vector = [0] * 4
            or_vector[orientation] = 1
            state.extend(or_vector)

        return np.asarray(state, dtype=np.float32)


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

        self.num_agents = self.init_num_players
        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx
        self.prev_valid_indices = None

        self.scheduler = OpenScheduler(self.num_agents, self.add_rate, self.del_rate, self.implicit_max_player_num,
                                       seed=self.seed)

        # 重置网格、RGB
        self.RGB_padded_grid = [[[0, 0, 255] for _ in range(2 * self.pads + self.grid_width)] for a in range(2 * self.pads + self.grid_height)]

        # Reset grid locations
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if "full_rgb" in self.obs_type:
            # Reset RGB Grid
            self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]

        # 可走坐标（levelMap 为 False 的位置可走）
        self.possibleCoordinates = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if
                                        not i]

        # 采样玩家位置（不重复）
        player_loc_idx = self.prng.choice(range(len(self.possibleCoordinates)), self.num_agents, replace=False).tolist()
        # player_loc_idx = random.sample(range(len(self.possibleCoordinates)), self.num_players)
        self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]

        # Reset initial player orientation and points
        self.player_orientation = [0 for a in range(self.num_agents)]
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


        # 初始化“距离食物”的塑形基线（仅为有效槽位计算）
        self.prev_dist_to_food = [None] * self.max_player_num
        for id in range(max(self.valid_indices)+1):
            px, py = self.player_positions[id][0], self.player_positions[id][1]
            self.prev_dist_to_food[self.valid_indices.index(id)] = \
                min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])
        # self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
        #                           for (px, py) in self.player_positions]
        self.remaining_timesteps = self.max_time_steps

        # 计算初始观测
        self.player_obs_type = [self.obs_type for _ in range(self.num_agents)]

        # --- 修改开始 ---
        # 1. 获取 Local Observations (obs)
        player_obses = [self.observation_computation(self.ma_obs_type, agent_id=id) for id in range(self.max_player_num)]

        # 2. 获取 Global State (share_obs)
        global_state = self.get_state()
        # 复制 state 以匹配 agent 数量 (因为 DDFG 的 ShareVecEnv 通常期望每个 agent 都有对应的 share_obs)
        # 结果是一个列表: [state, state, ..., state]
        share_obses = [global_state.copy() for _ in range(self.max_player_num)]

        # 3. 获取 Available Actions
        available_actions = []
        for i in range(self.max_player_num):
            # 判断该槽位是否有玩家 (vector obs 最后一位是 exist_flag)
            # 或者直接检查 self.valid_indices[i] != -1
            obs = player_obses[i]
            # 假设 obs 的最后一位是 exist_flag (1:存在, -1:不存在)
            exist_flag = obs[-1]

            avail = np.zeros(6, dtype=np.float32)
            if exist_flag > 0:
                # 存在的 agent：所有动作可用
                avail[:] = 1.0
            else:
                # 不存在的 agent：只有 No-Op (Action 4) 可用
                avail[4] = 1.0
            available_actions.append(avail)

        # 重新实例化并加载食物策略参数（避免并行进程下的引用问题）
        food_obses = [self.observation_computation(obs_type, agent_type="food", agent_id=id) for id, obs_type in
                      enumerate(self.food_obs_type)]
        self.food_obses = food_obses
        self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
                                       obs_type="partial_obs")
                              for a in range(self.max_food_num)]
        dirname = os.path.dirname(__file__)
        for idx, agent in enumerate(self.food_list):
            filename = os.path.join(dirname,
                                        ("assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
            agent.load_parameters(filename)

        # 返回 SMAC 风格的三元组: obs, share_obs, avail_acts
        return player_obses, share_obses, available_actions

    """方法作用：让“冷却结束”的已死食物随机复活到不冲突的空地。
        步骤：
            1) 找出可复活的食物索引（alive_status=False 且 frozen_time<=0）；
            2) 在不含玩家与存活食物的位置里随机采样新坐标；
            3) 标记这些食物为存活并更新其位置；
            4) 更新 prev_dist_to_food（用于塑形奖励基线）。
    """
    def revive(self):
        # find possible locations to revive dead prey
        coordinates_no_player = [a for a in self.possibleCoordinates if
                                 a not in self.player_positions and a not in self.food_positions]
        revived_idxes = []
        for idx, food in enumerate(self.food_positions):
            if self.food_frozen_time[idx] <= 0 and not self.food_alive_statuses[idx]:
                revived_idxes.append(idx)

        if len(revived_idxes) > 0:
            idxes = []
            for k in range(len(revived_idxes)):
                idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
                # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
                while idx in idxes:
                    idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
                    # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
                idxes.append(idx)
            coords = [coordinates_no_player[idx] for idx in idxes]

            coord_idx = 0
            for idx in revived_idxes:
                self.food_alive_statuses[idx] = True
                self.food_positions[idx] = coords[coord_idx]
                coord_idx += 1

        # self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
        #                           for (px, py) in self.player_positions]

        self.prev_dist_to_food = [None] * self.max_player_num
        for id in range(max(self.valid_indices)+1):
            px, py = self.player_positions[id][0], self.player_positions[id][1]
            self.prev_dist_to_food[self.valid_indices.index(id)] = \
                min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])

    """方法作用：推进所有“已死食物”的冷却时间（每步 -1）。"""
    def update_status(self):
        for idx in range(len(self.food_alive_statuses)):
            if not self.food_alive_statuses[idx]:
                self.food_frozen_time[idx] -= 1

    """方法作用：批量计算一组实体的下一位置与朝向（包装器）。
        步骤：
        1) zip 三个列表（动作 / 位置 / 朝向）；
        2) 对每个实体调用 calculate_indiv_position；
        3) 返回 [(nx,ny,ori), ...]。
    """
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
        x = pair[0]
        y = pair[1]
        next_x = x
        next_y = y

        # go forward
        if action == 0:  # 前进
            # Facing upwards
            if orientation == 0:
                next_x -= 1
            # Facing right
            elif orientation == 1:
                next_y += 1
            # Facing downwards
            elif orientation == 2:
                next_x += 1
            else:
                next_y -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)

        # Step right
        elif action == 1:  # 右移（相对朝向）
            # Facing upwards
            if orientation == 0:
                next_y += 1
            # Facing right
            elif orientation == 1:
                next_x += 1
            # Facing downwards
            elif orientation == 2:
                next_y -= 1
            else:
                next_x -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)

        # Step back
        elif action == 2:  # 后退
            # Facing upwards
            if orientation == 0:
                next_x += 1
            # Facing right
            elif orientation == 1:
                next_y -= 1
            # Facing downwards
            elif orientation == 2:
                next_x -= 1
            else:
                next_y += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)

        # Step left
        elif action == 3:  # 左移（相对朝向）
            # Facing upwards
            if orientation == 0:
                next_y -= 1
            # Facing right
            elif orientation == 1:
                next_x -= 1
            # Facing downwards
            elif orientation == 2:
                next_y += 1
            else:
                next_x += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)

        # stay still
        elif action == 4:  # 不动
            return (x, y, orientation)

        # rotate left
        elif action == 5:  # 左转
            new_orientation = 0
            if orientation == 0:
                new_orientation = 3
            elif orientation == 1:
                new_orientation = 0
            elif orientation == 2:
                new_orientation = 1
            else:
                new_orientation = 2

            return (x, y, new_orientation)

        # rotate right
        else:  # 右转
            new_orientation = 0
            if orientation == 0:
                new_orientation = 1
            elif orientation == 1:
                new_orientation = 2
            elif orientation == 2:
                new_orientation = 3
            else:
                new_orientation = 0

            return (x, y, new_orientation)

    """方法作用：根据玩家/食物位置关系更新奖励与食物存活状态（围捕/贴近惩罚/塑形）。
          步骤：
            1) 计算每名“有效玩家”的当前最近食物曼哈顿距离，与上一步比较给 0.01*(prev-cur)；
            2) 对每名玩家的四邻（上下左右）若碰到食物：
               - 统计 coopRadius 范围内靠近此食物的玩家数 close；
               - 若 close>1：触发围捕 -> 玩家加 groupMultiplier*close，食物扣 1、标记死亡、开始冷却；
               - 否则：单狼贴近 -> 玩家扣 close_penalty；
            3) 把仍存活食物重绘回网格（RGB 同步）。
          """
    def update_food_status(self):
        self.food_points = [0 for a in range(self.max_food_num)]

        enumFood = list(enumerate(self.food_positions))
        food_locations = [(food[0], food[1]) for idx, food in enumFood if self.food_alive_statuses[idx]]
        food_id = [idx for idx, food in enumFood if self.food_alive_statuses[idx]]

        player_locations = self.player_positions
        set_of_food_location = set(food_locations)

        # 当前距离（仅对有效槽位）
        cur_dist_to_food = [None] * self.max_player_num
        for id in range(max(self.valid_indices)+1):
            px, py = self.player_positions[id][0], self.player_positions[id][1]
            cur_dist_to_food[self.valid_indices.index(id)] = \
                min([abs(px - fx) + abs(py - fy) for fx, fy in self.food_positions])

        # cur_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in food_locations])
        #                     for (px, py) in self.player_positions]
        # 距离塑形奖励（不存在/新加入的槽位为 0）
        self.player_points = [0.1 * (prev_dist - cur_dist) if (prev_dist != None and cur_dist != None) else 0.0
                              for prev_dist, cur_dist in zip(self.prev_dist_to_food, cur_dist_to_food)]
        self.prev_dist_to_food = cur_dist_to_food

        self.food_points = [0 for _ in range(self.max_food_num)]

        # 围捕与贴近惩罚
        player_id_counter = 0
        for player_loc in player_locations:
            real_agent_id = self.valid_indices.index(player_id_counter)
            player_vicinities = [((player_loc[0] + a[0]), (player_loc[1] + a[1])) for a in
                                 [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            for player_vic in player_vicinities:
                if player_vic in set_of_food_location:
                    center = player_vic
                    enumerated = enumerate(player_locations)
                    close = [x for (x, (a, b)) in enumerated if abs(a - center[0]) + abs(b - center[1])
                             <= self.coopRadius]
                    if len(close) > 1:
                        # 成功围捕
                        self.player_points[real_agent_id] += self.groupMultiplier * len(close)
                        food_index = food_locations.index(center)
                        self.food_points[food_id[food_index]] += -1
                        self.food_alive_statuses[food_id[food_index]] = False
                        self.food_frozen_time[food_id[food_index]] = self.food_freeze_rate

                        # 擦除该食物在网格上的显示
                        self.grid[center[0]][center[1]] = 1
                        if "full_rgb" in self.obs_type:
                            self.RGB_grid[center[0]][center[1]] = [0, 0, 0]
                        self.RGB_padded_grid[center[0] + self.pads][center[1] + self.pads] \
                            = [0, 0, 0]

                    else: # 单狼贴近惩罚
                        self.player_points[real_agent_id] -= self.close_penalty

            player_id_counter += 1

        # 重绘仍存活的食物
        for idx, food in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = 3
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = [255, 0, 0]
                self.RGB_padded_grid[self.food_positions[idx][0] + self.pads][self.food_positions[idx][1] + self.pads] \
                    = [255, 0, 0]

    """方法作用：推进一个时间步（玩家/食物动作 → 位置更新 → 碰撞处理 → 绘制 → 计算奖励）。
           步骤：
             1) 记录玩家动作，时间步 -1；推进食物冷却并尝试复活；
             2) 按动作与朝向分别计算玩家与食物更新后的位置/朝向；
             3) 解决“同一格冲突”：如多个实体想进同一格，则全部回滚到各自前一格；
             4) 刷新 grid 与 RGB 缓存；
             5) 调用 update_food_status() 计算围捕/塑形/惩罚与食物生死变更。
           """
    def update_state(self, hunter_collective_action, food_collective_action):
        self.other_player_acts = hunter_collective_action
        self.remaining_timesteps -= 1
        self.update_status()
        self.revive()

        # 位置与朝向更新（玩家与食物）
        prev_player_position = self.player_positions
        prev_player_orientation = self.player_orientation
        prev_food_position = self.food_positions
        prev_food_orientation = self.food_orientation
        #Notes notes
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
            observations = []
            player_locs = [-1] * (2 * self.max_player_num)
            exact_a_loc = self.valid_indices[agent_id]
            if exact_a_loc >= 0:
                player_locs[0] = self.player_positions[exact_a_loc][0]
                player_locs[1] = self.player_positions[exact_a_loc][1]

            pointer = 1
            for other_agent_id in range(self.max_player_num):
                if other_agent_id != agent_id:
                    exact_a_loc = self.valid_indices[other_agent_id]
                    if exact_a_loc >= 0:
                        player_locs[2*pointer] = self.player_positions[exact_a_loc][0]
                        player_locs[2*pointer+1] = self.player_positions[exact_a_loc][1]
                    pointer += 1

            food_locs = [x for a in self.food_positions for x in list(a)]
            observations.extend(player_locs)
            observations.extend(food_locs)

            # 食物朝向 one-hot
            for orientation in self.food_orientation:
                or_vector = [0] * 4
                or_vector[orientation] = 1
                observations.extend(or_vector)

            # 槽位有效性：-1 表无效，1 表有效（并且若无效整行置 -1，方便上游掩码）
            if observations[0] == -1 :
                observations.append(-1)
                observations = [-1 for _ in range(len(observations))]
            else:
                observations.append(1)

            return np.asarray(observations)

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

            if orientation == 0:# 朝上：取上方扇区
                partial_ob = obs_grid[pos_0 - self.sight_radius:pos_0 + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]


            elif orientation == 1:# 朝右
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1:pos_1 + self.sight_radius + 1]

                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = partial_ob[::-1]

            elif orientation == 2: # 朝下
                partial_ob = obs_grid[pos_0:pos_0 + self.sight_radius + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
                partial_ob = np.fliplr(partial_ob)
                partial_ob = partial_ob[::-1]

            elif orientation == 3: # 朝左
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1 - self.sight_radius:pos_1 + 1]
                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = np.fliplr(partial_ob)

            return partial_ob

    """方法作用：环境一步交互（支持开放队伍规模）。
            入参：
              - action：长度=max_player_num 的列表，但只有 valid_indices>=0 的槽位有效；
            步骤：
              1) 将 action 重排为实际有效玩家顺序（restructured_action）；
              2) 让食物智能体根据其观测 self.food_obses 选择动作；
              3) 调用 update_state 进行世界推进与奖励更新；
              4) 调用 scheduler.open_process() 决定是否删除/添加玩家，并更新 valid_indices；
              5) 计算并缓存新的 food_obses（供下一步 prey 使用）；
              6) 返回 player 端的 (obs, reward, done, info)（obs 为所有槽位）。
            """
    def step(self, action):
        hunter_collective_action = list(action)
        # 把“按槽位顺序”的动作，转成“按真实玩家索引顺序”的动作
        restructured_action = [hunter_collective_action[self.valid_indices.index(idx)] for idx in range(
                max(self.valid_indices)+1
        )]
        # 让食物使用其 DQN 策略
        food_collective_action = [prey.act(obs, epsilon=0.1) for prey, obs in zip(self.food_list, self.food_obses)]
        # 推进世界
        self.update_state(restructured_action, food_collective_action)

        # 开放进程：可能删/加 agent（回合内变 N 的核心）
        deleted, new_types = self.scheduler.open_process()
        self.prev_valid_indices = self.valid_indices.copy()
        self.del_agent(deleted)
        self.add_agent(new_types)

        # 1. Obs
        player_obses = [self.observation_computation(self.obs_type, agent_id=id)
                        for id in range(self.max_player_num)]

        # 2. Share Obs
        global_state = self.get_state()
        share_obses = [global_state.copy() for _ in range(self.max_player_num)]

        # 3. Rewards
        rewards=self.player_points

        # 4. Dones
        dones=[self.remaining_timesteps == 0 for a in range(len(self.player_points))]

        # 5. Infos
        infos = [{} for _ in range(self.num_agents)]

        # 6. Available Actions
        available_actions = []
        for i in range(self.max_player_num):
            exist_flag = player_obses[i][-1]
            avail = np.zeros(6, dtype=np.float32)
            if exist_flag > 0:
                avail[:] = 1.0
            else:
                avail[4] = 1.0  # No-Op
            available_actions.append(avail)

        #[self.remaining_timesteps == 0 for a in range(len(self.player_points))]
        # 同步更新食物的观测（环境内部用，不返回）
        food_returns = ([self.observation_computation(obs_type, agent_type="food", agent_id=id)
                         for id, obs_type in enumerate(self.food_obs_type)],
                        self.food_points, [self.remaining_timesteps == 0
                                           for a in range(self.max_food_num)])

        self.food_obses = food_returns[0]

        return player_obses, share_obses, rewards, dones, infos, available_actions

    """方法作用：用 pygame 可视化当前 grid。"""
    def render(self, mode='human', close=False):
        if self.visualizer is None:
            self.visualizer = Visualizer(self.grid, self.grid_height, self.grid_width)

        self.visualizer.grid = self.grid
        self.visualizer.render()
        self.visualizer.render()

    """方法作用：增加若干玩家（OpenScheduler 触发后调用）。
        步骤：
            1) 随机从空地挑选不冲突的坐标；
            2) 更新位置/朝向/网格显示；
            3) 在 valid_indices 里为新来的玩家分配空槽位（值=当前最大真实索引+1）。
    """
    def add_agent(self, new_types):
        def_orientation = 0
        available_pos = list(set(self.possibleCoordinates) - set(self.player_positions) -
                             set(self.food_positions))
        pos_idxes = self.prng.choice(len(available_pos), len(new_types), replace=False).tolist()
        added_pos = [available_pos[a] for a in pos_idxes]
        orientation = [def_orientation for _ in range(len(added_pos))]
        self.player_orientation.extend(orientation)
        self.player_positions.extend(added_pos)
        for a in added_pos:
            self.grid[a[0]][a[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
            self.num_agents += 1
            offset = max(self.valid_indices) + 1
            possible_indices = [idx_s for idx_s, a in enumerate(self.valid_indices) if a == -1]
            idx_val = self.prng.choice(len(possible_indices), 1).tolist()[0]
            self.valid_indices[possible_indices[idx_val]] = offset

    """方法作用：删除若干玩家（OpenScheduler 触发后调用）。
        步骤：
        1) 擦除其在网格的显示，弹出位置/朝向；
        2) num_agents -= 1；
        3) 更新 valid_indices：>idx 的都左移一位（-1），等于 idx 的置 -1（空槽）。
    """
    def del_agent(self, agent_id):
        for idx in agent_id:
            a = self.player_positions[idx]
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
            self.player_positions.pop(idx)
            self.player_orientation.pop(idx)
            self.num_agents -= 1
            for idx_val, a in enumerate(self.valid_indices):
                if a > idx:
                    self.valid_indices[idx_val] -= 1
                if a == idx:
                    self.valid_indices[idx_val] = -1

"""开放调度器：以几何分布随机决定“删除/添加”玩家的数量与时机。
    关键参数：
      - add_rate / remove_rate：每步触发添加/删除的概率；
      - max_available_agents：可容纳的最大玩家数（不会超过这个上限）；
      - min_alive_time：玩家加入后至少存活多少步才有机会被删除（防止抖动）。
    接口：
      - open_process()：返回 (deleted_idxs, new_obs_type_list)，供环境执行增删。
"""
class OpenScheduler(object):
    def __init__(self,num_agents, add_rate, remove_rate, max_available_agents, min_alive_time=25, seed=0):
        self.available_agents = num_agents
        self.max_available_agents = max_available_agents
        self.alive_time = [0] * self.available_agents
        self.min_alive_time = min_alive_time
        self.seed = seed
        self.prng = RandomState(self.seed)
        self.timestep = 0
        # Use geometric distribution to sample remove or del
        self.geometric_add_rate = add_rate
        self.geometric_remove_rate = remove_rate

    """添加 agent_nums 个玩家，返回它们的观测类型列表（这里简单返回 'vector'）。"""
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
        # 采一个“计划删除人数”的随机量（上限 2，以 0.7 的概率取1，以 0.3 的概率取 2）
        removed_amount = min(min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1, self.available_agents-1),
                             len(eligible_idxs))
        removed_indices = []
        if not removed_amount == 0:
            #否则在 eligible_idxs 里无放回随机挑 removed_amount 个索引并返回
            removed_indices = self.prng.choice(len(eligible_idxs), removed_amount, replace=False).tolist()
            removed_indices = [eligible_idxs[k] for k in removed_indices]
        return removed_indices

    """方法作用：推进一个调度步，可能删除/添加玩家。
        步骤：
          1) alive_time 全 +1；
          2) 以 remove_rate 决定是否删除；若删除 → agent_removal_sampler 采样并 del_agents；
          3) 以 add_rate 决定是否添加；若添加 → 以 {1,2} 概率采样添加数量，受上限约束；
          4) 返回 (deleted_idxs, new_obs_type_list)。
    """
    def open_process(self):
        self.alive_time = [x+1 for x in self.alive_time] # 1) 全员寿命 +1
        self.timestep+=1

        remove = False
        if self.prng.uniform() < self.geometric_remove_rate:#以伯努利触发一次“删除流程”
            remove = True

        add = False
        if self.prng.uniform() < self.geometric_add_rate: #伯努利触发新增
            add = True
        deleted_idxs = []

        if remove:
            removed_indices = self.agent_removal_sampler() # 采样删除对象
            deleted_idxs = self.del_agents(removed_indices) # 从调度器状态删除

        new_obs_type = []
        if add:
            agent_nums = min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1,
                                self.max_available_agents - self.available_agents)
            new_obs_type = self.add_agents(agent_nums)
        return deleted_idxs, new_obs_type

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
