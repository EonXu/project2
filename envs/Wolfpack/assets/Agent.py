import random
from .ReplayMemory import ReplayMemoryLite
from .QNetwork import DQN
from .misc import hard_copy, soft_copy
import torch
import torch.optim as optim
import numpy as np

class Agent(object):
    def __init__(self, agent_id, obs_type):
        self.agent_id = agent_id
        self.obs_type = obs_type

    def get_obstype(self):
        return self.obs_type

class DQNAgent(Agent):
    def __init__(self, agent_id, args=None, obs_type="partial_obs",
                 obs_height=9, obs_width=17, mode="test"):
        super(DQNAgent, self).__init__(agent_id, obs_type)

        self.agent_id = agent_id
        self.obs_type = obs_type
        self.args = {} if args is None else args
        self.color = (255, 0, 0)
        self.mode = mode

        # 私有 RNG，不再使用 random.random / np.random 全局状态
        base_seed = int(self.args.get("seed", 0))
        self.seed = base_seed + int(agent_id) * 1009
        self.rng = np.random.RandomState(self.seed)

        self.experience_replay = ReplayMemoryLite(
            state_h=obs_height,
            state_w=obs_width,
            with_gpu=self.args.get("with_gpu", False),
            seed=self.seed + 17,
        )

        self.dqn_net = DQN(
            17, 9, 32,
            self.args["max_seq_length"],
            7,
            mode="partial"
        )

        if not self.mode == "test":
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args["lr"])
            self.target_dqn_net = DQN(
                17, 9, 32,
                self.args["max_seq_length"],
                7,
                mode="partial"
            )
            hard_copy(self.target_dqn_net, self.dqn_net)

        self.recent_obs_storage = np.zeros(
            [self.args["max_seq_length"], obs_height, obs_width, 3],
            dtype=np.float32
        )

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def act(self, obs, added_features=None, mode="train", epsilon=0.01):
        self.recent_obs_storage = np.roll(self.recent_obs_storage, axis=0, shift=-1)
        self.recent_obs_storage[-1] = obs

        net_inp = torch.as_tensor(
            self.recent_obs_storage.transpose([0, 3, 1, 2])[None],
            dtype=torch.float32
        )

        with torch.no_grad():
            _, indices = torch.max(self.dqn_net(net_inp), dim=-1)

        action = int(indices.item())

        # 当前 wolfpack 里 DQNAgent 默认 mode="test"，通常不会进入此分支；
        # 保留确定性私有 RNG，防止后续把 prey 改成 train/eval-random 时失控。
        if self.mode != "test":
            if self.rng.rand() < float(epsilon):
                action = int(self.rng.randint(0, 7))

        return action

    def store_exp(self, exp):
        self.experience_replay.insert(exp)

    def get_obs_type(self):
        return self.obs_type

    def update(self):
        if self.experience_replay.size < self.args['sampling_wait_time']:
            return
        batched_data = self.experience_replay.sample(self.args['batch_size'])
        state, action, reward, dones, next_states = batched_data[0], batched_data[1], batched_data[2], \
                                                    batched_data[3], batched_data[4]

        state = state.permute(0, 1, 4, 2, 3)
        next_states = next_states.permute(0, 1, 4, 2, 3)

        predicted_value = self.dqn_net(state).gather(1, action.long())
        target_values = reward + self.args['disc_rate'] * (1 - dones) * torch.max(self.target_dqn_net(next_states),
                                                                                  dim=-1, keepdim=True)[0]
        loss = 0.5 * torch.mean((predicted_value - target_values.detach()) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net)