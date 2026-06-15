"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod


class CloudpickleWrapper(object):
    """
    使用 cloudpickle 序列化环境，确保多进程传递环境对象时不报错。
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

"""
MARL 向量化环境基类。
专门针对返回 (obs, share_obs, rewards, dones, infos, avail_acts) 的环境。
"""
class ShareVecEnv(ABC):

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        """重置所有环境"""
        pass

    @abstractmethod
    def step_async(self, actions):
        """异步发送动作"""
        pass

    @abstractmethod
    def step_wait(self):
        """等待并接收结果"""
        pass

    def step(self, actions):
        """同步执行一步"""
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def close_extras(self):
        pass


# -------------------------------------------------------------------
# 1. 串行包装器 (Dummy) - 用于调试或单线程运行
# -------------------------------------------------------------------
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(env_fns), env.observation_space,
                             env.share_observation_space, env.action_space)
        self.actions = None
        # 读取环境属性
        self.num_agents = env.max_player_num


    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # 串行执行所有环境的 step
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        # 解包结果：results 是一个 list of tuples
        # zip(*results) 会把所有环境的 obs 放在一起，所有 rews 放在一起...
        obs, share_obs, rews, dones, infos, available_actions = map(np.array, zip(*results))
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()


# -------------------------------------------------------------------
# 2. 并行包装器 (Subproc) - 用于大规模训练加速
# -------------------------------------------------------------------

# 子进程运行的函数
def share_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                # 执行环境 step
                ob, s_ob, reward, done, info, available_actions = env.step(data)
                # Wolfpack 的 done 已经是列表格式，不需要像 Gym 那样处理自动 reset
                # 但如果环境全部结束，通常 Runner 会调用 reset，或者环境内部自动 reset
                # 这里我们保持原样返回，由 Runner 决定是否 reset
                if 'bool' in done.__class__.__name__:  # 处理单个 bool done 的情况 (防御性编程)
                    if done: ob, s_ob, available_actions = env.reset()
                elif np.all(done):  # 处理 array done
                    ob, s_ob, available_actions = env.reset()

                remote.send((ob, s_ob, reward, done, info, available_actions))

            elif cmd == 'reset':
                ob, s_ob, available_actions = env.reset()
                remote.send((ob, s_ob, available_actions))

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_attr':
                # 通用获取属性的方法
                remote.send(getattr(env, data, None))

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        env_fns: 返回环境实例的函数列表
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        # 创建管道
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # 启动子进程
        self.ps = [Process(target=share_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # 设置为守护进程，主进程死掉子进程也会死
            p.start()

        for remote in self.work_remotes:
            remote.close()  # 主进程关闭不需要的一端

        # 获取环境基本信息 (只需问第一个环境)
        self.remotes[0].send(('get_attr', 'observation_space'))
        observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_attr', 'share_observation_space'))
        share_observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_attr', 'action_space'))
        action_space = self.remotes[0].recv()

        self.remotes[0].send(('get_attr', 'num_agents'))
        self.num_agents = self.remotes[0].recv()

        # 获取 unit_dim (如果不存在则返回 0)
        self.remotes[0].send(('get_attr', 'unit_dim'))
        self.unit_dim = self.remotes[0].recv()
        if self.unit_dim is None: self.unit_dim = 0

        ShareVecEnv.__init__(self, nenvs, observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True