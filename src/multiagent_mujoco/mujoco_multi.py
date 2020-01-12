from copy import deepcopy
import gym
from gym.spaces import Box
from multiagent.environment import MultiAgentEnv as OpenAIMultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np


from .multiagentenv import MultiAgentEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs

# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"] # e.g. Ant-v2
        self.agent_conf = kwargs["env_args"]["agent_conf"] # e.g. '2x3'

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals  = get_parts_and_edges(self.scenario,
                                                                                                  self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get("agent_obsk", None) # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get("agent_obsk_agents", False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v2"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v2", "HumanoidStandup-v2"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                elif self.scenario in ["Reacher-v2"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"

            k_split = self.k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk+1)]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            self.global_categories = self.global_categories_label.split(",") if self.global_categories_label is not None else []

        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id,
                                                    self.agent_partitions,
                                                    self.mujoco_edges,
                                                    k=self.agent_obsk,
                                                    kagents=False,) for agent_id in range(self.n_agents)]

        # load scenario from script
        self.episode_limit = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            self.wrapped_env = NormalizedActions(gym.make(self.scenario))
        else:
            assert False,  "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()

        pass

    def step(self, actions):
        flat_actions = np.concatenate([actions[i] for i in range(self.n_agents)])
        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(flat_actions)
        self.steps += 1

        info = {}
        info.update(info_n)

        if done_n:
            if self.steps < self.episode_limit:
                info["episode_limit"] = False   # the next state will be masked out
            else:
                info["episode_limit"] = True    # the next state will not be masked out

        return reward_n, done_n, info

    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        obs_n = []
        for a in range(self.n_agents):
            obs_n.append(self.get_obs_agent(a))
        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            return build_obs(self.env,
                                  self.k_dicts[agent_id],
                                  self.k_categories,
                                  self.mujoco_globals,
                                  self.global_categories,
                                  vec_len=getattr(self, "obs_size", None))

    def get_obs_size(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])


    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        return self.env._get_obs()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self): # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def get_env_info(self):
        from gym import spaces
        acdimjoint = self.env.action_space.low.shape[0]
        acdim = acdimjoint // self.n_agents
        action_spaces = tuple([Box(self.env.action_space.low[
                                   a*acdim:(a+1)*acdim], self.env.action_space.high[
                                   a*acdim:(a+1)*acdim]) for a in range(self.n_agents)])
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": action_spaces,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info

if __name__ == "__main__":
    bthigh = Node("bthigh", 3, 12)
    bshin = Node("bshin", 4, 13)
    bfoot = Node("bfoot", 5, 14)
    fthigh = Node("fthigh", 6, 15)
    fshin = Node("fshin", 7, 16)
    ffoot = Node("ffoot", 8, 17)

    edges = [Edge(bfoot, bshin),
             Edge(bshin, bthigh),
             Edge(bthigh, fthigh),
             Edge(fthigh, fshin),
             Edge(fshin, ffoot)]

    agent_partition = [(bfoot, bshin, bthigh),
                       (ffoot, fshin, fthigh)]

    obslen = 10
    qpos = np.arange(20)
    qvel = np.arange(20)
    nobs = get_nearest_obs(1, agent_partition, edges, obslen, qpos, qvel, k=1)
    print(nobs)
