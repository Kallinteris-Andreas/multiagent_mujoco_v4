from functools import partial
import gymnasium as gym
import gymnasium
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
import pettingzoo
import numpy as np
import numpy

from .multiagentenv import MultiAgentEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs

# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
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


class MujocoMulti(pettingzoo.utils.env.ParallelEnv):
    def __init__(self, batch_size=None, **kwargs):
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Ant-v4
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario, self.agent_conf)

        #Petting Zoo API
        self.possible_agents = range(len(self.agent_partitions))
        self.agents = self.possible_agents

        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get("agent_obsk", None) # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get("agent_obsk_agents", False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v4", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                elif self.scenario in ["Reacher-v4"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
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
                                                kagents=False,) for agent_id in range(self.num_agents)]

        # load scenario from script
        self.env_version = kwargs["env_args"].get("env_version", 4)
        if self.env_version == 4:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:  # env not in gym
                assert False, 'not tested'
                if self.scenario in ["manyagent_ant"]:
                    from .manyagent_ant import ManyAgentAntEnv as this_env
                elif self.scenario in ["manyagent_swimmer"]:
                    from .manyagent_swimmer import ManyAgentSwimmerEnv as this_env
                elif self.scenario in ["coupled_half_cheetah"]:
                    from .coupled_half_cheetah import CoupledHalfCheetah as this_env
                else:
                    raise NotImplementedError('Custom env not implemented!')
                self.wrapped_env = NormalizedActions(this_env(**kwargs["env_args"]))
        else:
            assert False,  "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        #self.obs_size = self.get_obs_size()

        #Petting ZOO API
        self.observation_spaces, self.action_spaces = {}, {}
        for a, partition in enumerate(self.agent_partitions):
            self.action_spaces[a] = gymnasium.spaces.Box(low=-1, high=1, shape=(len(partition),), dtype=numpy.float32) #TODO LH
            self.observation_spaces[a] = gymnasium.spaces.Box(low=-1, high=1, shape=(len(self._get_obs_agent(a)),), dtype=numpy.float32) #TODO LH

        pass

    def step(self, actions: dict[str, numpy.float32]):
        _, reward_n, is_terminal_n, is_truncated_n, info_n = self.wrapped_env.step(self._map_actions(actions))

        rewards, terminations, truncations, info = {},{},{},{}
        for agent_id in self.agents:
            rewards[str(agent_id)] = reward_n
            terminations[str(agent_id)] = is_terminal_n
            truncations[str(agent_id)] = is_truncated_n
            info[str(agent_id)] = info_n

        return self._get_obs(), rewards, terminations, truncations, info
    
    def _map_actions(self, actions: dict[str, numpy.float32]):
        'Maps actions back into MuJoCo action space'
        env_actions = np.zeros((self.env.action_space.shape[0],)) + np.nan
        for agent_id, partition in enumerate(self.agent_partitions):
            for i, body_part in enumerate(partition):
                #assert env_actions[body_part.act_ids] != env_actions[body_part.act_ids], "FATAL: At least one env action is doubly defined!"
                env_actions[body_part.act_ids] = actions[str(agent_id)][i]
        
        assert not np.isnan(env_actions).any(), "FATAL: At least one env action is undefined!"
        return env_actions

    def observation_space(self, agent: str):
        return self.observation_spaces[int(str)]

    def action_space(self, agent: str):
        return self.action_spaces[int(str)]
    
    def state(self):
        return self.env.unwrapped._get_obs()

    def _get_obs(self):
        'Returns all agent observations in a dict[str, ActionType]'
        observations = {}
        for agent_id in self.agents:
            observations[str(agent_id)] = self._get_obs_agent(agent_id)
        return observations

    def _get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env.unwrapped._get_obs()
        else:
            return build_obs(self.env,
                                  self.k_dicts[agent_id],
                                  self.k_categories,
                                  self.mujoco_globals,
                                  self.global_categories,
                                  vec_len=getattr(self, "obs_size", None))


    def reset(self, seed=None, return_info=False, options=None):
        """ Returns initial observations and states"""
        self.timelimit_env.reset(seed=seed)
        if return_info == False:
            return self._get_obs()
        else:
            return self._get_obs(), None

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

    def seed(self, seed: int = None):
        raise NotImplementedError