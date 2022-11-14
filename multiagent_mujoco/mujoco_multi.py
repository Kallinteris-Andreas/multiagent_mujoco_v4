import gymnasium
import pettingzoo
import numpy

from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs
#from obsk import get_joints_at_kdist, get_parts_and_edges, build_obs


class MujocoMulti(pettingzoo.utils.env.ParallelEnv):
    def __init__(self, scenario: str, agent_conf: str, agent_obsk: int, render_mode: str=None):
        scenario += '-v4'

        self.agent_action_partitions, mujoco_edges, self.mujoco_globals = get_parts_and_edges(scenario, agent_conf)

        self.possible_agents = [str(agent_id) for agent_id in range(len(self.agent_action_partitions))]
        self.agents = self.possible_agents

        self.agent_obsk = agent_obsk # if None, fully observable else k>=0 implies observe nearest k agents or joints

        if self.agent_obsk is not None:
            if scenario in ["Ant-v4", "manyagent_ant"]:
                k_categories_label = "qpos,qvel,cfrc_ext|qpos"
            elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
                k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
            elif scenario in ["Reacher-v4"]:
                k_categories_label = "qpos,qvel,fingertip_dist|qpos"
            elif scenario in ["coupled_half_cheetah"]:
                k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
            else:
                k_categories_label = "qpos,qvel|qpos"

            k_split = k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk+1)]

            self.global_categories = []


        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id,
                                                self.agent_action_partitions,
                                                mujoco_edges,
                                                k=self.agent_obsk,
                                                kagents=False,) for agent_id in range(self.num_agents)]

        # load scenario from script
        try:
            self.env = (gymnasium.make(scenario, render_mode=render_mode))
        except gymnasium.error.Error:  # env not in gymnasium
            #assert False, 'Non-Gymnasium Enviroments have not been implamented'
            if scenario in ["manyagent_ant-v4"]:
                from .manyagent_ant import ManyAgentAntEnv as this_env
            elif scenario in ["manyagent_swimmer-v4"]:
                from .manyagent_swimmer import ManyAgentSwimmerEnv as this_env
            elif scenario in ["coupled_half_cheetah-v4"]:
                from .coupled_half_cheetah import CoupledHalfCheetah as this_env
            else:
                raise NotImplementedError('Custom env not implemented!')
            self.env = gymnasium.wrappers.TimeLimit(this_env(agent_conf), max_episode_steps=1000)#TODO add compatability

        self.observation_spaces, self.action_spaces = {}, {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            self.action_spaces[agent_id] = gymnasium.spaces.Box(low=-1, high=1, shape=(len(partition),), dtype=numpy.float32) #TODO LH
            self.observation_spaces[agent_id] = gymnasium.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(len(self._get_obs_agent(agent_id)),), dtype=numpy.float32) #TODO LH

        pass

    def step(self, actions: dict[str, numpy.float32]):
        _, reward_n, is_terminal_n, is_truncated_n, info_n = self.env.step(self.map_actions(actions))

        rewards, terminations, truncations, info = {},{},{},{}
        observations = self._get_obs()
        for agent_id in self.agents:
            rewards[str(agent_id)] = reward_n
            terminations[str(agent_id)] = is_terminal_n
            truncations[str(agent_id)] = is_truncated_n
            info[str(agent_id)] = info_n

        if is_terminal_n or is_truncated_n:
            self.agents = []

        return observations, rewards, terminations, truncations, info
    
    def map_actions(self, actions: dict[str, numpy.float32]):
        'Maps actions back into MuJoCo action space'
        env_actions = numpy.zeros((self.env.action_space.shape[0],)) + numpy.nan
        for agent_id, partition in enumerate(self.agent_action_partitions):
            for i, body_part in enumerate(partition):
                assert numpy.isnan(env_actions[body_part.act_ids]), "FATAL: At least one env action is doubly defined!"
                env_actions[body_part.act_ids] = actions[str(agent_id)][i]
        
        assert not numpy.isnan(env_actions).any(), "FATAL: At least one env action is undefined!"
        return env_actions

    def observation_space(self, agent: str):
        return self.observation_spaces[int(agent)]

    def action_space(self, agent: str):
        return self.action_spaces[int(agent)]
    
    def state(self):
        return self.env.unwrapped._get_obs()

    def _get_obs(self):
        'Returns all agent observations in a dict[str, ActionType]'
        observations = {}
        for agent_id in self.agents:
            observations[str(agent_id)] = self._get_obs_agent(int(agent_id))
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
        self.env.reset(seed=seed)
        self.agents = self.possible_agents
        if return_info == False:
            return self._get_obs()
        else:
            return self._get_obs(), None

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: int = None):
        raise NotImplementedError