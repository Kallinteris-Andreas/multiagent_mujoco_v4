from pettingzoo.test import parallel_api_test  # noqa: E402
from multiagent_mujoco.mujoco_multi import MujocoMulti

if __name__ == "__main__":
    for ok in [None, 0, 1]:
        scenario="Ant"
        agent_conf="2x4"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Ant"
        agent_conf="2x4d"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Ant"
        agent_conf="4x2"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="HalfCheetah"
        agent_conf="2x3"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="HalfCheetah"
        agent_conf="6x1"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Hopper"
        agent_conf="3x1"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Humanoid"
        agent_conf="9|8"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="HumanoidStandup"
        agent_conf="9|8"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Reacher"
        agent_conf="2x1"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Swimmer"
        agent_conf="2x1"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        scenario="Walker2d"
        agent_conf="2x3"
        parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        #disabled since they do not on the original MaMuJoCo
        #scenario="manyagent_swimmer"
        #agent_conf="10x2"
        #parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        #disabled since they do not on the original MaMuJoCo
        #scenario="manyagent_ant"
        #agent_conf="2x3"
        #parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)

        #disabled since they do not on the original MaMuJoCo
        #scenario="coupled_half_cheetah"
        #agent_conf="1p1"
        #parallel_api_test(MujocoMulti(scenario=scenario, agent_conf=agent_conf, agent_obsk=ok), num_cycles=1_000_000)