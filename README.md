This project has been integrated into the [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) repo.

# This is a fork of the orignal [MaMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)


```diff
+ Now ALL the environments work
+ Now uses the standard ‘PettingZoo.ParallelEnv‘ API.
+ Now Uses a modern version of ‘Gymnasium‘ (10.8v → 26.3v).
+ Now Uses newer MuJoCo bindings (‘mujoco-py‘ → ‘mujoco‘).
+ Now Uses newer `Gymansium.MuJoco` Environments (v2 -> v4).
+ Now includes new mapping functions (RL -> MARL, MARL -> OTHER MARL).
+ Now Has a mechanism for allowing researchers to easily create new agent factorizations.
+ Now also supports `Gymansium.MuJoCo.Pusher`.
+ Now supports observing global state.
+ Cleaned up the code base significantly.
+ Have written unit-tests and fixed a TON of bugs.
+ Have written some Documentation (There was virtually None).
+ Fixed some agent factorizations not having global observations.
- Is no longer backwards compatible with the old factorizations nor are the returns comparable.
```

# Multi-Agent Mujoco
Benchmark for Continuous Multi-Agent Robotic Control, based on Farama Foundation's Mujoco Gymnasium environments.

<img src="https://github.com/schroederdewitt/multiagent_mujoco/blob/master/docs/images/mamujoco.jpg" width="900" height="384">

Described the paper [Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2003.06709) by Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Böhmer and Shimon Whiteson, Torr Vision Group and Whiteson Research Lab, University of Oxford, 2020

# Installation

```
git clone https://github.com/Kallinteris-Andreas/multiagent_mujoco_v4.git
cd multiagent_mujoco_v4 
pip install .
```

# Example

```python
import numpy
from multiagent_mujoco import mamujoco_v0

if __name__ == "__main__":
    env = mamujoco_v0.parallel_env(scenario='Ant', agent_conf='2x4', agent_obsk=0, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Humanoid', agent_conf='9|8', agent_obsk=0, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Reacher', agent_conf='2x1', agent_obsk=1, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='coupled_half_cheetah', agent_conf='1p1', agent_obsk=1, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='manyagent_swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='coupled_half_cheetah', agent_conf='1p1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='manyagent_swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')
    
    n_episodes = 1
    debug_step = 0

    for e in range(n_episodes):
        obs = env.reset()
        terminated = {'0': False}
        truncated = {'0': False}
        episode_reward = 0

        while not terminated['0'] and not truncated['0']:
            state = env.state()

            actions = {}
            for agent_id in env.agents:
                avail_actions = env.action_space(agent_id)
                action = numpy.random.uniform(avail_actions.low[0], avail_actions.high[0], avail_actions.shape[0])
                actions[str(agent_id)] = action

            obs, reward, terminated, truncated, info = env.step(actions)

            episode_reward += reward['0']

        print("Total reward in episode {} = {}".format(e, episode_reward))
    env.close()
```

# Documentation

The DOC pages, have been moved to https://robotics.farama.org/envs/MaMuJoCo/
