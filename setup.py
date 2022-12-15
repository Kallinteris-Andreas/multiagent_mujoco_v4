from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """MAMuJoCo - Multi-Agent MuJoCo

Benchmark for Continuous Multi-Agent Robotic Control, based on OpenAI's Mujoco Gym environments.
Described in the paper Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control 
by Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Böhmer and Shimon Whiteson, 
Torr Vision Group and Whiteson Research Lab, University of Oxford, 2020

Contact CSDW at cs@robots.ox.ac.uk
"""

extras_deps = {
      "dev": [
            "pre-commit>=2.0.1",
            "black>=19.10b0",
            "flake8>=3.7",
            "flake8-bugbear>=20.1",
      ],
}

setup(
      name="MAMujoco",
      version="1.4.0",
      description="MAMuJoCo - Multi-Agent MuJoCo.",
      long_description=description,
      author="Kallinteris Andreas",
      author_email="kallinteris@proton.me",
      license="Apache 2.0 License",
      keywords="Robotics, MuJoCo, Multi-Agent Reinforcement Learning, PettingZoo",
      url="https://github.com/Kallinteris-Andreas/multiagent_mujoco_v4",
      packages=[
            "multiagent_mujoco",
            "multiagent_mujoco.assets",
      ],
      package_data={'multiagent_mujoco.assets': ['*.xml*']},
      extras_require=extras_deps,
      install_requires=[
            "gymnasium==0.27.0",
            "mujoco>=2.2.0",
            "PettingZoo==1.22.2",
            "Jinja2>=3.0.3",
            #"scipy>=1.9.0",
            #"numpy>=1.24.5",
            #"glfw>=2.5.1",
            #"Cython>=0.29.28"
      ],
)
