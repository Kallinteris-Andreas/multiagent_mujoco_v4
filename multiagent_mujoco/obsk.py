import itertools
from copy import deepcopy

import numpy as np
import numpy


class Node:
    def __init__(
        self,
        label,
        qpos_ids,  # Note: as far as i can tell this does nothing
        qvel_ids,  # Note: as far as i can tell this does nothing
        act_ids,
        body_fn=None,
        bodies=None,
        extra_obs=None,
        tendons=None,
    ):
        """
        A node of the mujoco graph for single action, and it's corrisponding observetions
        """
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = {} if extra_obs is None else extra_obs
        self.body_fn = body_fn
        self.tendons = tendons
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class HyperEdge:
    def __init__(self, *edges: Node):
        self.edges = set(edges)

    def __contains__(self, item):
        return item in self.edges

    def __str__(self):
        return "HyperEdge({})".format(self.edges)

    def __repr__(self):
        return "HyperEdge({})".format(self.edges)


def get_joints_at_kdist(
    agent_partition: list[tuple[Node, ...]],
    hyperedges: list[HyperEdge],
    k: int = 0,
) -> dict[int : list[Node]]:
    """Identify all joints at distance <= k from agent agent_id

    :param agent_partition: tuples of nodes of an agent
    :param hyperedges: hyperedges of the graph
    :param k: kth degree (number of nearest joints to observe)
    :return:
        dict with k as key, and list of joints/nodes at that distance
    """

    if k is None:
        return None

    def _adjacent(lst):
        # return all sets adjacent to any element in lst
        ret = set()
        for element in lst:
            ret = ret.union(
                set(
                    itertools.chain(
                        *[
                            e.edges.difference({element})
                            for e in hyperedges
                            if element in e
                        ]
                    )
                )
            )
        return ret

    explored_nodes = set(agent_partition)
    new_nodes = explored_nodes
    k_dict = {0: sorted(list(new_nodes), key=lambda x: x.label)}
    for key in range(1, k + 1):
        new_nodes = _adjacent(new_nodes) - explored_nodes
        explored_nodes = explored_nodes.union(new_nodes)
        k_dict[key] = sorted(list(new_nodes), key=lambda x: x.label)
    # TODO assert that the nodes are mutally exclusive
    return k_dict


def build_obs(
    env, k_dict, k_categories, global_dict, global_categories
) -> numpy.ndarray:
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param k_dict: k_dict
    :param qpos: qpos numpy array
    :param qvel: qvel numpy array
    :return:
    observation vector
    """

    # TODO: This needs to be fixed, it was designed for half-cheetah only!
    # if add_global_pos:
    #    obs_qpos_lst.append(global_qpos)
    #    obs_qvel_lst.append(global_qvel)

    body_set_dict = {}
    obs_lst = []
    # Add parts attributes
    for k in sorted(list(k_dict.keys())):
        for node in k_dict[k]:
            for category in k_categories[k]:
                if category in node.extra_obs:
                    items = node.extra_obs[category](env).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                else:
                    if category in [
                        "qvel",
                        "qpos",
                    ]:  # this is a "joint position/velocity" item
                        items = getattr(env.unwrapped.data, category)[
                            getattr(node, "{}_ids".format(category))
                        ]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif category in ["qfrc_actuator"]:  # this is a "vel position" item
                        items = getattr(env.unwrapped.data, category)[
                            getattr(node, "{}_ids".format("qvel"))
                        ]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif category in [
                        "cvel",
                        "cinert",
                        "cfrc_ext",
                    ]:  # this is a "body position" item
                        if node.bodies is not None:
                            for b in node.bodies:
                                if category not in body_set_dict:
                                    body_set_dict[category] = set()
                                if b not in body_set_dict[category]:
                                    items = getattr(env.unwrapped.data, category)[
                                        b
                                    ].tolist()
                                    items = getattr(node, "body_fn", lambda _id, x: x)(
                                        b, items
                                    )
                                    obs_lst.extend(
                                        items if isinstance(items, list) else [items]
                                    )
                                    body_set_dict[category].add(b)

    # Add global attributes
    body_set_dict = {}
    for category in global_categories:
        if category in ["qvel", "qpos"]:  # this is a "joint position" item
            for j in global_dict.get("joints", []):
                items = getattr(env.unwrapped.data, category)[
                    getattr(j, "{}_ids".format(category))
                ]
                obs_lst.extend(items if isinstance(items, list) else [items])
        else:
            for b in global_dict.get("bodies", []):
                if category not in body_set_dict:
                    body_set_dict[category] = set()
                if b not in body_set_dict[category]:
                    obs_lst.extend(getattr(env.unwrapped.data, category)[b].tolist())
                    body_set_dict[category].add(b)

    return np.array(obs_lst)


def get_parts_and_edges(
    label: str, partitioning: str
) -> tuple[list[tuple[Node, ...]], list[HyperEdge], dict[str : list[Node]]]:
    """
    :param label: the mujoco task to partion
    :param partitioning: the partioneing scheme
    :return:
        the partition of the mujoco graph nodes, the graph edges, and global nodes
    """
    if label in ["HalfCheetah-v4"]:

        # define Mujoco graph
        bthigh = Node("bthigh", -6, -6, 0)
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        edges = [
            HyperEdge(bfoot, bshin),
            HyperEdge(bshin, bthigh),
            HyperEdge(bthigh, fthigh),
            HyperEdge(fthigh, fshin),
            HyperEdge(fshin, ffoot),
        ]

        root_x = Node("root_x", 0, 0, -1, extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 1, 1, -1)
        root_y = Node("root_y", 2, 2, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "2x3":
            parts = [(bfoot, bshin, bthigh), (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Ant-v4"]:

        # define Mujoco graph
        torso = 1
        front_left_leg = 2
        aux_1 = 3
        ankle_1 = 4
        front_right_leg = 5
        aux_2 = 6
        ankle_2 = 7
        back_leg = 8
        aux_3 = 9
        ankle_3 = 10
        right_back_leg = 11
        aux_4 = 12
        ankle_4 = 13

        hip1 = Node(
            "hip1",
            -8,
            -8,
            2,
            bodies=[torso, front_left_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  #
        ankle1 = Node(
            "ankle1",
            -7,
            -7,
            3,
            bodies=[front_left_leg, aux_1, ankle_1],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        hip2 = Node(
            "hip2",
            -6,
            -6,
            4,
            bodies=[torso, front_right_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        ankle2 = Node(
            "ankle2",
            -5,
            -5,
            5,
            bodies=[front_right_leg, aux_2, ankle_2],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        hip3 = Node(
            "hip3",
            -4,
            -4,
            6,
            bodies=[torso, back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        ankle3 = Node(
            "ankle3",
            -3,
            -3,
            7,
            bodies=[back_leg, aux_3, ankle_3],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        hip4 = Node(
            "hip4",
            -2,
            -2,
            0,
            bodies=[torso, right_back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        ankle4 = Node(
            "ankle4",
            -1,
            -1,
            1,
            bodies=[right_back_leg, aux_4, ankle_4],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,

        edges = [
            HyperEdge(ankle4, hip4),
            HyperEdge(ankle1, hip1),
            HyperEdge(ankle2, hip2),
            HyperEdge(ankle3, hip3),
            HyperEdge(hip4, hip1, hip2, hip3),
        ]

        free_joint = Node(
            "free",
            0,
            0,
            -1,
            extra_obs={
                "qpos": lambda env: env.unwrapped.data.qpos[:7],
                "qvel": lambda env: env.unwrapped.data.qvel[:6],
                "cfrc_ext": lambda env: np.clip(
                    env.unwrapped.data.cfrc_ext[0:1], -1, 1
                ),
            },
        )
        globals = {"joints": [free_joint]}

        if partitioning == "2x4":  # neighbouring legs together
            parts = [(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d":  # diagonal legs together
            parts = [(hip1, ankle1, hip3, ankle3), (hip2, ankle2, hip4, ankle4)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Hopper-v4"]:

        # define Mujoco-Graph
        thigh_joint = Node(
            "thigh_joint",
            -3,
            -3,
            0,
            extra_obs={
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[-3]]), -10, 10
                )
            },
        )
        leg_joint = Node(
            "leg_joint",
            -2,
            -2,
            1,
            extra_obs={
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[-2]]), -10, 10
                )
            },
        )
        foot_joint = Node(
            "foot_joint",
            -1,
            -1,
            2,
            extra_obs={
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[-1]]), -10, 10
                )
            },
        )

        edges = [HyperEdge(foot_joint, leg_joint), HyperEdge(leg_joint, thigh_joint)]

        root_x = Node(
            "root_x",
            0,
            0,
            -1,
            extra_obs={
                "qpos": lambda env: np.array([]),
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[1]]), -10, 10
                ),
            },
        )
        root_z = Node(
            "root_z",
            1,
            1,
            -1,
            extra_obs={
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[1]]), -10, 10
                )
            },
        )
        root_y = Node(
            "root_y",
            2,
            2,
            -1,
            extra_obs={
                "qvel": lambda env: np.clip(
                    np.array([env.unwrapped.data.qvel[2]]), -10, 10
                )
            },
        )
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "3x1":
            parts = [(thigh_joint,), (leg_joint,), (foot_joint,)]

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Humanoid-v4", "HumanoidStandup-v4"]:

        # define Mujoco-Graph
        abdomen_y = Node(
            "abdomen_y", -16, -16, 0
        )  # act ordering bug in env -- double check!
        abdomen_z = Node("abdomen_z", -17, -17, 1)
        abdomen_x = Node("abdomen_x", -15, -15, 2)
        right_hip_x = Node("right_hip_x", -14, -14, 3)
        right_hip_z = Node("right_hip_z", -13, -13, 4)
        right_hip_y = Node("right_hip_y", -12, -12, 5)
        right_knee = Node("right_knee", -11, -11, 6)
        left_hip_x = Node("left_hip_x", -10, -10, 7)
        left_hip_z = Node("left_hip_z", -9, -9, 8)
        left_hip_y = Node("left_hip_y", -8, -8, 9)
        left_knee = Node("left_knee", -7, -7, 10)
        right_shoulder1 = Node("right_shoulder1", -6, -6, 11)
        right_shoulder2 = Node("right_shoulder2", -5, -5, 12)
        right_elbow = Node("right_elbow", -4, -4, 13)
        left_shoulder1 = Node("left_shoulder1", -3, -3, 14)
        left_shoulder2 = Node("left_shoulder2", -2, -2, 15)
        left_elbow = Node("left_elbow", -1, -1, 16)

        edges = [
            HyperEdge(abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(left_elbow, left_shoulder1, left_shoulder2),
            HyperEdge(right_elbow, right_shoulder1, right_shoulder2),
            HyperEdge(left_knee, left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(right_knee, right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(
                right_shoulder1, right_shoulder2, abdomen_x, abdomen_y, abdomen_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, left_hip_x, left_hip_y, left_hip_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z
            ),
        ]

        globals = {}

        if (
            partitioning == "9|8"
        ):  # 17 in total, so one action is a dummy (to be handled by pymarl)
            # isolate upper and lower body
            parts = [
                (
                    left_shoulder1,
                    left_shoulder2,
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_elbow,
                ),
                (
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_knee,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Reacher-v4"]:

        # define Mujoco-Graph
        body0 = 1
        body1 = 2
        fingertip = 3
        joint0 = Node(
            "joint0",
            -4,
            -4,
            0,
            bodies=[body0, body1],
            extra_obs={
                "qpos": (
                    lambda env: np.array(
                        [
                            np.sin(env.unwrapped.data.qpos[-4]),
                            np.cos(env.unwrapped.data.qpos[-4]),
                        ]
                    )
                )
            },
        )
        joint1 = Node(
            "joint1",
            -3,
            -3,
            1,
            bodies=[body1, fingertip],
            extra_obs={
                "fingertip_dist": (
                    lambda env: env.get_body_com("fingertip")
                    - env.get_body_com("target")
                ),
                "qpos": (
                    lambda env: np.array(
                        [
                            np.sin(env.unwrapped.data.qpos[-3]),
                            np.cos(env.unwrapped.data.qpos[-3]),
                        ]
                    )
                ),
            },
        )
        edges = [HyperEdge(joint0, joint1)]

        worldbody = 0
        target = 4
        target_x = Node(
            "target_x", -2, -2, -1, extra_obs={"qvel": (lambda env: np.array([]))}
        )
        target_y = Node(
            "target_y", -1, -1, -1, extra_obs={"qvel": (lambda env: np.array([]))}
        )
        globals = {"bodies": [worldbody, target], "joints": [target_x, target_y]}

        if partitioning == "2x1":
            # isolate upper and lower arms
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Swimmer-v4"]:

        # define Mujoco-Graph
        joint0 = Node("rot2", -2, -2, 0)  # TODO: double-check ids
        joint1 = Node("rot3", -1, -1, 1)

        edges = [HyperEdge(joint0, joint1)]
        globals = {}

        if partitioning == "2x1":
            # isolate upper and lower body
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Walker2d-v4"]:

        # define Mujoco-Graph
        thigh_joint = Node("thigh_joint", -6, -6, 0)
        leg_joint = Node("leg_joint", -5, -5, 1)
        foot_joint = Node("foot_joint", -4, -4, 2)
        thigh_left_joint = Node("thigh_left_joint", -3, -3, 3)
        leg_left_joint = Node("leg_left_joint", -2, -2, 4)
        foot_left_joint = Node("foot_left_joint", -1, -1, 5)

        edges = [
            HyperEdge(foot_joint, leg_joint),
            HyperEdge(leg_joint, thigh_joint),
            HyperEdge(foot_left_joint, leg_left_joint),
            HyperEdge(leg_left_joint, thigh_left_joint),
            HyperEdge(thigh_joint, thigh_left_joint),
        ]
        globals = {}

        if partitioning == "2x3":
            # isolate upper and lower body
            parts = [
                (foot_joint, leg_joint, thigh_joint),
                (
                    foot_left_joint,
                    leg_left_joint,
                    thigh_left_joint,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["coupled_half_cheetah-v4"]:

        # define Mujoco graph
        tendon = 0

        bthigh = Node(
            "bthigh",
            -6,
            -6,
            0,
            tendons=[tendon],
            extra_obs={
                "ten_J": lambda env: env.unwrapped.data.ten_J[tendon],
                "ten_length": lambda env: env.unwrapped.data.ten_length,
                "ten_velocity": lambda env: env.unwrapped.data.ten_velocity,
            },
        )
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        bthigh2 = Node(
            "bthigh2",
            -6,
            -6,
            6,
            tendons=[tendon],
            extra_obs={
                "ten_J": lambda env: env.unwrapped.data.ten_J[tendon],
                "ten_length": lambda env: env.unwrapped.data.ten_length,
                "ten_velocity": lambda env: env.unwrapped.data.ten_velocity,
            },
        )
        bshin2 = Node("bshin2", -5, -5, 7)
        bfoot2 = Node("bfoot2", -4, -4, 8)
        fthigh2 = Node("fthigh2", -3, -3, 9)
        fshin2 = Node("fshin2", -2, -2, 10)
        ffoot2 = Node("ffoot2", -1, -1, 11)

        edges = [
            HyperEdge(bfoot, bshin),
            HyperEdge(bshin, bthigh),
            HyperEdge(bthigh, fthigh),
            HyperEdge(fthigh, fshin),
            HyperEdge(fshin, ffoot),
            HyperEdge(bfoot2, bshin2),
            HyperEdge(bshin2, bthigh2),
            HyperEdge(bthigh2, fthigh2),
            HyperEdge(fthigh2, fshin2),
            HyperEdge(fshin2, ffoot2),
        ]
        globals = {}

        root_x = Node("root_x", 0, 0, -1, extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 1, 1, -1)
        root_y = Node("root_y", 2, 2, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "1p1":
            parts = [
                (bfoot, bshin, bthigh, ffoot, fshin, fthigh),
                (bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2),
            ]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["manyagent_swimmer-v4"]:

        # Generate asset file
        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        # Note: Default Swimmer corresponds to n_segs = 3

        # define Mujoco-Graph
        joints = [
            Node("rot{:d}".format(i), -n_segs + i, -n_segs + i, i)
            for i in range(0, n_segs)
        ]
        edges = [HyperEdge(joints[i], joints[i + 1]) for i in range(n_segs - 1)]
        globals = {}

        parts = [
            tuple(joints[i * n_segs_per_agents : (i + 1) * n_segs_per_agents])
            for i in range(n_agents)
        ]
        return parts, edges, globals

    elif label in ["manyagent_ant-v4"]:  # TODO: FIX!

        # Generate asset file
        try:
            n_agents = int(partitioning.split("x")[0])
            n_segs_per_agents = int(partitioning.split("x")[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        # # define Mujoco graph
        # torso = 1
        # front_left_leg = 2
        # aux_1 = 3
        # ankle_1 = 4
        # right_back_leg = 11
        # aux_4 = 12
        # ankle_4 = 13
        #
        # off = -4*(n_segs-1)
        # hip1 = Node("hip1", -4-off, -4-off, 2, bodies=[torso, front_left_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist()) #
        # ankle1 = Node("ankle1", -3-off, -3-off, 3, bodies=[front_left_leg, aux_1, ankle_1], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        # hip4 = Node("hip4", -2-off, -2-off, 0, bodies=[torso, right_back_leg], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        # ankle4 = Node("ankle4", -1-off, -1-off, 1, bodies=[right_back_leg, aux_4, ankle_4], body_fn=lambda _id, x:np.clip(x, -1, 1).tolist())#,
        #
        # edges = [HyperEdge(ankle4, hip4),
        #          HyperEdge(ankle1, hip1),
        #          HyperEdge(hip4, hip1),
        #          ]

        edges = []
        joints = []
        for si in range(n_segs):

            torso = 1 + si * 7
            front_right_leg = 2 + si * 7
            aux1 = 3 + si * 7
            ankle1 = 4 + si * 7
            back_leg = 5 + si * 7
            aux2 = 6 + si * 7
            ankle2 = 7 + si * 7

            off = -4 * (n_segs - 1 - si)
            hip1n = Node(
                "hip1_{:d}".format(si),
                -4 - off,
                -4 - off,
                2 + 4 * si,
                bodies=[torso, front_right_leg],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle1n = Node(
                "ankle1_{:d}".format(si),
                -3 - off,
                -3 - off,
                3 + 4 * si,
                bodies=[front_right_leg, aux1, ankle1],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            hip2n = Node(
                "hip2_{:d}".format(si),
                -2 - off,
                -2 - off,
                0 + 4 * si,
                bodies=[torso, back_leg],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )
            ankle2n = Node(
                "ankle2_{:d}".format(si),
                -1 - off,
                -1 - off,
                1 + 4 * si,
                bodies=[back_leg, aux2, ankle2],
                body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
            )

            edges += [
                HyperEdge(ankle1n, hip1n),
                HyperEdge(ankle2n, hip2n),
                HyperEdge(hip1n, hip2n),
            ]
            if si:
                edges += [HyperEdge(hip1m, hip2m, hip1n, hip2n)]

            hip1m = deepcopy(hip1n)
            hip2m = deepcopy(hip2n)
            joints.append([hip1n, ankle1n, hip2n, ankle2n])

        free_joint = Node(
            "free",
            0,
            0,
            -1,
            extra_obs={
                "qpos": lambda env: env.unwrapped.data.qpos[:7],
                "qvel": lambda env: env.unwrapped.data.qvel[:6],
                "cfrc_ext": lambda env: np.clip(
                    env.unwrapped.data.cfrc_ext[0:1], -1, 1
                ),
            },
        )
        globals = {"joints": [free_joint]}

        parts = [
            [
                x
                for sublist in joints[
                    i * n_segs_per_agents : (i + 1) * n_segs_per_agents
                ]
                for x in sublist
            ]
            for i in range(n_agents)
        ]

        return parts, edges, globals
    else:
        if partitioning is None:
            print("Warning: using single agent on unknown MuJoCo Enviroment: " + label)
            return tuple([tuple("0")]), None, None
        raise Exception("UNKNOWN label enviroment: {}".format(label))
