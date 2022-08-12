from __future__ import annotations

import copy
import logging
import random
from typing import List, Optional, TypedDict

from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.mol_tree_node import MolTreeNode, TraceOrder
from rjt_rl.rjt.tree_utils import TRAV_UP, bfs_traverse, dfs_traverse
from rjt_rl.rjt.utils import set_batch_node_id
from rjt_rl.rl.envs.actions import Action2

logger = logging.getLogger(__name__)


class FragDataType(TypedDict):
    state: MolTree
    action: list[int]


def make_step_frags3(
    mol_tree: MolTree,
    samp: Optional[int] = None,
    shuffle: bool = True,
    depth_first: bool = True,
) -> List[FragDataType]:
    """
    Make step dataset for env ver.3
    (root node/traverse order randomization)
    """
    super_root = MolTreeNode()
    super_root.idx = -1
    super_root.nid = -1

    mt = copy.deepcopy(mol_tree)
    set_batch_node_id([mt])
    if shuffle:
        root_node = random.choice(mt.nodes)
    else:
        root_node = mt.nodes[0]

    if depth_first:
        s: TraceOrder = []
        dfs_traverse(s, root_node, super_root)
    else:
        s = bfs_traverse(root_node)

    # clear all connection info
    for node in mt.nodes:
        node.neighbors = []

    mol_tree.dump_tree(logger.debug)

    dataset: list[FragDataType] = []
    for m in s:
        node_x, node_y, direction = m

        if direction == TRAV_UP:
            continue

        nds = copy.deepcopy(node_x.collect_nodes())

        eoe = False
        if len(mt.nodes) == len(nds) + 1:
            eoe = True

        root_node = nds[0]
        if shuffle:
            random.shuffle(nds)
        root_index = nds.index(root_node)

        sub_tree = MolTree.from_nodes(nds[0], nds, set_nid=False)

        sz_x = node_x.size()
        sz_y = node_y.size()

        ac = [-1, -1, -1, -1, -1, -1]

        ac[Action2.TARGET_PRED_ID] = root_index
        ac[Action2.WORD_PRED_ID] = node_y.safe_wid
        s1 = node_x.node_slots[node_y.safe_nid]
        s2 = node_y.node_slots[node_x.safe_nid]
        if sz_x <= 2 and sz_y <= 2:
            ac[Action2.DIR_PRED_ID] = -1
            ac[Action2.SITE1_PRED_ID] = -1
            ac[Action2.SITE2_PRED_ID] = -1
        elif sz_x > 2 and sz_y > 2:
            if type(s1) is tuple:
                assert type(s2) is tuple
                # idir
                ac[Action2.DIR_PRED_ID] = s1[1] + 1
                # site1
                ac[Action2.SITE1_PRED_ID] = s1[0]
                # site2
                ac[Action2.SITE2_PRED_ID] = s2[0]
            else:
                assert type(s1) is int
                assert type(s2) is int
                # idir (spiro atom)
                ac[Action2.DIR_PRED_ID] = 0 + 1
                # site1
                ac[Action2.SITE1_PRED_ID] = s1
                # site2
                ac[Action2.SITE2_PRED_ID] = s2
        elif sz_x <= 2:
            assert type(s2) is int
            # sz_y > 2
            # idir
            ac[Action2.DIR_PRED_ID] = -1
            # site1
            ac[Action2.SITE1_PRED_ID] = -1
            # site2
            ac[Action2.SITE2_PRED_ID] = s2
        elif sz_y <= 2:
            assert type(s1) is int
            # sz_x > 2
            # idir
            ac[Action2.DIR_PRED_ID] = -1
            # site1
            ac[Action2.SITE1_PRED_ID] = s1
            # site2
            ac[Action2.SITE2_PRED_ID] = -1

        if eoe:
            # stop action
            ac[Action2.STOP_PRED_ID] = 1
        else:
            # proceed action
            ac[Action2.STOP_PRED_ID] = 0

        dataset.append({"state": sub_tree, "action": ac})

        if eoe:
            break

        # proceed to the next node
        if node_x not in node_y.neighbors:
            node_y.neighbors.append(node_x)
        if node_y not in node_x.neighbors:
            node_x.neighbors.append(node_y)

    logger.debug(f"dataset: {dataset}")

    if type(samp) is int and samp > 0:
        sel_dat = random.sample(dataset[:-1], samp)
        # always include EOE step
        sel_dat.append(dataset[-1])
        dataset = sel_dat

    return dataset
