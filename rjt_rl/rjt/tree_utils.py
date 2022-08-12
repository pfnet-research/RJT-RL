from __future__ import annotations

from collections import deque

from .mol_tree_node import MolTreeNode, NodeIDFunc, PropOrder, TraceOrder

TRAV_DOWN = 1
TRAV_UP = 0


def dfs_traverse(
    stack: TraceOrder,
    x: MolTreeNode,
    fa: MolTreeNode,
    id_fn: NodeIDFunc = lambda x: x.safe_nid,
) -> None:
    for y in x.neighbors:
        if id_fn(y) == id_fn(fa):
            continue
        stack.append((x, y, TRAV_DOWN))
        dfs_traverse(stack, y, x, id_fn)
        stack.append((y, x, TRAV_UP))


def calc_prop_order2(
    root_node: MolTreeNode,
    id_fn: NodeIDFunc = lambda x: x.safe_nid,
) -> tuple[PropOrder, PropOrder]:
    queue = deque([root_node])
    visited = set([id_fn(root_node)])
    root_node.depth = 0
    order1: PropOrder = []
    order2: PropOrder = []
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if id_fn(y) not in visited:
                queue.append(y)
                visited.add(id_fn(y))
                assert x.depth is not None
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth - 1].append((x, y))
                order2[y.depth - 1].append((y, x))
    return order1, order2


def bfs_traverse(node: MolTreeNode) -> TraceOrder:
    order, _ = calc_prop_order2(node)
    order1 = [j for i in order for j in i]
    s = [(x, y, TRAV_DOWN) for (x, y) in order1]
    return s
