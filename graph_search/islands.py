"""Compute the number of islands for a given graph"""

import networkx as nx


def islands(graph: nx.Graph):
    # while True:
    n_islands = int(0)
    virgin = set([*graph.nodes])
    visited = set()
    while virgin:
        frontier = set([virgin.pop()])
        while frontier:
            n = frontier.pop()
            ns = list(graph.neighbors(n))
            frontier.update(set(ns) - visited)
            visited.update(ns)
        virgin -= visited
        n_islands += 1
    return n_islands


if __name__ == '__main__':
    G = nx.grid_graph(dim=[10, 10])
    n = islands(G)
    assert n == 1
