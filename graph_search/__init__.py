"""
Comparison between various graph-search algorithms.

reference: https://www.redblobgames.com/pathfinding/a-star/introduction.html#dijkstra
ref_2: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
"""
from collections import deque, defaultdict
import networkx as nx
import numpy as np


def backtrack(link, start, goal):
    curr = goal
    inds, ds = [], []
    while curr in link:
        prev, d = link[curr]
        inds.append(curr)
        ds.append(d)
        if prev == start:
            inds.append(prev)
            return inds[::-1], ds[::-1]
        curr = prev
    return None, None


def bfs(graph: nx.Graph, start, goal=None, *_):
    """Breath-first Search

    :param graph: a graph object
    :param start: the starting node
    :param goal: optional, when not supplied generates the entire path tree.
    :return:
    """
    frontier = deque()
    tree = dict()

    frontier.append(start)
    while len(frontier):
        current = frontier.popleft()
        for n in graph.neighbors(current):
            if n in tree:
                continue
            # only used to compute path length
            edge_len = graph.edges[current, n].get('weight', 1)
            tree[n] = current, edge_len
            if goal == n:
                return backtrack(tree, start, goal)
            frontier.append(n)
    return None, None


from heapq import heapify, heappush, heappop


class PriorityQue:
    def __init__(self):
        self.queue = []

    def push(self, priority, *args):
        heappush(self.queue, (priority, *args))

    def pop(self):
        return heappop(self.queue)

    def __len__(self):
        return len(self.queue)


def heuristic_search(graph: nx.Graph, start, goal, heuristic):
    """Heuristic Search

    :param graph:
    :param start:
    :param goal:
    :param heuristic:
    :return:
    """
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, start)
    while len(frontier):
        _, current = frontier.pop()
        ns = [*graph.neighbors(current)]
        to_goals = heuristic(ns, [goal])
        for ind, n in enumerate(ns):
            if n in visited:
                continue
            # only used to compute path length
            edge_len = graph.edges[current, n].get('weight', 1)
            to_goal = to_goals[ind]
            visited[n] = current, edge_len
            if goal == n:
                return backtrack(visited, start, goal)
            frontier.push(to_goal, n)
    return None, None


# note: the termination condition in dikjstra's can be a function.
#       However, I decided against doing this, because adding the
#       goal node to the graph and only use the identity for termination
#       has the benefit that we can use the same neighborhood
#       relationship to define the last leg, as opposed to using
#       1-step lookahead, which overlaps with the neighbor function.
#
# note-2: nice hack: we can overload the __eq__ meta method of the goal
#         object, to return True if the goal n falls within a set of nodes
#         on the graph. In this case we just return a proxy object as the goal.
#         To this end, we revert the equal sign.
#         - Ge :)
def dijkstra(graph: nx.Graph, start, goal, *_):
    """Dijkstra's

    :param graph:
    :param start:
    :param goal:
    :return:
    """
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, start)
    while len(frontier):
        sofar, current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            edge_len = graph.edges[current, n].get('weight', 1)
            visited[n] = current, edge_len
            if goal == n:
                return backtrack(visited, start, goal)
            frontier.push(sofar + edge_len, n)
    return None, None


def a_star(graph: nx.Graph, start, goal, heuristic):
    """A*

    :param graph:
    :param start:
    :param goal:
    :param heuristic:
    :return:
    """
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, 0, start)
    while len(frontier):
        _, sofar, current = frontier.pop()
        ns = [*graph.neighbors(current)]
        to_goals = heuristic(ns, [goal])
        for ind, n in enumerate(ns):
            if n in visited:
                continue
            edge_len = graph.edges[current, n].get('weight', 1)
            to_goal = to_goals[ind]
            visited[n] = current, edge_len
            if goal == n:
                return backtrack(visited, start, goal)
            frontier.push(sofar + edge_len + to_goal, sofar + edge_len, n)
    return None, None


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        plt.arrow(x, y, (x_ - x) * 0.9, (y_ - y) * 0.8, **kwargs, head_width=0.2, head_length=0.2,
                  length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)

    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)
    plt.gca().set_aspect('equal')


def patch_graph(G):
    queries = defaultdict(lambda: 0)
    _neighbors = G.neighbors

    def neighbors(n):
        # queries[n] += 1  # no global needed bc mutable.
        ns = list(_neighbors(n))
        for n in ns:
            queries[n] += 1
        return ns

    G.neighbors = neighbors
    return queries


# export all of the methods here.
methods = {"bfs": bfs, "heuristic": heuristic_search,
           "dijkstra": dijkstra, "a_star": a_star}

short_names = {"bfs": "bfs", "heuristic": "heuristic",
               "dijkstra": "Dijkstra's", "a_star": "A*"}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from collections import defaultdict


    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=2, axis=-1) * 5


    n, start, goal = 7, (0, 0), (6, 6)

    G = nx.grid_graph(dim=[n, n])
    queries = patch_graph(G)

    fig = plt.figure(figsize=(4, 4))

    path, ds = bfs(G, start, goal)
    print("       bfs", *path)
    plt.subplot(2, 2, 1)
    plt.title('Breath-first')
    plot_trajectory_2d(path, label="bfs")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.6)

    queries.clear()
    path, ds = heuristic_search(G, start, goal, heuristic)
    print("heuristics", *path)
    plt.subplot(2, 2, 2)
    plt.title('Heuristic Search')
    plot_trajectory_2d(path, label="heuristics")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.6)

    G = nx.grid_graph(dim=[n, n])
    G.add_edge((0, 0), (0, 1), weight=1)
    for e in G.edges(data=True):
        e[-1]['weight'] = 1

    queries = patch_graph(G)

    path, ds = dijkstra(G, start, goal)
    print("  dijkstra", *path)
    plt.subplot(2, 2, 3)
    plt.title('Dijkstra')
    plot_trajectory_2d(path, label="dijkstra")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.6)

    queries.clear()
    path, ds = a_star(G, start, goal, heuristic)
    print("        a*", *path)
    plt.subplot(2, 2, 4)
    plt.title('A*')
    plot_trajectory_2d(path, label="A*")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.6)

    plt.legend(loc="upper left", bbox_to_anchor=(0.45, 0.8), framealpha=1, frameon=False, fontsize=12)
    from ml_logger import logger

    plt.tight_layout()
    logger.savefig("../figures/search_range.png", dpi=300)
    plt.show()
