"""
Comparison between various graph-search algorithms.

reference: https://www.redblobgames.com/pathfinding/a-star/introduction.html#dijkstra
ref_2: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
"""
from collections import deque, defaultdict
import networkx as nx
import numpy as np


def backtrack(link, start, goal):
    node = goal
    yield node
    i = 0
    while node in link:
        node = link[node]
        yield node
        if node == start:
            break


def bfs(graph: nx.Graph, start, goal=None):
    """breath-first search
    :param graph: a graph object
    :param start: the starting node
    :param goal: optional, when not supplied generates the entire path tree.
    :return:
    """
    frontier = deque()
    visited = dict()

    frontier.append(start)
    while len(frontier):
        current = frontier.popleft()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            frontier.append(n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=2) * 5


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


def heuristic_search(graph: nx.Graph, start, goal):
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, start)
    while len(frontier):
        _, current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            to_goal = heuristic(n, goal)
            frontier.push(to_goal, n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


def dijkstra(graph: nx.Graph, start, goal):
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, start)
    while len(frontier):
        sofar, current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            edge_len = graph.get_edge_data(current, n)['weight']
            frontier.push(sofar + edge_len, n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


def a_star(graph: nx.Graph, start, goal):
    frontier = PriorityQue()
    visited = dict()

    frontier.push(0, 0, start)
    while len(frontier):
        _, sofar, current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            edge_len = graph.get_edge_data(current, n)['weight']
            to_goal = heuristic(n, goal)
            frontier.push(sofar + edge_len + to_goal, sofar + edge_len, n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        plt.arrow(x, y, (x_ - x) * 0.9, (y_ - y) * 0.9, **kwargs, head_width=0.1, head_length=0.1,
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from collections import defaultdict

    n, start, goal = 7, (0, 0), (6, 6)

    G = nx.grid_graph(dim=[n, n])
    queries = patch_graph(G)

    fig = plt.figure(figsize=(4, 4))

    path = bfs(G, start, goal)
    print("       bfs", *path)
    plt.subplot(2, 2, 1)
    plt.title('Breath-first')
    plot_trajectory_2d(path, label="bfs")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.3)

    queries.clear()
    path = heuristic_search(G, start, goal)
    print("heuristics", *path)
    plt.subplot(2, 2, 2)
    plt.title('Heuristic Search')
    plot_trajectory_2d(path, label="heuristics")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.3)

    G = nx.grid_graph(dim=[n, n])
    G.add_edge((0, 0), (0, 1), weight=1)
    for e in G.edges(data=True):
        e[-1]['weight'] = 1

    queries = patch_graph(G)

    path = dijkstra(G, start, goal)
    print("  dijkstra", *path)
    plt.subplot(2, 2, 3)
    plt.title('Dijkstra')
    plot_trajectory_2d(path, label="dijkstra")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.3)

    queries.clear()
    path = a_star(G, start, goal)
    print("        a*", *path)
    plt.subplot(2, 2, 4)
    plt.title('A*')
    plot_trajectory_2d(path, label="A*")
    plt.scatter(*zip(*queries.keys()), color="#23aaff", linewidths=0, alpha=0.3)

    plt.legend(loc="upper left", bbox_to_anchor=(0.45, 0.8), framealpha=1, frameon=False, fontsize=12)
    from ml_logger import logger

    plt.tight_layout()
    logger.savefig("../figures/search_range.png", dpi=300)
    plt.show()
