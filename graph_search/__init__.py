"""
Comparison between various graph-search algorithms.

reference: https://www.redblobgames.com/pathfinding/a-star/introduction.html#dijkstra
ref_2: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
"""
from collections import deque
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
    frontier = deque()
    visited = dict()

    frontier.append(start)
    while len(frontier):
        current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            frontier.append(n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)


from heapq import heapify, heappush, heappop


class PriorityQue:
    def __init__(self):
        self.queue = []

    def push(self, priority, ind):
        heappush(self.queue, (priority, ind))

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

    frontier.push(0, start)
    while len(frontier):
        sofar, current = frontier.pop()
        for n in graph.neighbors(current):
            if n in visited:
                continue
            edge_len = graph.get_edge_data(current, n)['weight']
            to_goal = heuristic(n, goal)
            frontier.push(sofar + edge_len + to_goal, n)
            visited[n] = current
            if n == goal:
                return list(backtrack(visited, start, goal))[::-1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    G = nx.grid_graph(dim=[5, 5])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=25)
    plt.show()

    path = bfs(G, (0, 0), (4, 4))
    print("       bfs", *path)

    path = heuristic_search(G, (0, 0), (4, 4))
    print("heiristics", *path)

    G = nx.grid_graph(dim=[5, 5])
    G.add_edge((0, 0), (0, 1), weight=1)
    for e in G.edges(data=True):
        e[-1]['weight'] = 1
    path = dijkstra(G, (0, 0), (4, 4))
    print("  dijkstra", *path)

    path = a_star(G, (0, 0), (4, 4))
    print("        a*", *path)
