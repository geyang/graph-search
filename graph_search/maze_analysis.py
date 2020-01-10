"""
Comparison between various graph-search algorithms.

reference: https://www.grayblobgames.com/pathfinding/a-star/introduction.html#dijkstra
ref_2: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
"""
from collections import deque, defaultdict
from functools import partial

import networkx as nx
import numpy as np
import gym
import ge_world
from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto


class Args(ParamsProto):
    env_id = "CMazeDiscreteIdLess-v0"
    n_envs = 10
    n_rollout = 50
    n_timesteps = 1
    neighbor_r = 0.036
    neighbor_r_min = 0.02

    # plotting
    visualize_graph = True


def d(xy, xy_):
    return np.linalg.norm(xy - xy_, ord=2)


# @proto_partial(Args)
def sample_trajs(seed, env_id=Args.env_id):
    import matplotlib.pyplot as plt

    np.random.seed(seed)
    env = gym.make(env_id)
    env.reset()

    trajs = []
    for i in range(Args.n_rollout):
        obs = env.reset()
        path = [obs['x']]
        trajs.append(path)
        for t in range(Args.n_timesteps - 1):
            obs, reward, done, info = env.step(np.random.randint(low=0, high=7))
            path.append(obs['x'])
    trajs = np.array(trajs)
    # fig = plt.figure(figsize=(3, 3))
    # for path in trajs:
    #     plt.plot(*zip(*path), color="gray")
    # plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.show()
    from ml_logger import logger
    logger.print(f'seed {seed} has finished sampling.', color="green")
    return trajs


def plot_graph(graph):
    # fig = plt.figure(figsize=(3, 3))
    nx.draw(graph, [n['pos'] for n in graph.nodes.values()],
            node_size=0, node_color="gray", alpha=0.7, edge_color="gray")
    plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.show()


def maze_graph(trajs):
    all_nodes = np.concatenate(trajs)
    graph = nx.Graph()
    for i, xy in enumerate(all_nodes):
        graph.add_node(i, pos=xy)
    for i, a in graph.nodes.items():
        for j, b in graph.nodes.items():
            if d(a['pos'], b['pos']) < Args.neighbor_r and \
                    d(a['pos'], b['pos']) > Args.neighbor_r_min:
                graph.add_edge(i, j, weight=d(a['pos'], b['pos']))

    # if Args.visualize_graph:
    #     plot_graph(graph)

    return graph


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


def heuristic(a, b, G, scale=1):
    a, b = G.nodes[a]['pos'], G.nodes[b]['pos']
    return np.linalg.norm(np.array(a) - np.array(b), ord=2) * scale


def heuristic_search(graph: nx.Graph, start, goal, heuristic):
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


def a_star(graph: nx.Graph, start, goal, heuristic):
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
        dx = (x_ - x)
        dy = (y_ - y)
        d = np.linalg.norm([dx, dy], ord=2)
        plt.arrow(x, y, dx * 0.8, dy * 0.8, **kwargs, head_width=d * 0.3, head_length=d * 0.3,
                  length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)


def set_fig():
    # plt.gca().set_axis_off()
    # plt.gca().set_aspect('equal')
    # plt.xlim(-0.26, .26)
    # plt.ylim(-0.26, .26)
    # plt.gca().set_scale(5)
    plt.xlim(-24, 24)
    plt.ylim(-24, 24)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(True)
    # plt.gca().spines['bottom'].set_visible(True)


def get_neighbor(G, pos):
    ds = np.array([d(n['pos'], pos) for i, n in G.nodes.items()])
    return np.argmin(ds)


def ind2pos(G, inds, scale=1):
    return [G.nodes[n]['pos'] * scale for n in inds]


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
    from collections import defaultdict
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from multiprocessing.pool import Pool
    from ml_logger import logger

    p = Pool(10)
    traj_batch = p.map(sample_trajs, range(Args.n_envs))

    G = maze_graph(np.concatenate(traj_batch))

    queries = patch_graph(G)

    start, goal = get_neighbor(G, (-0.16, 0.16)), get_neighbor(G, (-0.16, -0.16))

    fig = plt.figure(figsize=(4, 4), dpi=300)

    path = bfs(G, start, goal)
    logger.log(bfs=len(queries.keys()))
    print(f"       bfs len: {len(path)}", *path)
    print(f"# of queries {len(queries.keys())}")
    plt.subplot(2, 2, 1)
    plt.title(f'Breath-first')
    # plot_graph(G)
    plot_trajectory_2d(ind2pos(G, path, 100), label="bfs")
    plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.6)
    set_fig()

    queries.clear()
    path = heuristic_search(G, start, goal, partial(heuristic, G=G))
    logger.log(heuristic=len(queries.keys()))
    print(f"heuristics len: {len(path)}", *path)
    plt.subplot(2, 2, 2)
    plt.title('Heuristic')
    # plot_graph(G)
    plot_trajectory_2d(ind2pos(G, path, 100), label="heuristics")
    plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.6)
    set_fig()

    queries.clear()
    path = dijkstra(G, start, goal)
    logger.log_key_value("Dijkstra's", len(queries.keys()))
    print(f"  dijkstra len: {len(path)}", *path)
    plt.subplot(2, 2, 3)
    plt.title(f"Dijkstra's")
    # plot_graph(G)
    plot_trajectory_2d(ind2pos(G, path, 100), label="dijkstra")
    plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.6)
    set_fig()

    queries.clear()
    path = a_star(G, start, goal, partial(heuristic, G=G, scale=3))
    logger.log_key_value("A*", len(queries.keys()))
    print(f"        a* len: {len(path)}", *path)
    plt.subplot(2, 2, 4)
    plt.title('A*')
    # plot_graph(G)
    plot_trajectory_2d(ind2pos(G, path, 100), label="A*")
    plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.6)
    set_fig()

    # plt.legend(loc="upper left", bbox_to_anchor=(0.45, 0.8), framealpha=1, frameon=False, fontsize=12)
    plt.tight_layout()
    logger.savefig("../figures/maze_plans.png", dpi=300)
    plt.show()
    plt.close()

    logger.key_value_cache.data.pop('__timestamp')
    data = logger.key_value_cache.data

    # colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']

    fig = plt.figure(figsize=(3.8, 3), dpi=300)
    plt.title('Planning Cost')
    # for i, (k, v) in enumerate(data.items()):
    #     plt.bar(k, v, color=colors[i])
    plt.bar(data.keys(), data.values(), color="gray", width=0.8)
    plt.ylim(0, max(data.values()) * 1.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig("../figures/maze_cost.png", dpi=300)
    plt.ylabel('# of distance lookup')
    plt.show()

    logger.print('done', color="green")
