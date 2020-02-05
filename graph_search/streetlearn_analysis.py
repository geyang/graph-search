from functools import partial

import networkx as nx
import numpy as np
import gym
from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto

from graph_search import methods, short_names
from streetlearn import StreetLearnDataset


class Args(ParamsProto):
    env_id = "streetlearn_small"
    neighbor_r = 2.4e-4
    neighbor_r_min = None

    h_scale = 1.2

    # plotting
    visualize_graph = True


def load_streetlearn(data_path="~/fair/streetlearn/processed-data/manhattan-large", pad=0.1):
    from streetlearn import StreetLearnDataset
    import matplotlib.pyplot as plt
    from os.path import expanduser
    path = expanduser(data_path)
    d = StreetLearnDataset(path)
    d.select_bbox(-73.997, 40.726, 0.01, 0.008)
    d.show_blowout("NYC-large", show=True)

    a = d.bbox[0] + d.bbox[2] * pad, d.bbox[1] + d.bbox[3] * pad
    b = d.bbox[0] + d.bbox[2] * (1 - pad), d.bbox[1] + d.bbox[3] * (1 - pad)
    (start, _), (goal, _) = d.locate_closest(*a), d.locate_closest(*b)

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(*d.lng_lat[start], marker="o", s=100, linewidth=3,
                edgecolor="black", facecolor='none', label="start")
    plt.scatter(*d.lng_lat[goal], marker="x", s=100, linewidth=3,
                edgecolor="none", facecolor='red', label="end")
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.7), framealpha=1,
               frameon=False, fontsize=12)
    d.show_blowout("NYC-large", fig=fig, box_color='gray', box_alpha=0.1,
                   show=True, set_lim=True)

    return d, start, goal

    # 1. get data
    # 2. build graph
    # 3. get start and goal
    # 4. make plans


def plot_graph(graph):
    # fig = plt.figure(figsize=(3, 3))
    nx.draw(graph, [n['pos'] for n in graph.nodes.values()],
            node_size=0, node_color="gray", alpha=0.7, edge_color="gray")
    plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.show()


def maze_graph(dataset: StreetLearnDataset):
    from tqdm import tqdm

    all_nodes = dataset.lng_lat
    graph = nx.Graph()
    for node, xy in enumerate(tqdm(all_nodes, desc="build graph")):
        graph.add_node(node, pos=xy)

    for node, a in tqdm(graph.nodes.items(), desc="add edges"):
        (ll,), (ds,), (ns,) = dataset.neighbor([node], r=Args.neighbor_r)
        for neighbor, d in zip(ns, ds):
            graph.add_edge(node, neighbor, weight=d)

    return graph
    # if Args.visualize_graph:
    #     plot_graph(graph)
    #     plt.gca().set_aspect(dataset.lat_correction)
    #     plt.show()


# noinspection PyPep8Naming,PyShadowingNames
def heuristic(a, b, G: nx.Graph, scale=1, lat_correction=1 / 0.74):
    a = [G.nodes[n]['pos'] for n in a]
    b = [G.nodes[n]['pos'] for n in b]
    magic = [1, lat_correction]
    return np.linalg.norm((np.array(a) - np.array(b)) * magic, ord=1, axis=-1) * scale


def plot_trajectory_2d(path, color='black', **kwargs):
    for (x, y), (x_, y_) in zip(path[:-1], path[1:]):
        dx = (x_ - x)
        dy = (y_ - y)
        d = np.linalg.norm([dx, dy], ord=2)
        plt.arrow(x, y, dx * 0.8, dy * 0.8, **kwargs, head_width=d * 0.3, head_length=d * 0.3,
                  length_includes_head=True, head_starts_at_zero=True, fc=color, ec=color)


def set_fig(dataset: StreetLearnDataset):
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.gca().set_aspect(dataset.lat_correction)


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
    from waterbear import DefaultBear
    import matplotlib.pyplot as plt
    from ml_logger import logger

    dataset, start, goal = load_streetlearn()
    G = maze_graph(dataset)
    queries = patch_graph(G)

    # goal -= 120 # 10 worked well
    cache = DefaultBear(dict)

    fig = plt.figure(figsize=(4, 4), dpi=300)

    for i, (key, search) in enumerate(methods.items()):
        queries.clear()
        name = search.__name__
        title, *_ = search.__doc__.split('\n')
        short_name = short_names[key]

        path, ds = search(G, start, goal, partial(heuristic, G=G, scale=1.2))
        cache.cost[short_name] = len(queries.keys())
        cache.len[short_name] = len(ds)
        print(f"{key:>10} len: {len(path)}", f"cost: {len(queries.keys())}")
        plt.subplot(2, 2, i + 1)
        plt.title(title, pad=10)
        # plot_graph(G)
        plot_trajectory_2d(ind2pos(G, path, 100), label=short_name)
        plt.scatter(*zip(*ind2pos(G, queries.keys(), 100)), color="gray", s=3, alpha=0.1)
        set_fig(dataset)

    # plt.legend(loc="upper left", bbox_to_anchor=(0.45, 0.8), framealpha=1, frameon=False, fontsize=12)
    plt.tight_layout()
    logger.savefig("../figures/streetlearn_plans.png", dpi=300)
    plt.show()
    plt.close()

    # colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247']
    # for i, (k, v) in enumerate(cache.items()):
    #     plt.bar(k, v, color=colors[i])

    fig = plt.figure(figsize=(3.8, 3), dpi=300)
    plt.title('Planning Cost')
    plt.bar(cache.cost.keys(), cache.cost.values(), color="gray", width=0.8)
    plt.ylim(0, max(cache.cost.values()) * 1.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig("../figures/streetlearn_cost.png", dpi=300)
    plt.ylabel('# of distance lookup')
    plt.show()

    fig = plt.figure(figsize=(3.8, 3), dpi=300)
    plt.title('Plan Length')
    plt.bar(cache.len.keys(), cache.len.values(), color="gray", width=0.8)
    plt.ylim(0, max(cache.len.values()) * 1.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig("../figures/streetlearn_length.png", dpi=300)
    plt.ylabel('Path Length')
    plt.show()

    logger.print('done', color="green")
