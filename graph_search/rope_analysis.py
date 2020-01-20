"""
Before running this, make sure that the chunk prefix are computed.
Look for those code in plan2vec_experiments/rope/rope_pairwise.py

First make sure you have a trained rope local metric function,
if not train one.

Then run the `pair_wise` function, which computes this matrix in
parallel chunks.
"""

import networkx as nx
import numpy as np
# import gym
# from params_proto import proto_partial
from params_proto.neo_proto import ParamsProto

from graph_search import bfs, heuristic_search, dijkstra, a_star
from streetlearn import StreetLearnDataset


class Args(ParamsProto):
    env_id = "streetlearn_small"
    n_envs = 10
    n_rollout = 50
    n_timesteps = 1
    neighbor_r = 2.4e-4
    neighbor_r_min = None

    h_scale = 1
    # threshold = 0.61
    threshold = 0.06

    # plotting
    visualize_graph = False


def plot_graph(graph):
    print('plotting the graph')
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=25)
    print('plotting finished.')


def rope_graph():
    from plan2vec_experiments import instr
    from plan2vec_experiments.rope.rope_pairwise import load_pairwise, build_graph

    # thunk = instr(load_pairwise)
    # ds, all_images = thunk()
    ds, all_images = load_pairwise()
    graph = build_graph(ds, threshold=Args.threshold)
    # graph.show("figures/rope_graph.html")

    # Note: What is the average score for 1-step and 2-step neighbors?
    eye = np.eye(len(ds), dtype=bool)
    shifted = np.concatenate([np.zeros([len(ds), 1], dtype=bool), eye[:, :-1], ], axis=-1)
    avg_neighbor_score = ds[shifted].mean()
    logger.print(f"average score for neighbors is: {avg_neighbor_score}")

    # visualize using anealed position
    if Args.visualize_graph:
        plot_graph(graph)
    logger.savefig("figures/rope_network_graph.png")
    plt.tight_layout()
    plt.show()

    return graph, ds, all_images


# index-based heuristic
def heuristic(a, b):
    return abs(a - b)


def plot_trajectory_rope(path, ds, all_images, n_cols=8, title=None, key="figures/rope_play.png"):
    from math import ceil
    from matplotlib.gridspec import GridSpec
    from ml_logger import logger

    stack = all_images[path][:, 0]
    n, h, w, *c = stack.shape
    n_rows = ceil(n / n_cols)
    # todo: add color background -- need to decide on which library to use.
    fig = plt.figure(figsize=np.array(1.4, dtype=int) * [n_cols, n_rows])

    if title:
        plt.suptitle(title, fontsize=14)
    gs = GridSpec(n_rows, 10)
    index = 0
    for _row in range(n_rows):
        for _col in range(10):
            if index == len(stack):
                break
            plt.subplot(gs[_row, _col])
            plt.imshow(stack[index], cmap='gray')
            plt.text(0, 10, f"#{path[index]}", fontsize=8)
            if index > 0:
                plt.text(0, 60, f"{ds[index - 1]:0.2f}", fontsize=8)
            plt.axis('off')
            index += 1

    plt.tight_layout()
    logger.savefig(key=key)
    plt.show()
    return
    # # composite = np.zeros([h * n_rows, w * n_cols, *c], dtype='uint8')
    # # for i in range(n_rows):
    # #     for j in range(n_cols):
    # #         k = i * n_cols + j
    # #         if k >= n:
    # #             break
    # #         # todo: remove last index
    # #         composite[i * h: i * h + h, j * w: j * w + w] = stack[k]
    # #
    # # plt.imshow(composite, cmap="gray")
    # # plt.gca().set_axis_off()
    #
    # from math import ceil
    #
    # n_rows = ceil(n / n_cols)
    # fig = plt.figure(figsize=np.array(1.4, dtype=int) * [n_cols, n_rows])
    # plt.suptitle(f'Summary of Trajectory #{traj_ind}', fontsize=14)
    # gs = GridSpec(n_rows, 10)
    # index = -1
    # for _row in range(n_rows):
    #     for _col in range(10):
    #         index += 1
    #         plt.subplot(gs[_row, _col])
    #         try:
    #             plt.imshow(picked[index], cmap='gray')
    #             plt.text(0, 10, f"#{index * 20}", fontsize=8)
    #         except:
    #             pass
    #         plt.axis('off')


# logger.log_images(np.concatenate(all_images)[:, 0], key=key)


def set_fig():
    pass


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

    # the local metric in this case really does not work well.
    # The data ontains some bad samples and it is throwing off my
    from plan2vec_experiments import instr

    # chunk_prefix
    G, pairwise, all_images = instr(rope_graph)()
    queries = patch_graph(G)

    # def plan_experiment(start, goal, G, queries):
    # start, goal = 1310, 1320
    start, goal = 1111, 2121
    # goal -= 120 # 10 worked well
    cache = DefaultBear(dict)
    logger.configure()

    path = [*range(start, goal + 1)]
    ds = [pairwise[i, j] for i, j in zip(path[:-1], path[1:])]
    plot_trajectory_rope(path, ds, all_images, title=f'Ground-Truth', key=f"../figures/rope_plans/bfs.png")

    queries.clear()
    path = bfs(G, start, goal)
    ds = [pairwise[i, j] for i, j in zip(path[:-1], path[1:])]
    cache.cost['bfs'] = len(queries.keys())
    cache.len['bfs'] = len(path)
    print(f"       bfs len: {len(path)}", *path)
    print(f"# of queries {len(queries.keys())}")
    plot_trajectory_rope(path, ds, all_images, title=f'Breath-first', key=f"../figures/rope_plans/bfs.png")

    queries.clear()
    path = heuristic_search(G, start, goal, heuristic)
    ds = [pairwise[i, j] for i, j in zip(path[:-1], path[1:])]
    cache.cost.update(heuristic=len(queries.keys()))
    cache.len['heuristic'] = len(path)
    print(f"heuristics len: {len(path)}", *path)
    plot_trajectory_rope(path, ds, all_images, title='Heuristic', key=f"../figures/rope_plans/bfs.png")

    queries.clear()
    path = dijkstra(G, start, goal)
    ds = [pairwise[i, j] for i, j in zip(path[:-1], path[1:])]
    cache.cost["Dijkstra's"] = len(queries.keys())
    cache.len["Dijstra's"] = len(path)
    print(f"  dijkstra len: {len(path)}", *path)
    plot_trajectory_rope(path, ds, all_images, title=f"Dijkstra's", key=f"../figures/rope_plans/bfs.png")

    queries.clear()
    path = a_star(G, start, goal, heuristic)
    ds = [pairwise[i, j] for i, j in zip(path[:-1], path[1:])]
    cache.cost["A*"] = len(queries.keys())
    cache.len["A*"] = len(path)
    print(f"        a* len: {len(path)}", *path)
    plot_trajectory_rope(path, ds, all_images, title="A*", key=f"../figures/rope_plans/bfs.png")

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
    logger.savefig("../figures/rope_cost.png", dpi=300)
    plt.ylabel('# of distance lookup')
    plt.show()

    fig = plt.figure(figsize=(3.8, 3), dpi=300)
    plt.title('Plan Length')
    plt.bar(cache.len.keys(), cache.len.values(), color="gray", width=0.8)
    plt.ylim(0, max(cache.len.values()) * 1.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    logger.savefig("../figures/rope_length.png", dpi=300)
    plt.ylabel('# of Steps')
    plt.show()

    logger.print('done', color="green")
