class Graph:
    def nodes(self):
        raise NotImplementedError

    def neighbors(self, key: int):
        raise NotImplementedError


class SimpleGraph(Graph):
    def __init__(self, num_nodes=None):
        self.edges = {}
        if num_nodes:
            self.nodes = [*range(num_nodes)]

    def init_random(self, threshold=0.9):
        import numpy as np
        i, j = np.meshgrid(self.nodes, self.nodes)
        l = len(self.nodes)
        dense_edges = np.random.rand(l, l) > threshold
        for i, j in np.concatenate([i[..., None], j[..., None]], axis=-1)[dense_edges]:
            if i == j:
                continue
            self.edges[(int(i), int(j))] = 1
            # symmetric.
            self.edges[(int(j), int(i))] = 1

    def edge(self, a, b):
        return self.edges.get((a, b), None)

    def neighbors(self, i):
        ns = set()
        for n in self.nodes:
            e = self.edges.get((i, n), None)
            if e:
                ns.add(n)
        return list(ns)

    def distance(self, a, b):
        from graph_search import bfs
        return len(bfs(self, a, b))

    def __repr__(self):
        return f"{list(self.edges.keys())}"

    def show(self, file="state_graph.html", physics=True):
        from pyvis.network import Network

        net = Network()
        for (i, j), weight in self.edges.items():
            net.add_node(i)
            net.add_node(j)
            net.add_edge(i, j, value=weight)

        net.toggle_physics(physics)
        net.show(file)


class WeightedGraph(SimpleGraph):
    def init_random(self, threshold=0.9, low=1, high=1):
        import random
        import numpy as np
        i, j = np.meshgrid(self.nodes, self.nodes)
        l = len(self.nodes)
        dense_edges = np.random.rand(l, l) > threshold
        for i, j in np.concatenate([i[..., None], j[..., None]], axis=-1)[dense_edges]:
            if i == j:
                continue
            weight = random.uniform(low, high)
            self.edges[(int(i), int(j))] = weight
            self.edges[(int(j), int(i))] = weight


if __name__ == '__main__':
    example_graph = SimpleGraph(num_nodes=10)
    example_graph.init_random()
    print(example_graph)
    example_graph.show()
