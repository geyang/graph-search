"""
The difference between dijkstra and breath-first-search
is that dijkstra does not have an explicit concept of
"frontier", because all unvisited nodes can be thought
of as the frontier.

for this reason we need a heap that we can update the
priority with.
"""
from heapq import heappush, heappop, heapify
from graph_search.graph import WeightedGraph

REMOVED = "<removed>"


class PriorityQueue:
    """Implements priority update."""

    def __init__(self, nodes: dict):
        self.index = nodes
        self.heap = list(nodes.values())
        heapify(self.heap)

    def remove(self, node):
        self.index[node][-1] = REMOVED
        del self.index[node]

    def __len__(self):
        return len(self.index.keys())

    def __getitem__(self, item):
        return self.index[item]

    def update(self, node, priority, blob):
        self.index[node][-1] = REMOVED
        new = [priority, node, blob]
        self.index[node] = new
        heappush(self.heap, new)

    def pop(self):
        while len(self.heap):
            priority, node, blob = heappop(self.heap)
            if blob == REMOVED:
                continue
            del self.index[node]
            return priority, node, blob


def dijkstra(graph: WeightedGraph, start: int, goal: int = None):
    index = dict()
    for node in graph.nodes:
        value, path = (0, [start]) if node == start else (float('inf'), tuple())
        index[node] = [value, node, path]

    queue = PriorityQueue(index)
    visited = dict()

    # When you are tired, go home.
    # eat between 1:30-2:00. 15:34 is way too late.
    # the mac double meal is only 4.33 including tax.

    while len(queue):
        length, current, path = queue.pop()
        print(f"Visiting: {current}, depth: {len(path) - 1}, length: {length},", end=' ')
        if current == goal:
            return path, length
        visited[current] = length, path
        ns = graph.neighbors(current)
        print(f"neighbors: {ns}")
        for next in ns:
            if next in visited:
                continue
            new_length = length + graph.edge(current, next)
            shortest_length, next, _ = queue[next]
            if new_length < shortest_length:
                queue.update(next, new_length, [*path, next])


if __name__ == '__main__':
    from graph_search.graph import WeightedGraph

    example_graph = WeightedGraph(num_nodes=10)
    example_graph.init_random(0.87, low=0.5)
    print(example_graph)
    example_graph.show("../figures/graph.html")

    path, length = dijkstra(example_graph, 0, 8)
    print(path, length)
