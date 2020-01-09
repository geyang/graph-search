# graph_search

collection of graph-search algorithms

## Constructing and Visualizing the State Graph

use a grid-graph.

## Graph Search Algorithms

- *breath first*: use path_length and push order as priority.
- *heuristic*: use D(next, goal) as priority.
- *dijkstra*: use path_length as priority.
- *a**: use d(path) + D(next, goal) as priority.

This is reflected in the implementation in [./graph_search](./graph_search/__init__.py).

<p align="center">
   <img width="300px" height="300px"
        alt="bfs,heuristic,dijkstra and a* algorithms" 
        src="figures/comparison.png"/>
</p>

## Prioritized Search

A planning heuristic helps reduce the planning expenditure. The left column are breath-first-search and dijkstra, both do not use a planning heuristic. On the right are heuristic search and A*.

The <span color="#23aaff">blue</span> colored dots represent the nodes the search algorithm has "touched". Heuristics help reduce the cost during planning.

<p align="center">
   <img width="300px" height="300px"
        alt="bfs,heuristic,dijkstra and a* algorithms" 
        src="figures/search_range.png"/>
</p>

## Graph Interface

We need three methods:

- `get_edge_data(node, node_2)`
- `neighbors(node)`
- `heuristics(next, goal)`
