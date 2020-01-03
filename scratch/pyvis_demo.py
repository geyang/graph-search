from pyvis.network import Network

net = Network()

net.add_node(1, label="Nonde 1")
net.add_node(2)

net.toggle_physics(True)
net.show("test.html")

# import networkx as nx
#
# nxg = nx.complete_graph(10)
