import networkx as nx
import matplotlib.pyplot as plt

# Create an undirected graph
G = nx.Graph()

# Add nodes with features (for now, just labels)
G.add_node(1, label="Person A")
G.add_node(2, label="Person B")
G.add_node(3, label="Person C")

# Add edges (relationships)
G.add_edge(1, 2)
G.add_edge(2, 3)

# Draw the graph
plt.figure(figsize=(5, 5))
nx.draw(G, with_labels=True, node_color="lightblue", node_size=2000, font_size=15)
plt.show()