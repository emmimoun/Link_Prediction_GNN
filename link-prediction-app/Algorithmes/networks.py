import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

G=nx.Graph()
G.add_edge('1','2')

nx.draw(G)
plt.show()