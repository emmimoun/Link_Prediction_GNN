import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import math
import pandas as pd
import numpy as np



def get_graph_info(graph):
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Available nodes:", list(graph.nodes))
    print("Available edges:", list(graph.edges))
    if type(graph) == nx.classes.digraph.DiGraph:
        print("Connected components:", 
              list(nx.weakly_connected_components(directed_graph)))
    else:
        print("Connected components:", list(nx.connected_components(graph)))
    print("Node degree:", dict(graph.degree()))
    
    
  

#edgelist = [('0', '1'), ('0', '2'), ('0', '3'), ('0', '4'),('20', '32'), ('20', '33'), ('22', '32'), ('22', '33'), ('23', '25'),('15', '33'), ('18', '32'), ('18', '33'), ('19','33'), ('8', '33'), ('9', '33'), ('13', '33'), ('14', '32'), ('14', '33'), ('15', '32'), ('0', '5'), ('0', '6'), ('0', '7'), ('0', '8'), ('0', '10'), ('0', '11'), ('0', '12'), ('0', '13'), ('0', '17'), ('0', '19'), ('0', '21'), ('0', '31'), ('1', '2'), ('1', '3'), ('1', '7'), ('1', '13'), ('1', '17'), ('1', '19'), ('1', '21'), ('1', '30'), ('2', '3'), ('2', '7'), ('2', '8'), ('2', '9'), ('2', '13'), ('2', '27'), ('2', '28'), ('2', '32'), ('3', '7'), ('3', '12'), ('3', '13'), ('4', '6'), ('4', '10'), ('5', '6'), ('5', '10'), ('5', '16'), ('6', '16'), ('8', '30'), ('8', '32'),  ('23', '27'), ('23', '29'), ('23', '32'), ('23', '33'), ('24', '25'), ('24', '27'), ('24', '31'), ('25', '31'), ('26', '29'), ('26', '33'), ('27', '33'), ('28', '31'), ('28', '33'), ('29', '32'), ('29', '33'), ('30', '32'), ('30', '33'), ('31', '32'), ('31', '33'), ('32', '33')]


mygraph = nx.Graph(nx.karate_club_graph())   
    

#get_graph_info(mygraph)   


# club color reference
nation_color_dict = {
    'Mr. Hi': '#aff8df',
    'Officer': '#ffcbc1'
}

# function to assign color for node visualization
def create_node_colors_from_graph(graph, club_color_dict):
    node_colors = []
    for node, club in list(graph.nodes(data="club")):
        if club in club_color_dict:
            node_colors.append(club_color_dict[club])
    return node_colors

# get node colors for plotting
node_colors = create_node_colors_from_graph(mygraph, nation_color_dict)    



# create visualization
pos = nx.spring_layout(mygraph, 
                       k=0.3, iterations=50,
                       seed=2)

plt.figure(1,figsize=(10,6)) 
nx.draw(mygraph,
        pos = pos,
        node_color=node_colors,
        node_size=1000,
        with_labels=True,
        font_size = 20,
        font_color='black')
plt.title("Karate Club Social Network")
plt.show()



############### Link Prediction  #######################


# get the common neighbor dataframe
def get_common_neighbors(graph):
    potential_egdes = nx.non_edges(graph)
    common_neighbors = []
    for source, target in potential_egdes:
        common_neighbors.append([source, target, 
            len(list(nx.common_neighbors(graph, source, target)))])
    common_neighbors = sorted(common_neighbors, key=lambda x: x[-1], reverse=True)
    return pd.DataFrame(common_neighbors, 
                        columns=["source", "target", "common_neighbors"])
    
    

common_neighbors_df = get_common_neighbors(mygraph)

print( "the common neighbor measure for possible connections/edges ")
print(common_neighbors_df["common_neighbors"].value_counts(normalize=True) )  

print(common_neighbors_df["common_neighbors"].value_counts())

#print(common_neighbors_df.head())

#print(list(nx.common_neighbors(mygraph, 2, 33)))

# function to return a node's neighbor nodes as a list
def get_neighbors(graph, node):
    return [n for n in graph.neighbors(node)]


# function to return the subgraph containing 2 nodes' common neighbors
def get_common_neighbor_subgraph(graph, source, target):
    nodes = [source, target] + list(nx.common_neighbors(mygraph, source, target))
    return graph.subgraph(nodes)


# function create node color list for likely connected two nodes with their
# common neighbors
def create_source_target_colors(graph, source, target):
    nodes = list(graph.nodes())
    potential_connected_node_colors = ["#EFD1BB"] * len(nodes)
    for index in range(len(nodes)):
        if nodes[index] in [source, target]:
            potential_connected_node_colors[index] = "#F47315"
    return potential_connected_node_colors



def visualize_likely_connected_nodes(graph, source, target):
    # get subgraph
    subgraph = get_common_neighbor_subgraph(mygraph, source, target)

    # create visualization
    node_colors = create_source_target_colors(subgraph, source, target)
    title = f"Karate Club Social Network: Common Neighbors of Node {source} and {target}"
    pos = nx.spring_layout(subgraph, 
                           k=0.3, iterations=50,
                           seed=2)

    plt.figure(1,figsize=(10,6)) 
    nx.draw(subgraph,
            pos = pos,
            node_size=1000,
            node_color=node_colors,
            with_labels=True,
            font_size = 20,
            font_color='black')
    plt.title(title)
    plt.show() 
    
visualize_likely_connected_nodes(mygraph, 2, 33) 
 
print("les voisins commun entre 2 et 33")  

x=list(nx.common_neighbors(mygraph, 2, 33))
print(x)




##################   REssource allocation index 

# function return the resource allocation index dataframe sorted by the
# resource allocation index value
def get_resource_allocation_index(graph):
    return pd.DataFrame(sorted(list(nx.resource_allocation_index(graph)),
                        key=lambda x: x[-1], reverse=True),
                        columns=["source", "target", "resource_allocation_index"])

resource_allocation_index_df = get_resource_allocation_index(mygraph)
resource_allocation_index_df.resource_allocation_index.plot.hist(bins=10, figsize=(8,5), 
    title="Resource Allocation Index Distribution");    

plt.show()

print(resource_allocation_index_df.head())
    
    
visualize_likely_connected_nodes(mygraph, 2, 33)    