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
 

edgelist = [('0', '1'), ('0', '2'), ('0', '3'), ('0', '4'),('20', '32'), ('20', '33'), ('22', '32'), ('22', '33'), ('23', '25'),('15', '33'), ('18', '32'), ('18', '33'), ('19','33'), ('8', '33'), ('9', '33'), ('13', '33'), ('14', '32'), ('14', '33'), ('15', '32'), ('0', '5'), ('0', '6'), ('0', '7'), ('0', '8'), ('0', '10'), ('0', '11'), ('0', '12'), ('0', '13'), ('0', '17'), ('0', '19'), ('0', '21'), ('0', '31'), ('1', '2'), ('1', '3'), ('1', '7'), ('1', '13'), ('1', '17'), ('1', '19'), ('1', '21'), ('1', '30'), ('2', '3'), ('2', '7'), ('2', '8'), ('2', '9'), ('2', '13'), ('2', '27'), ('2', '28'), ('2', '32'), ('3', '7'), ('3', '12'), ('3', '13'), ('4', '6'), ('4', '10'), ('5', '6'), ('5', '10'), ('5', '16'), ('6', '16'), ('8', '30'), ('8', '32'),  ('23', '27'), ('23', '29'), ('23', '32'), ('23', '33'), ('24', '25'), ('24', '27'), ('24', '31'), ('25', '31'), ('26', '29'), ('26', '33'), ('27', '33'), ('28', '31'), ('28', '33'), ('29', '32'), ('29', '33'), ('30', '32'), ('30', '33'), ('31', '32'), ('31', '33'), ('32', '33')]


mygraph = nx.Graph(edgelist)   
    

get_graph_info(mygraph)   



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


############## Community Detection  Girvan-Newman Community Detection Modularity Trend #####################

# function to return a node's neighbor nodes as a list
def get_neighbors(graph, node):
    return [n for n in graph.neighbors(node)] 


# function to return the subgraph containing 2 nodes' common neighbors
def get_common_neighbor_subgraph(graph, source, target):
    nodes = [source, target] + list(nx.common_neighbors(mygraph, source, target))
    return graph.subgraph(nodes)



from networkx.algorithms.community.centrality import girvan_newman
import networkx.algorithms.community as nx_comm

# find communities
girvan_newman_communities = list(girvan_newman(mygraph))


modularity_df = pd.DataFrame([[k+1, round(nx_comm.modularity(mygraph, girvan_newman_communities[k]), 6)]
                for k in range(len(girvan_newman_communities))],
                            columns=["k", "modularity"])
modularity_df.plot.bar(x="k", figsize=(10,6), title="Girvan-Newman Community Detection Modularity Trend");

plt.show()



# function create node color list for less than 7 communities
# when there are more than 6 colors, visualization can be confusing for human
def create_community_node_colors(graph, communities):
    number_of_colors = len(communities[0])
    colors = ["#EF9A9A", "#BA68C8", "#64B5F6", "#81C784",
              "#FFF176", "#BDBDBD"][:number_of_colors]
    node_colors = []
    
    # iterate each node in the graph and find which community it belongs to
    # if the current node is found at a specific community, add color to the 
    # node_colors list
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors



def visualize_communities(graph, communities):
    # create visualization
    node_colors = create_community_node_colors(graph, communities)
    modularity = round(nx_comm.modularity(graph, communities), 6)
    title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
    pos = nx.spring_layout(graph, 
                           k=0.3, iterations=50,
                           seed=2)

    plt.figure(1,figsize=(10,6)) 
    nx.draw(graph,
            pos = pos,
            node_size=1000,
            node_color=node_colors,
            with_labels=True,
            font_size = 20,
            font_color='black')
    plt.title(title)
    plt.show() 

    
visualize_communities(mygraph, girvan_newman_communities[0])




visualize_communities(mygraph, girvan_newman_communities[2])





visualize_communities(mygraph, girvan_newman_communities[3])