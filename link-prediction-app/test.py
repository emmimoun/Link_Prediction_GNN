# test only (import sys;sys.path.append("../"))  #the purpose is to reach to the parent directory
# to fix the default port run $kill $(lsof -ti:7687)
"""
known fails
G.add_node("Betul",age=4)
G.add_node("Betul",age=5) #this does not update the first one

G.nodes['Betul']['age'] = 5 #also does not work

list(G.edges(data=True)) it would be nice to display labels here

G.edges(['Betul','Nurgul']) #FAILS


"""





from neo4j import GraphDatabase

import networks as nxj
from networks import community










#edges= [('sam', 'an'), ('an', 'is') ]
#G = nx.Graph(edges)   
#get_graph_info(G)  
#plt.figure(1,figsize=(10,6)) 
#nx.draw(G, with_labels=True,font_size = 20, font_color='black')
#plt.show()





driver = GraphDatabase.driver(uri="bolt://localhost:11003",auth=("sam","sam"))
mygraph = nxj.Graph(driver)

mygraph.delete_all()


edgelist = [('0', '1'), ('0', '2'), ('0', '3'), ('0', '4'),('20', '32'), ('20', '33'), ('22', '32'), ('22', '33'), ('23', '25'),('15', '33'), ('18', '32'), ('18', '33'), ('19','33'), ('8', '33'), ('9', '33'), ('13', '33'), ('14', '32'), ('14', '33'), ('15', '32'), ('0', '5'), ('0', '6'), ('0', '7'), ('0', '8'), ('0', '10'), ('0', '11'), ('0', '12'), ('0', '13'), ('0', '17'), ('0', '19'), ('0', '21'), ('0', '31'), ('1', '2'), ('1', '3'), ('1', '7'), ('1', '13'), ('1', '17'), ('1', '19'), ('1', '21'), ('1', '30'), ('2', '3'), ('2', '7'), ('2', '8'), ('2', '9'), ('2', '13'), ('2', '27'), ('2', '28'), ('2', '32'), ('3', '7'), ('3', '12'), ('3', '13'), ('4', '6'), ('4', '10'), ('5', '6'), ('5', '10'), ('5', '16'), ('6', '16'), ('8', '30'), ('8', '32'),  ('23', '27'), ('23', '29'), ('23', '32'), ('23', '33'), ('24', '25'), ('24', '27'), ('24', '31'), ('25', '31'), ('26', '29'), ('26', '33'), ('27', '33'), ('28', '31'), ('28', '33'), ('29', '32'), ('29', '33'), ('30', '32'), ('30', '33'), ('31', '32'), ('31', '33'), ('32', '33')]


mygraph.identifier_property = 'identifiant'
mygraph.relationship_type = 'Connait'
mygraph.node_label = 'Personne'

mygraph.add_edges_from(edgelist)


mygraph.add_edges_from([ ('1', 'sami'),('sami', 'anis'), ('anis', 'amine'), ('amine', 'aymen'), ('sami', 'aymen') ,('halim', 'aymen'), ('halim', 'anis')])


def get_graph_info(graph):
    print("number of nodes", graph.__len__())
    print("Available nodes:", list(graph.nodes))
    print("Available edges:", list(graph.edges))


get_graph_info(mygraph)

nxj.draw(mygraph)



betweenness_centrality=nxj.betweenness_centrality(mygraph)
print("betweenness_centrality")
print(betweenness_centrality)


closeness_centrality=nxj.closeness_centrality(mygraph)
print("closeness_centrality")
print(closeness_centrality)


pagerank=nxj.pagerank(mygraph)
print("pagerank")
print(pagerank)


triangles=nxj.triangles(mygraph)
print("triangles")
print(triangles)


clustering=nxj.clustering(mygraph)
print("clustering")
print(clustering)



degree_centrality=nxj.degree_centrality(mygraph)
print("degree_centrality")
print(degree_centrality)


connected_components=list(nxj.community.connected_components(mygraph))
print("Community detection connected_components")
print(connected_components)

label_propagation=list(nxj.community.label_propagation_communities(mygraph))
print("Community detection Label propagation")
print(label_propagation)


#list(nxj.community.label_propagation_communities(mygraph))

#nxj.shortest_path(mygraph, source=1, target=4)










