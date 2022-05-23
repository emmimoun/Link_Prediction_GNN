from neo4j import GraphDatabase
import networks as nxj
from networks import community
import  numpy as np
from sklearn.svm import SVC
from neo4j import GraphDatabase
import networks as netwXj
from networks import community
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime, time
print ('Last run on: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + repr(time.tzname))

def create_database(mygraph,database_name):
    if database_name in show_databases(mygraph):
        print('Already existe')
        return False
    else :
        create_or_replace_database(mygraph,database_name)
        print(database_name,' created successfully !')
        return True

def get_graph_list(G):
    edges=list(G.edges(data=True))
    nodes=list(G.nodes(data=True))
    return nodes,edges
def use_database(mygraph, graph_name='neo4j'):
    driver = GraphDatabase.driver(uri="bolt://localhost:11003",auth=("emmimoun","emmimoun"),database=graph_name)
    mygraph = nxj.Graph(driver)
    #mygraph.delete_all()
    mygraph.identifier_property = 'identifiant'
    mygraph.relationship_type = '-'
    mygraph.node_label = 'Personne'
    print(graph_name)
    return mygraph

def create_or_replace_database(mygraph,database_name):
    query = "CREATE OR REPLACE DATABASE %s" %database_name
    with mygraph.driver.session() as session:
        session.run(query)
def show_databases(G):
    # doesn't currently support `weight`, `k`, `endpoints`, `seed`
    query = "show databases"
    params = G.base_params()
    with G.driver.session() as session:
        result = [row["name"] for row in session.run(query, params)]
    return result

def txt_to_graph(mygraph,nodes_file_txt="nodes.txt",edges_file_txt="edges.txt"):
    mygraph.delete_all()
    mygraph.identifier_property = 'identifiant'
    mygraph.relationship_type = '-'
    mygraph.node_label = 'Personne'
    FNodes = csv.reader(open(nodes_file_txt),delimiter='\n')
    FEdges = csv.reader(open(edges_file_txt),delimiter='\n')
    mygraph.delete_all()
    for row in FNodes:
        rowlist=row[0].split(' ')
        mygraph.add_node(
            rowlist[0],
            familyName=rowlist[1],
            firstName=rowlist[2],
            dateOfBirth=rowlist[3])
    for row in FEdges:
        rowlist=row[0].split(' ')
        mygraph.add_edge(rowlist[0],rowlist[1],timpstamp=rowlist[2],bool10=rowlist[3])

def graph_to_txt(mygraph,file_edges_from_neo4j="edges_tompon.txt",file_nodes_from_neo4j="nodes_tompon.txt"):
    mynodes,myedges=get_graph_list(mygraph)

    print('graph_to_txt  function')
    ef=open(file_edges_from_neo4j,'w')
    for edge in myedges:
        listAttributeEdge=[edge[0],edge[1]]
        listAttributeEdge.append(edge[2]['timpstamp'])
        listAttributeEdge.append(edge[2]['bool10'])
        print(edge,listAttributeEdge)
        ef.write(''+str(listAttributeEdge[0])+' '+str(listAttributeEdge[1])+' '+str(listAttributeEdge[2])+' '+str(listAttributeEdge[3])+'\n')
        
    ef.close()

    nf=open(file_nodes_from_neo4j,'w')

    for i in list(range(len(mynodes))):
        node=mynodes[i]
        id=node[0]
        #print('id',id)
        attributes=node[1]
        #print('attributes',attributes)
        default = 'Unknown'
        firstName=attributes.get('firstName', default)
        #print('firstName',firstName)
        familyName=attributes.get('familyName', default)
        #print('familyName',familyName)
        dateOfBirth=attributes.get('dateOfBirth', default)
        #print('dateOfBirth',dateOfBirth)
        #print('\n------------------------------\n')
        listAttribute=[id,firstName,familyName,dateOfBirth]
        nf.write(''+str(listAttribute[0])+' '+str(listAttribute[1])+' '+str(listAttribute[2])+' '+str(listAttribute[3])+'\n')
    nf.close()
    print('succes')

def initGraph():
    driver = GraphDatabase.driver(uri="bolt://localhost:11003",auth=("emmimoun","emmimoun"),database='neo4j')
    mygraph = nxj.Graph(driver)
    #mygraph.delete_all()
    mygraph.identifier_property = 'identifiant'
    mygraph.relationship_type = '-'
    mygraph.node_label = 'Personne'
    return mygraph

