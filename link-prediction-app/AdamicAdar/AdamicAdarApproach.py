
import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
#from matplotlib.ticker import NullFormatter  
import multiprocessing 
from multiprocessing import Manager
import math
import csv
import json
from matplotlib.ticker import NullFormatter  
from flask.json import jsonify
import scipy.sparse as sp
import matplotlib.patches as mpatches
from operator import itemgetter
import pickle

#dynamic graph
time1=0
Result=[]
train_test_split=None
dynamic=True
#methods 'adamic','jaccard','preferential','resource_allocation'
methods='jaccard'
type_graph='small'

#function graph
def graphe_TO_json(g):
    
    data =  json_graph.node_link_data(g,{"link": "links", "source": "source", "target": "target","weight":"weight"})
    data['nodes'] = [ {"id": i,"degree":g.degree[i],"neighbors":[n for n in g.neighbors(i)]} for i in range(len(data['nodes'])) ]
    data['links'] = [ {"source":u,"target":v,"weight":(g.degree[u]+g.degree[v])/2} for u,v in g.edges ]
    return data


def facebook_graph():
    FielName="C:\\Users\\islem\\Desktop\\AdamicAdar\\facebook.txt"
    Graphtype=nx.Graph()
    g= nx.read_edgelist(FielName,create_using=Graphtype,nodetype=int)
    
    return graphe_TO_json(g) 
#---------------------


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.01, prevent_disconnect=True):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # Remove diagonal elements
    #adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    #adj.eliminate_zeros()
    # Check that diag is zero:
    #assert np.diag(adj.todense()).sum() == 0
    g = nx.from_scipy_sparse_matrix(adj)
    
    #orig_num_cc = nx.number_connected_components(g)
    adj_triu = sp.triu(adj,k=1) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    
    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        if len(test_edges) == num_test :
            break
        node1 = edge[0]
        node2 = edge[1]  
        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if  nx.is_isolate(g,node1) or nx.is_isolate(g,node2) :
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)
     # Both edge lists full --> break loop
    if (len(test_edges) < num_test):
        print ("WARNING: not enough removable edges to perform full train-test split!")
        print ("Num. (test, val) edges requested: :",num_test)
        print ("Num. (test, val) edges returned: (", len(test_edges), ")")

    
    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)
    # assert: test, val, train positive edges disjoint
    assert test_edges.isdisjoint(train_edges)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)
    
    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])   
    # NOTE: these edge lists only contain single direction of edge!
    print('end split test')
    return adj_train, train_edges,test_edges,test_edges_false

def prediction_link(g,ebunch,method):
    if g.is_directed(): # Only works for undirected graphs
        g= g.to_undirected()
   # Unpack input
    matrix =[]
    matrix_normalize =[]
    max=0
    if method=='adamic':
        for u, v, p in nx.adamic_adar_index(g, ebunch=ebunch): # (u, v) = node indices, p = resource_allocation index
            tupl=(u,v,p)
            matrix.append(tupl) 
            if max<p:
               max=p
    elif method=='jaccard':
        for u, v, p in nx.jaccard_coefficient(g, ebunch=ebunch): # (u, v) = node indices, p = resource_allocation index
            tupl=(u,v,p)
            matrix.append(tupl) 
            if max<p:
               max=p
    elif method=='preferential':
        for u, v, p in nx.preferential_attachment(g, ebunch=ebunch): # (u, v) = node indices, p = resource_allocation index
            tupl=(u,v,p)
            matrix.append(tupl) 
            if max<p:
               max=p
    elif method=='resource_allocation':
        for u, v, p in nx.resource_allocation_index(g, ebunch=ebunch): # (u, v) = node indices, p = resource_allocation index
            tupl=(u,v,p)
            matrix.append(tupl) 
            if max<p:
               max=p
    else:
        return max,matrix_normalize

     # Normalize matrix
    if max==0:
       print(' predection none')
    else:
       for u, v, p in matrix: #u source v target p probability
          d=p/max
          tupl=(u,v,d)
          matrix_normalize.append(tupl) 
         
    return max,matrix_normalize


    
def adamic_adar_sc(g, ebunch):
    return prediction_link(g,ebunch,'adamic')

def split_test(test,frac):
    test_split=[]
    if len(test)<frac:
        frac=len(test)
    for i in reversed(range(frac)):
        test_split.append(test.pop(i))
   
    return test_split

def get_ebunch(train_test_split):
    adj_train, train_edges,test_edges,test_edges_false = train_test_split
 
    test_edges_list = test_edges.tolist() # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
   
    return (test_edges_list + test_edges_false_list)

def new_Link(matrix,ebunch):
    new_link=[]
    #matrix symetric
    for u,v,p in matrix:
          Tuple=(u,v)
          #probability of new link matrix[edges[0]][edges[1]]
          if p>0:
              if p>np.random.rand(): 
                  new_link.append(Tuple)
              else:
                  ebunch.append(Tuple)
          else:
            ebunch.append(Tuple)
    
    return new_link         

def test(train_test_split,method='adamic',split=0.025):
    result=[]
    adj_train, train_edges,test_edges,test_edges_false=train_test_split
    g= nx.Graph(adj_train)
    #40% of edges for test (hidden edges) 
    ebunch=get_ebunch(train_test_split)
    #1% links added for each iteration
    frac=int(len(ebunch)*split)
    print("init---------------")
    print("links added for each iteration:",frac)
    print("edges:",len(g.edges))
    print("tets_edges:",len(test_edges))
    print("tets_edges_false:",len(test_edges_false))
    print("method:",method)
    print('#-------------------')

    while len(ebunch)>0: 
      ebunchi=split_test(ebunch,frac=frac)
           # matrix=>1% of test edges (tuple(u,v,p))
      max,matrix=adamic_adar_sc(g,ebunchi)
           #new link added for networks
      new_link=new_Link(matrix,ebunch)
          #update networks
      g.add_edges_from(new_link)     
      result.append({'matrix':matrix,'new_link':new_link,'method':method})
          #test to avoid infinite loop
      if max==0:
             max,matrix=adamic_adar_sc(g,ebunch) 
             if max==0:
                 print('end prediction')
                 break
   
    return result


g=facebook_graph()

FielName="C:\\Users\\islem\\Desktop\\AdamicAdar\\0.edges"

Graphtype=nx.Graph()

g= nx.read_edgelist(FielName,create_using=Graphtype,nodetype=int)

print(type(g))





adj = nx.adjacency_matrix(g)

print("the original grpah shape number of nodes adj matrix",adj.shape)



train_test_split=mask_test_edges(adj,test_frac=0.2)

adj_train,train_edges,test_edges,test_edges_false=train_test_split

print("edges:",len(g.edges))
print("tets_edges:",len(test_edges))
print("tets_edges_false:",len(test_edges_false))
print(test_edges)




Result=test(train_test_split,'adamic')




with open('Adamic_Adar_Aproach_Result.json', "w") as f:
    json.dump(Result, f)



