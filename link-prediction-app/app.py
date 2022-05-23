
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

import os
import json
import torch
import torch.nn as nn
from encoders import Encoder
from aggregators import MeanAggregator
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn import metrics
import random
import networkx as nx
from networkx.algorithms import community
import matplotlib

from networks import graph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import connect_to_graph

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'python_login'  


#    - the following will be our login page, which will use both GET and POST requests
# Intialize MySQL
mysql = MySQL(app)



@app.route("/index")
def index():
	
	return render_template('index.html',session=session)
	

@app.route("/")
def homefirst():
	# return render_template('index.html')
	return redirect(url_for('login'))
@app.route("/test1")
def test1():
	return render_template('testload.html')

@app.route("/log")
def log():
	return render_template('log.html')   


@app.route('/update_json', methods=['GET', 'POST'])
def update_json(): 
	#print(request.json['nameF'])
	print('Hiiiii')
	#print(format(os.getcwd()))
	print('hello')
	'''with open("datasets/feature_Graph_input.txt", "w+") as outfile1:
		Lines=request.json['nameT']
		outfile1.writelines(Lines)'''

	Lines=request.json['Lines']
	#Lines.split("#")
	#for line in Lines:
	#	print(line)
	#Lines.remove("\n")
	print(Lines)
	with open("datasets/feature_Graph_input.txt", "w+") as outfile1:
		outfile1.writelines(Lines)
	'''with open("datasets/Graph_input.json", "w+") as outfile:
         json.dump(request.json['Lines'], outfile)'''

	print("done")
	return 'ok'





@app.route('/get_graphs', methods=['GET', 'POST'])
def get_graphs(): 
	list_graph=connect_to_graph.show_databases(connect_to_graph.initGraph())
	print(list_graph)
	dict_graph={'list_graph':list_graph}

	return  dict_graph 




@app.route('/load_graph_from_database', methods=['GET', 'POST'])
def load_graph_from_database(): 
	print('load_graph_from_database','submit ================')
	print('le graph est',request.json['graph_name'])
	# mygraph=connect_to_graph.use_database(mygraph,request.json['graph_name'])
	mygraph=connect_to_graph.initGraph()
	connect_to_graph.graph_to_txt(mygraph,file_edges_from_neo4j='datasets/Graph_input.txt',file_nodes_from_neo4j='datasets/feature_Graph_input.txt')

	return  'ok' 






@app.route('/add_node')
def add_node(): 
	f= open('datasets/Graph_input.json')
	f1= open('datasets/feature_Graph_input.txt')
	file = json.load(f)
	file1=f1.readlines()
	#print('file1= '+str(len(file1)))
	lines=""
	for line in file1:
		line=line.rstrip('\n')
		lines=lines+"#"+line

	return  lines 


@app.route('/save_in_file_png')
def save_in_file_png(): 
	 input_graph="datasets/Graph_input.txt"
	 #input_graph="datasets/"+request.json['nameOfFile']
	 Links=[]
	 label_test=[]

	 with open(input_graph) as fp:
		 for i, line in enumerate(fp):
			 temp=line.split(" ")
			 left=int(temp[0])
			 right=int(temp[1])
			 if ([left, right] not in Links ) and ([right, left] not in Links):
				 Links.append([left, right])
	 List = []
	 

	 for j in range (0,len(Links)) :
		 print(Links[j])
		 List.append(Links[j][0])
		 List.append(Links[j][1])


	 Set = [] 
	 for i in List :
		 if i not in Set: 
			 Set.append(i)



	 G = nx.Graph()
	 for item in Set:
	 	G.add_node(item)

	 for link_item in Links:
	 	G.add_edge(link_item[0],link_item[1])
	 nx.draw(G)
	 plt.show(block=False)
	 plt.savefig("Graph.png", format="PNG")
	 os.startfile('Graph.png')
	 print(os.path.abspath(os.getcwd()))

	 return 'ok'


@app.route('/save_in_file_pdf')
def save_in_file_pdf(): 
	 input_graph="datasets/Graph_input.txt"
	 #input_graph="datasets/"+request.json['nameOfFile']
	 Links=[]
	 label_test=[]

	 with open(input_graph) as fp:
		 for i, line in enumerate(fp):
			 temp=line.split(" ")
			 left=int(temp[0])
			 right=int(temp[1])
			 if ([left, right] not in Links ) and ([right, left] not in Links):
				 Links.append([left, right])
	 List = []
	 

	 for j in range (0,len(Links)) :
		 print(Links[j])
		 List.append(Links[j][0])
		 List.append(Links[j][1])


	 Set = [] 
	 for i in List :
		 if i not in Set: 
			 Set.append(i)



	 G = nx.Graph()
	 for item in Set:
	 	G.add_node(item)

	 for link_item in Links:
	 	G.add_edge(link_item[0],link_item[1])
	 nx.draw(G)
	 plt.show(block=False)
	 plt.savefig("Graph_pdf.pdf", format="PDF")
	 time.sleep (2)
	 os.startfile('Graph_pdf.pdf')


	 return 'ok'

@app.route('/save_in_file_jpeg')
def save_in_file_jpeg(): 
	 input_graph="datasets/Graph_input.txt"
	 #input_graph="datasets/"+request.json['nameOfFile']
	 Links=[]
	 label_test=[]

	 with open(input_graph) as fp:
		 for i, line in enumerate(fp):
			 temp=line.split(" ")
			 left=int(temp[0])
			 right=int(temp[1])
			 if ([left, right] not in Links ) and ([right, left] not in Links):
				 Links.append([left, right])
	 List = []
	 

	 for j in range (0,len(Links)) :
		 print(Links[j])
		 List.append(Links[j][0])
		 List.append(Links[j][1])


	 Set = [] 
	 for i in List :
		 if i not in Set: 
			 Set.append(i)



	 G = nx.Graph()
	 for item in Set:
	 	G.add_node(item)

	 for link_item in Links:
	 	G.add_edge(link_item[0],link_item[1])
	 nx.draw(G)
	 plt.show(block=False)
	 plt.savefig("Graph_jpeg.jpeg", format="JPEG")
	 os.startfile('Graph_jpeg.jpeg')

	 return 'ok'





@app.route('/save_in_file_svg', methods=['POST'])
def save_in_file_svg(): 
	text=request.json['svg_text']
	svg_file = open('Graph_svg.txt', 'w+')
	svg_file.write(text)

	svg_file.close()
	os.startfile('Graph_svg.txt')

	
	return 'ok'



@app.route('/generate_graph', methods=['POST'])
def generate_graph(): 
	node_num=request.json['node_num']
	edge_num=request.json['edge_num']
	methode_gen=request.json['methode_gen']


	#print("generer graph : node ="+node_num+", edge ="+edge_num+" , method:"+methode_gen)
	if methode_gen=="powerlaw_cluster_graph":
		Graph=nx.powerlaw_cluster_graph(int(node_num), int(edge_num), 0.5, seed=None)

	else:
		if methode_gen=="dual_barabasi_albert_graph":
			Graph=nx.dual_barabasi_albert_graph(int(node_num), int(edge_num), int(edge_num), 0.5, seed=None, initial_graph=None)
	print(Graph)
	new_graph = open('datasets/new_graph.txt', 'w+')

	for line in nx.generate_edgelist(Graph, data=False):
		print(line)
		new_graph.write(line+"\n")
	new_graph.close()

	return 'ok'










@app.route('/add_link')
def add_link(): 
	f= open('datasets/Graph_input.json')
	file = json.load(f)
	return  file  


class SupervisedGraphSage(nn.Module):

	def __init__(self, num_classes, enc,name):
		super(SupervisedGraphSage, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.name=name
		if name!="activation" and name!="origin":
			self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		else:
			self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim*2))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		#original
		#embeds = torch.cat((embeds[:,0],embeds[:,1]),0).unsqueeze(1)
		#mean
		if self.name=="mean":
			embeds=(embeds[:,0]+embeds[:,1])/2
			embeds = embeds.unsqueeze(1)
		#hadamard
		elif self.name=='had':
			embeds=embeds[:,0].mul(embeds[:,1])
			embeds = embeds.unsqueeze(1)
		#weight-l1
		elif self.name=="w1":
			embeds=torch.abs(embeds[:,0]-embeds[:,1])
			embeds = embeds.unsqueeze(1)
		#weight-l2
		elif self.name=="w2":
			embeds=torch.abs(embeds[:,0]-embeds[:,1]).mul(torch.abs(embeds[:,0]-embeds[:,1]))
			embeds = embeds.unsqueeze(1)
		#activation
		elif self.name=='activation':
			embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)
			embeds = F.relu(embeds)
		elif self.name=='origin':
			embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)

		scores = self.weight.mm(embeds)

		return scores.t()

	def loss(self, labels,agg1,agg2,edge):
		agg1.time = edge[0][2]
		agg2.time = edge[0][2]
		scores = F.softmax(self.forward([edge[0][0],edge[0][1]]),dim=1)
		predict_y = []
		predict_y.append(scores.cpu().detach().numpy()[0][1])
		for i in range(1, len(edge)):
			print(i)
			agg1.time = edge[i][2]
			agg2.time = edge[i][2]
			#print("edge:"+str(edge[i][0])+","+str(edge[i][1]))
			temp = self.forward([edge[i][0], edge[i][1]])
			temp=F.softmax(temp,dim=1)
			predict_y.append(temp.cpu().detach().numpy()[0][1])
			scores=torch.cat((scores, temp), 0)
		return self.xent(scores, labels),predict_y

@app.route('/transform', methods=['GET', 'POST'])
def transform():

	#####################################
	 
	 def find_all_other_edges (list_edges):
		 list_edges_left=[]
		 Set_nodes=[]
		 for i in range(0,len(list_edges)):
			 if(list_edges[i][0] not in Set_nodes):
				 Set_nodes.append(list_edges[i][0])
			 if(list_edges[i][1] not in Set_nodes):
				 Set_nodes.append(list_edges[i][1])
			 print (i)
		 for j in range(0,len(Set_nodes)):
			 for k in range(j+1,len(Set_nodes)):
				 if((j is not k) and ([Set_nodes[j],Set_nodes[k]] not in list_edges) and ([Set_nodes[k],Set_nodes[j]] not in list_edges)):
					 list_edges_left.append([Set_nodes[j],Set_nodes[k],k%20+1])
					 '''if(len(list_edges_left)>10):
					 	break'''

		 return list_edges_left

	# def find_index_node_in_json_file(json,index_node):


	 name="origin"
	 adj_list={}
	 adj_time={}
	 with open('feature_random_contact.txt') as fp:
		 feat_data = np.genfromtxt(fp)[:, 1:]
	 for i in range(0,len(feat_data)):
		 adj_list[i] = []
		 adj_time[i] = []
	
	 
	#  name="origin"
	#  adj_list={}
	#  adj_time={}
	#  with open('feature_random_contact.txt') as fp:
	# 	 feat_data = np.genfromtxt(fp)[:, 1:]
	#  with open('datasets/feature_Graph_input.txt')as feat_file:
	# 	 list_of_id_feat=[]
	# 	 for i, line in enumerate(feat_file):
	# 		 temp=line.split(" ")
	# 		 id_=int(temp[0])
	# 		 list_of_id_feat.append(id_)

	#  for i in list_of_id_feat:#range(0,len(feat_data)):
	# 	 adj_list[i] = []
	# 	 adj_time[i] = []
	#  print('feat_data')
	#  print(feat_data)
	 




	 edge_test = []
	 edge_test_1 = []
	 dimension=128
	
	 
	#loss, predict_y = graphsage2.loss(Variable(torch.LongTensor(label_test)), agg1, agg2, edge_test)

	####################################



	 #print(request.json['nameOfFile'])
	 input_graph="datasets/Graph_input.txt"
	 #input_graph="datasets/"+request.json['nameOfFile']
	 Links=[]
	 label_test=[]
	 with open(input_graph) as fp:
		 for i, line in enumerate(fp):
			 temp=line.split(" ")
			 left=int(temp[0])%250
			 right=int(temp[1])%250
			 edge_test.append([left, right,int(temp[2])])
			 edge_test_1.append([left+1, right+1,int(temp[2])])
			 #label_test.append(1)
			 if left not in adj_list:
				 adj_list[left]=[right]
				 adj_time[left]=[int(temp[2])]
			 else:
				 adj_list[left].append(right)
				 adj_time[left].append(int(temp[2]))
			 
			 
			 #label_test.append(1)
			 if right not in adj_list:
				 adj_list[right]=[left]
				 adj_time[right]=[int(temp[2])]
			 else:
				 adj_list[right].append(left)
				 adj_time[right].append(int(temp[2]))
			 if ([left, right] not in Links ) and ([right, left] not in Links):
				 Links.append([left, right])
			 #if i == 30:
				 #break
			


	 
	 print('self.adj_lists',adj_list)
	 print('#############################')
	 print('self.adj_time',adj_time)

	 features = nn.Embedding(len(feat_data), 1000)
	 features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
	 agg1 = MeanAggregator(features, cuda=True)
	 agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
	 enc1 = Encoder(features, 1000, dimension, adj_list,adj_time, agg1, gcn=True, cuda=False)
	 enc1.num_samples = 5
	 enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, dimension, adj_list, adj_time,agg2,base_model=enc1, gcn=True, cuda=False)
	 enc2.num_samples = 5
	 graphsage2 = SupervisedGraphSage(2, enc2,name)
	 graphsage2.load_state_dict(torch.load('model/TDGNN.pth'))
	 print('ffffffffffffffff  111')
	 #graphsage2.eval()

	 #edge_test_bar=find_all_other_edges(edge_test) 
	 

	 #print('**********************  printing the non existing edges ***************************')
	 #for element in edge_test_bar:
		 #print(element)


	 #predict_y=graphsage2.predict(edge_test)
	 #edge_for_prediction=[[1,3,2],[3,4,2],[1,4,2]]

	 List = []
	 

	 for j in range (0,len(Links)) :
		 print(Links[j])
		 List.append(Links[j][0])
		 List.append(Links[j][1])

	 edge_for_prediction=[]
	 #first choise
	 edge_for_prediction=find_all_other_edges(Links)


	 print("edge_for_prediction")
	 print(edge_for_prediction)
	 #print("edge_for_prediction")
	 dict_node={}

	 Set = [] 
	 for i in List :
		 if i not in Set: 
			 Set.append(i)
	 #input_graph="datasets/"+request.json['nameOfFile']
	 

	 with open("datasets/feature_Graph_input.txt") as fginput:
		 for i, line in enumerate(fginput):
			 temp=line.split(" ")
			 print("test set :::::: "+str(temp[0]))
			 if((temp[0]!="" and temp[0]!="\n") and (int(temp[0]) not in Set)):
			 	Set.append(int(temp[0]))


	 







	 #second choice
	 '''for jj in range(0,int(len(Set)/2)):
	 	node1=random.choice(Set)
	 	node2=random.choice(Set)
	 	if(([node1,node2] not in Links) and ([node2,node1] not in Links)):
	 		edge_for_prediction.append([node1,node2,2])'''


	 print("edge_for_prediction")
	 print(edge_for_prediction)
	 print("edge_for_prediction")
	 # we can load real value in  label test

	 for jjj in range(0,len(edge_for_prediction)):
	 	label_test.append(1)

	 	


	 #label_test=[1,1,1]
	 loss, predict_y = graphsage2.loss(Variable(torch.LongTensor(label_test)), agg1, agg2, edge_for_prediction)
	 print('ffffffffffffffff')
	 predict_y1=[]
	 
	 for i in range(0,len(predict_y)):
		 if predict_y[i] > 0.3:
			 predict_y1.append(1)
		 else:
			 predict_y1.append(0)

	 print('yyyyyyyyyyyyyyyyyyyy')
	 new_links=[]
	 for j in range(0,len(predict_y)):
		 print(predict_y1[j]," is the edge = ",edge_for_prediction[j][0]," , ",edge_for_prediction[j][1])
		 #if ([edge_for_prediction[j][0],edge_for_prediction[j][1]] not in Links) and ([edge_for_prediction[j][1],edge_for_prediction[j][0]] not in Links):
		 if(predict_y1[j]==1):
		 	new_links.append([edge_for_prediction[j][0],edge_for_prediction[j][1],edge_for_prediction[j][2]])

	 
	 def occ_cal(Word,List):
			cpt=0
			for index in range(0,len(List)):
				if(Word==List[index][0] or Word==List[index][1]):
					cpt=cpt+1
			return cpt
	 
	 
	 
	 fjson = open('datasets/Graph_input.json', 'w+')
	 ##
	 fjson_out = open('datasets/Graph_output.json', 'w+')
	 fjson_out.write('{"directed": false, "multigraph": false, "graph": {}, "nodes": [')
	 ##
	 fjson.write('{"directed": false, "multigraph": false, "graph": {}, "nodes": [')
	 print('{"directed": false, "multigraph": false, "graph": {}, "nodes": [')
	 #featurejson=
	 feat_name_dict={}
	 feat_last_name_dict={}
	 feat_birthday_dict={}
	 feature_file_json="datasets/feature_Graph_input.txt"
	 with open(feature_file_json) as feajson:
		 for i, line in enumerate(feajson):
		 	if(line!="" and line!="\n"):
		 		temp=line.split(" ")
		 		print(temp)
		 		feat_name_dict[temp[0]]=temp[1]
		 		feat_last_name_dict[temp[0]]=temp[2]
		 		feat_birthday_dict[temp[0]]=temp[3].rstrip('\n')
			 

	 print('Set')
	 print(Set)
	 print('feat_name_dict keys')
	 print(feat_name_dict)

	 for index in range(0,len(Set)) :
		 line1=''
		 #print( 'index = ' +str(index) )

		 #print('Set[index]')
		 #print(Set[index])
		 #print('feat_name_dict keys')
		 #print(feat_name_dict.keys())


		 if str(Set[index]) in feat_name_dict:

		 	name=str('"'+feat_name_dict[str(Set[index])]+'"')
		 else:
		 	name='null'


		 if str(Set[index]) in feat_last_name_dict :
		 	last_name=str('"'+feat_last_name_dict[str(Set[index])]+'"')
		 else:
		 	last_name='null'

		 if str(Set[index]) in feat_birthday_dict :
		 	birthday_date=str('"'+feat_birthday_dict[str(Set[index])]+'"')
		 else:
		 	birthday_date='null'

		 print("name :"+ name)
		 print("last name:"+ last_name)
		 print("birthday_date:" +birthday_date )



		 if(index==len(Set)-1):
			 line1='{"name": '+str(Set[index])+' , "degree":'+ str(occ_cal(Set[index],Links))+' , "info_name":'+ name+' , "info_last_name":'+last_name+' , "birthday_date":'+birthday_date+' } '
		 else:
			 line1='{"name": '+str(Set[index])+' , "degree":'+ str(occ_cal(Set[index],Links))+' , "info_name":'+ name+' , "info_last_name":'+last_name+' , "birthday_date":'+birthday_date+'}, '
		 dict_node[Set[index]]=index
		 fjson.write(line1)
		 fjson_out.write(line1)


	 fjson.write('],  "links": [') 
	 fjson_out.write('],  "links": [')
	 for j in range (0,len(Links)) :
		
			# print(str(predict_y1[j]))
			 fjson_out.write('{"source": '+str(dict_node[Links[j][0]])+', "target": '+str(dict_node[Links[j][1]])+' ,"instance": 0}, ')
			 if(j==len(Links)-1):
				 line='{"source": '+str(dict_node[Links[j][0]])+', "target": '+str(dict_node[Links[j][1]])+', "instance": 0}'
			 else:
				 line='{"source": '+str(dict_node[Links[j][0]])+', "target": '+str(dict_node[Links[j][1]])+' ,"instance": 0}, '
			
			 fjson.write(line)
			 
			#tmmp=line.split(,)

			 # print(line) # <----> correct




	 for ii in range (0,len(new_links)) :
		
			# print(str(predict_y1[j]))
			 if(ii==len(new_links)-1):
				 line='{"source": '+str(dict_node[new_links[ii][0]])+', "target": '+str(dict_node[new_links[ii][1]])+', "instance": '+str(new_links[ii][2])+' }'
			 else:
				 line='{"source": '+str(dict_node[new_links[ii][0]])+', "target": '+str(dict_node[new_links[ii][1]])+' ,"instance": '+str(new_links[ii][2])+' }, '
			
			 fjson_out.write(line)


	 print(']}')
	 fjson.write(']}')

	 print(dict_node)
	 fjson.close()
	 fjson_out.write(']}')
	 fjson_out.close()




	 statistics_dict={}
	 G = nx.Graph()
	 for item in Set:
	 	G.add_node(item)

	 for link_item in Links:
	 	G.add_edge(link_item[0],link_item[1])


	 
	 print("number of nodes= "+str(G.number_of_nodes()))


	 #get_graph()
	 
	 '''
	 
	 list_edges=[[1,2],[2,3],[2,4]]

	 list_left= find_all_other_edges(list_edges)
	 print('test function find_all_other_edges')
	 for element in list_left:
		print(element)
	 '''
	 dict_new_links={}

	 for item in new_links:
	 	dict_new_links[item[2]]=0

	 for item in new_links:
	 	dict_new_links[item[2]]=dict_new_links[item[2]]+1

	 #dict_result_={"dict_new_links":dict_new_links}

	 max_instance=max(dict_new_links.keys())
	 #last_value=len(Links)
	 init_numb_links=len(Links)
	 cont_dict_new_links={}
	 dict_new_links_tmp={}
	 for i in range(0,max_instance):
	 	if i in dict_new_links.keys():
	 		dict_new_links_tmp[i]=dict_new_links[i]
	 	else:
	 		dict_new_links_tmp[i]=0


	 print(dict_new_links_tmp)
	 cont_dict_new_links[0]=init_numb_links
	 for i in range(1,max_instance):
	 	cont_dict_new_links[i]=dict_new_links_tmp[i]+cont_dict_new_links[i-1]


	 if max_instance<100 :
	 	max_value=max(cont_dict_new_links.values())
	 	for j in range(max_instance,100):
	 		cont_dict_new_links[j]=max_value
	 print(cont_dict_new_links)








	 '''for i in range(0,max_instance):
	 	if i not in dict_new_links.keys():
	 		cont_dict_new_links[i]=last_value
	 	else:
	 		cont_dict_new_links[i]=dict_new_links[i]+init_numb_links
	 		last_value=cont_dict_new_links[i]
	 	
	 print(cont_dict_new_links)'''
	 f_plot_links = open('datasets/f_plot_links.txt', 'w+')
	 for key, value in cont_dict_new_links.items():
	 	f_plot_links.write(str(key)+" "+str(value)+"\n")
	 f_plot_links.close()

	 



	 return "ok"









@app.route('/get_graph')
def get_graph():
	
	 input_graph="datasets/Graph_input.txt"
	 #input_graph="datasets/"+request.json['nameOfFile']
	 Links=[]
	 label_test=[]

	 with open(input_graph) as fp:
		 for i, line in enumerate(fp):
			 temp=line.split(" ")
			 left=int(temp[0])
			 right=int(temp[1])
			 if ([left, right] not in Links ) and ([right, left] not in Links):
				 Links.append([left, right])
	 List = []
	 

	 for j in range (0,len(Links)) :
		 print(Links[j])
		 List.append(Links[j][0])
		 List.append(Links[j][1])


	 Set = [] 
	 for i in List :
		 if i not in Set: 
			 Set.append(i)



	 G = nx.Graph()
	 for item in Set:
	 	G.add_node(item)

	 for link_item in Links:
	 	G.add_edge(link_item[0],link_item[1])
	 number_of_nodes=G.number_of_nodes()
	 density=nx.density(G)
	 diameter=nx.diameter(G, e=None, usebounds=False)
	 list_triangles=nx.triangles(G, nodes=None)
	 number_triangles=0
	 for key in list_triangles:
	 	number_triangles=number_triangles+list_triangles[key]


	 avg_triangles=sum(list_triangles.values())/len(list_triangles.values())
	 max_triangles=max(list_triangles.values())
	 k_core=nx.core_number(G)
	 max_k_core=max(k_core.values())
	 max_betweenness=max(nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None))

	 pageranks=nx.pageranks = nx.pagerank(G)
	 max_pagerank=max(pageranks.values())
	 mean_distance=nx.average_shortest_path_length(G)

	 assortativity=abs(nx.degree_assortativity_coefficient(G, x='out', y='in', weight=None, nodes=None))
	 all_clique=nx.enumerate_all_cliques(G)
	 #print("assortativity = "+str(assortativity))

	 partition = community.greedy_modularity_communities(G)
	 number_of_communities=len(partition)
	 #print("number_of_communities= "+str(number_of_communities))







	 '''
	 switcher = {
 		'g1':'datasets\g1.json',
		'g2':'datasets\G2.json',
		'g3':'datasets\g3.json'
	 }
	 graph_name = str(request.data)
	 print ("***************************** "+graph_name+str(type(graph_name)) )
	 f = open(switcher.get(graph_name))
	 '''
	 dict_plot_links={}
	 with open("datasets/f_plot_links.txt") as fptxt:
		 for i, line in enumerate(fptxt):
			 temp=line.split(" ")
			 dict_plot_links[int(temp[0])]=int(temp[1])

	 f= open('datasets/Graph_input.json')
	 result = json.load(f)
	 dict_result={"graph":result, "dict_plot_links":dict_plot_links ,"number_of_nodes":number_of_nodes,"density":"{:.3f}".format(density),"number_triangles": number_triangles/3, "diameter":diameter,"max_k_core":max_k_core,"max_betweenness":max_betweenness,"max_pagerank":"{:.2f}".format(max_pagerank),"mean_distance":"{:.2f}".format(mean_distance),"assortativity":"{:.2f}".format(assortativity),"number_of_communities":number_of_communities,"avg_triangles":avg_triangles,"max_triangles":max_triangles}
	 return  dict_result  

@app.route('/run_linkPrediction')  
def run():
	f = open('datasets/Graph_output.json')
	result  = json.load(f)
	return  json.dumps(result) 


























































@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = '';session['loggedin'] = False
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
			
            # Redirect to home page
            return redirect(url_for('profile'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', False)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


# http://localhost:5000/pythonlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # nchofo ida account yexisty b MySQL --- donc POST ( form mriigla )
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        #  variables username password email
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # nchofo ida account yexisty b MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form khawiya... (makan lah ndiiro POST)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)



##add_graph
@app.route('/add_graph', methods=['GET', 'POST'])
def add_graph():
    # Output message if something goes wrong...
    msg = '';print(request.form);print('username',session)
	#;print('graph',request.form	)
    
	# nchofo ida account yexisty b MySQL --- donc POST ( form mriigla )
	
    if request.method == 'POST' and 'nodes_file' in request.form and 'graphname' in request.form and 'edges_file' in request.form:
        #  variables username password email
        nodes_file = request.form['nodes_file']
        edges_file = request.form['edges_file']
        graphname = request.form['graphname']
        # nchofo ida account yexisty b MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM base_de_donnes_de_graphe WHERE nom_de_base_de_donnes = %s', (graphname,))
        graph = cursor.fetchone();print(graph)
        # If account exists show error and validation checks
        if graph:
            msg = 'Graph Name already exists!'
        elif not re.match(r'[A-Za-z0-9]+', graphname):
            msg = 'Graph Name must contain only characters and numbers!'
        elif not graphname or not nodes_file or not edges_file:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            #cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form khawiya... (makan lah ndiiro POST)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('add_graph.html', msg=msg)


# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/profile')
def profile():
    # nchoofo ida rana m logyiin
    if 'loggedin' in session:
        # nsha9o info ta3 account ndirooha f profile
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()

		        
        cursor.execute('SELECT * FROM historique h, accounts a where a.id = h.id')
        # Fetch one record and return result
        historique = list(cursor.fetchall());cursor.execute('SELECT * FROM base_de_donnes_de_graphe g, accounts a where a.id = g.id')
		# print(historique);
        # Fetch one record and return result
        graphs = list(cursor.fetchall());#print(graphs)
		 
        # display profile html
        return render_template('profile.html', account=account,historique=historique,graphs=graphs)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))





























# ======== Main ============================================================== #
if __name__ == "__main__":
	app.run(debug=True, use_reloader=True, host="localhost")


