#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:33:46 2018

@author: wendymy
"""
#%%
import networkx as nx
import pandas as pd
import numpy as np
import os
from operator import itemgetter

os.chdir("/Users/wendymy/Documents/SI671/Homework2")
# load data 
network = pd.read_csv('network.tsv', header = None, delimiter='\t', encoding='utf-8')
test_dat = pd.read_csv("unlabeled-vertices.test.txt", header = None)

# create graph
# network_lst = list(zip(network[0], network[1]))
# G = nx.Graph(network_lst)
# nx.write_gpickle(G,"G.gpickle")
G_comp = nx.read_gpickle("G.gpickle")

#%%
# test the algorithms
network_removed = network.iloc[:30000:,:]
# network_selected = network.iloc[30000:,:]
# network_selected_lst = list(zip(network_selected[0], network_selected[1]))
# G_selected = nx.Graph(network_selected_lst)
# nx.write_gpickle(G_selected,"G_selected.gpickle")
G_selected = nx.read_gpickle("G_selected.gpickle")

# prediction algorithm evaluation function
def pred_eval(func, graph):
    preds_lst = list(func(graph))
    nodes_lst = [tuple(pred[0],pred[1]) for pred in preds_lst if pred[2] != 0]
    network_removed_lst = list(zip(network_removed[0], network_removed[1]))
    common_edges = set(network_removed_lst).intersection(nodes_lst)
    
    return(len(common_edges))
#%%
def pred(func, graph):
    preds_lst = list(func(graph))
    output_lst = [pred for pred in preds_lst if pred[2] != 0]
    output = output_lst.sort(key=itemgetter(2), reverse = True).iloc[:50000,:]
    df = pd.DataFrame.from_records(output)

    return (df)

#%%
# Part 2
# load packages
import numpy as np
import pandas as pd
import networkx as nx
from operator import itemgetter
from collections import Counter
import pickle

# build whole graph
network = pd.read_csv('network.tsv', header = None, delimiter='\t', encoding='utf-8')
network_lst = list(zip(network[0], network[1]))
G = nx.Graph(network_lst)

train = pd.read_csv("labeled-vertices.train.tsv", header = None, delimiter='\t', encoding='utf-8', 
                   names = ["node", "attr"])
test = pd.read_csv('unlabeled-vertices.test.txt', header = None, names = ["node"])

# find neighbors with radius = 2
test["neighbors"] = np.zeros(len(test.node))
test["neighbors_of_neighbors"] = np.zeros(len(test.node))
test["all_neighbors"] = np.zeros(len(test.node))

test = test.astype('object')
for i in range(len(test.node)):
    neighbors = set(G.neighbors(test.node[i]))
    test.loc[i,"neighbors"] = neighbors
    for n in neighbors:
        neighbors_of_neighbors = set(G.neighbors(n))
        all_neighbors = neighbors.union(neighbors_of_neighbors) - set([test.node[i]])
        test.loc[i,"neighbors_of_neighbors"] = all_neighbors 
    test.loc[i,"all_neighbors"] = neighbors.union(all_neighbors)

# with open('test_neighbors', 'wb') as fp:
#    pickle.dump(test, fp) 

#with open ('test_neighbors', 'rb') as fp:
#    test = pickle.load(fp)

#%%
# Use multiprocessing to calculate the length of neighbors
from multiprocessing import Pool
import numpy as np
import pandas as pd
from global_imports import test, worker_function

test = pd.read_csv("new_test.csv", index_col=0)
test.columns = ["node", "neighbors", "neighbors_of_neighbors", "all_neighbors"]

poolsize = 6 # number of CPU cores * 2
p = Pool(poolsize)

#def worker_function(i_values): # need to import from global_imports
#    
#    return([len(eval(i)) for i in i_values])

tasks = np.array_split(test.all_neighbors.values, poolsize) # break it up into 6 chunks
res = p.imap(worker_function, tasks) # tell the chunks to work on the worker_function
flatten_res = [item for sublist in res for item in sublist] # combine to one list 

test['n_len'] = flatten_res
test["pred"] = np.zeros(len(test.node))
test["f1score"] = np.zeros(len(test.node))

test.to_csv("test.csv")
#%%