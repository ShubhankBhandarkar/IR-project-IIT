# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:41 2018

@author: Shubhank
"""

import os
from elasticsearch import Elasticsearch
import time
import pandas as pd
import important_functions as funs

path_input = 'input/'
path_output = 'output/'
file_list = os.listdir('input/alldocs')

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Creating Index using elastic search
file_paths=[]
directory_path = path_input + 'alldocs'
for root,dirs,files in os.walk(directory_path):
        for f in files:
            file_paths.append(os.path.join(root,f))

count = 1
for file in file_paths:
    f=open(file,'rb')
    s=str(f.read())
    res=es.create(index='doc',id = count, doc_type='text',body={'content':s,'name':file})
    print(count)
    print(res)
    count=count+1

file_details = pd.DataFrame({'Name': file_list, 'Path': file_paths})

# Searching the queries in index
query_index = []
query_time = []
file_dict = dict()
f = open(path_input + 'query.txt')
for line in f:
    tic = time.time()
    line = line.strip()
    words = line.split(" ")
    index = words[0]
    try:
        query = words[2]
        line = words[3:]
    except:
        continue
    for x in words:
        query = query + " " + x
    res = es.search(index = 'doc', doc_type = 'text', body = {"from":0, "size":100, 
                                                              "query":{"match":{"content": query}}})
    query_index.append(index)
    rest = []
    for doc in res['hits']['hits']:
        if (doc['_source']['name']) not in rest:
            rest.append(doc['_source']['name'])
        if len(rest)>60:
            break
    file_dict[index] = rest
    query_time.append(time.time() - tic)
f.close()

# Formatting for output
for i in query_index:
    temp = pd.DataFrame(file_dict[i], columns = ['Path'])
    temp = temp.merge(file_details, on = 'Path', how = 'left')['Name']
    file_dict[i] = list(temp)

# Creating output
id_list = []
doc_list = []
for i in query_index:
    for j in file_dict[i]:
        id_list.append(i)
        doc_list.append(j)
elastic_output = pd.DataFrame({'QueryID': id_list, 'DocID': doc_list})
elastic_output.to_csv(path_output + 'elastic_output.csv', index = False)

# Formating original output file
true_output = pd.read_table(path_input + 'output.txt', sep = " ", header = None)
true_output.drop(2, axis = 1, inplace = True)
true_output.columns = ['QueryID', 'DocID']
true_output.sort_values(by = ['QueryID', 'DocID'], inplace = True)
true_output.reset_index(drop = True, inplace = True)
true_output['QueryID'] = true_output['QueryID'].astype(str)

elastic_time = pd.DataFrame({'QueryID': query_index, 'Time': query_time})
elastic_time.to_csv(path_output + 'elastic_time.txt', index=None, sep=' ', mode='a')
elastic_time.to_csv(path_output + 'elastic_time.csv', index=None)

# Calculation and Storing Precision and Recall
funs.get_precision_recall(actual=true_output, predicted=elastic_output,
                          filename='elastic_precision_recall.txt')





