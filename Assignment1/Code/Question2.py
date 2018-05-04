# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:27:22 2018

@author: Shubhank
"""

# Importing Libraries
import os
import time
import json
import pandas as pd
import important_functions as funs
from important_functions import unique_list

# Reading Files and Queries
file_list = os.listdir('input/alldocs')
path_input = 'input/'
path_output = 'output/'
path_docs = path_input + 'alldocs/'
query_input = open(path_input + 'query.txt','r')

# Processing files and Storing in list
tic = time.time()
i = 0
temp = [[]]
for file_name in file_list:
    with open(path_docs + file_name, 'r', encoding="utf8") as file:
        file = file.read()
        file = funs.text_processing(file)
        file = unique_list(file)
        temp.append([file_name, file])
        i = i + 1
time_taken = time.time() - tic
print(time_taken)

# Creating Dictionary of words for filelist
tic = time.time()
norm_dict = dict()
for i in range(1,len(temp)):
    norm_dict[temp[i][0]] = funs.tokenizer(temp[i][1])

# Creating inverted dictionary for each word
inv_dict = dict()
for i in file_list:
    for word in norm_dict[i]:
        if word not in inv_dict:
            inv_dict[word] = [i]
        else:
            inv_dict[word].append(i)
toc = time.time() - tic
print(toc)

'''
# Saving inverted index on disk
with open('inverted_dict.txt', 'w') as file:
     file.write(json.dumps(inv_dict))

# Loading inverted index stored on disk
with open('inverted_dict.txt', 'r') as inv_dict:
     inv_dict = inv_dict.read()
'''

# Searching the query
tic= time.time()
file_list = os.listdir('input/alldocs')
path_input = 'input/'
path_output = 'output/'
query_input = open(path_input + 'query.txt','r')

words = []
query_ids = []
for lines in query_input:
    query_ids.append(lines[0:3])
    query_line = lines[5:]
    query_line = funs.text_processing(query_line)
    query_line = funs.tokenizer(query_line)
    words.append(query_line)
toc = time.time() - tic
print('Time taken to read queries: ', toc)

result = []
query_time = []
for word in words:
    tic = time.time()
    for i in range(len(word)):
        if i == 0:
            out_temp = inv_dict[word[i]]
        else:
            temp = inv_dict[word[i]]
            out_temp = list(set(out_temp).intersection(set(temp)))
    result.append(out_temp)
    query_time.append(time.time() - tic)

# Storing Results
output_df = pd.DataFrame({'QueryID': query_ids, 'DocID':result})
temp = output_df.apply(lambda x: pd.Series(x['DocID']),axis=1).stack().reset_index(level=1, drop=True) # Expanding results
temp.name = 'DocID'
output_df = output_df.drop('DocID', axis=1).join(temp)
output_df.sort_values(by = ['QueryID', 'DocID'], inplace = True)
output_df.to_csv(path_output + 'Inverted1.csv', index = False)
output_df.to_csv(path_output + 'Inverted1.txt', header=None, index=None, sep=' ', mode='a')

# Formating original output file
true_output = pd.read_table(path_input + 'output.txt', sep = " ", header = None)
true_output.drop(2, axis = 1, inplace = True)
true_output.columns = ['QueryID', 'DocID']
true_output.sort_values(by = ['QueryID', 'DocID'], inplace = True)
true_output.reset_index(drop = True, inplace = True)

inverted_time = pd.DataFrame({'QueryID': query_ids, 'Time': query_time})
inverted_time.to_csv(path_output + 'inverted_time.csv', index = False)

# Calculation and Storing Precision and Recall
funs.get_precision_recall(actual=true_output, predicted=output_df,
                          filename='inverted_index_precision_recall.txt')


