# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:27:11 2018

@author: Shubhank
"""

import os
import re
import time
import pandas as pd
import important_functions as funs

# Reading Queries
file_list = os.listdir('input/alldocs')
path_input = 'input/'
path_output = 'output/'
query_input = open(path_input + 'query.txt','r')

tick = time.time()
words = []
query_ids = []
for lines in query_input:
    query_ids.append(lines[0:3])
    query_line = lines[5:]
    query_line = funs.text_processing(query_line)
    query_line = query_line.strip()
    words.append(query_line.split(" "))

files = []
for file_name in file_list:
    with open('input/alldocs/' + file_name, 'r', encoding="utf8") as file:
        file = file.read()
        file = funs.text_processing(file)
        files.append(file)

temp_filename = []
temp_query = []
temp_check = []
for query_id in query_ids:
    check = True
    i = 0
    for j in range(len(words[i])):
        x = bool(re.search(words[i][j], file))
        check = check * x
    i = i + 1
    temp_filename.append(file_name)
    temp_query.append(query_id)
    temp_check.append(check)


for file_name in file_list:
    with open('input/alldocs/' + file_name, 'r', encoding="utf8") as file:
        file = file.read()
        file = funs.text_processing(file)
        i = 0
        for query_id in query_ids:
            check = True
            for j in range(len(words[i])):
                x = bool(re.search(words[i][j], file))
                check = check * x
            i = i + 1
            temp_filename.append(file_name)
            temp_query.append(query_id)
            temp_check.append(check)
        
tock = time.time()
time_taken = tock - tick
print(time_taken)


final_output = pd.DataFrame({'DocID' : temp_filename, 'QueryID' : temp_query, 'Check':temp_check})
final_output = final_output.loc[final_output.Check == 1]
final_output = final_output.loc[final_output.QueryID != '\n']
final_output.sort_values(by = 'QueryID', inplace = True)
final_output.drop('Check', axis = 1, inplace = True)
final_output.reset_index(drop = True, inplace = True)
final_output.to_csv(path_output + 'grep_search.csv', index = False)


# Formating original output file
true_output = pd.read_table(path_input + 'output.txt', sep = " ", header = None)
true_output.drop(2, axis = 1, inplace = True)
true_output.columns = ['QueryID', 'DocID']
true_output.sort_values(by = ['QueryID', 'DocID'], inplace = True)
true_output.reset_index(drop = True, inplace = True)
true_output['QueryID'] = true_output['QueryID'].astype(str)

# Calculation and Storing Precision and Recall
funs.get_precision_recall(actual=true_output, predicted=final_output,
                          filename='grep_precision_recall.txt')




