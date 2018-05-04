# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:07:21 2018

@author: Shubhank
"""

import os
from elasticsearch import Elasticsearch
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import numpy as np
from collections import OrderedDict

#===================================================================================================
# Functions
#===================================================================================================

def calculate_scores(query_ids, actual, predicted):
    result = []
    for q_ID in query_ids:
        estimated = list(predicted[predicted['QueryID'] == q_ID]["DocID"])
        true = list(actual[actual["QueryID"] == q_ID]["DocID"])
        precision = len(list(set(estimated).intersection(set(true))))/float(len(estimated))
        recall = len(list(set(estimated).intersection(set(true))))/float(len(true))
        f1_score = 2 * precision * recall/(precision + recall)
        result.append([q_ID, precision, recall, f1_score])
    result = pd.DataFrame(result, columns = ['QueryID', 'Precision', 'Recall', 'F1score'])
    return result

# Creating tf-idf Matrix
def tf_idf_matrix(ids, sentences):
    start = time.time()
    corpus = ''
    for i in range(len(sentences)):
        corpus = corpus + ' ' + sentences[i]
    # Creating Term Document Matrix
    words = list(OrderedDict.fromkeys(corpus.split()))
    print('Corpus Created')
    #print('Length of word corpus is: ',len(words))

    tdm_frame = pd.DataFrame(0, index = ids, columns = words)
    # Creating Dictionary of words for filelist
    for i in range(len(sentences)):
        for j in sentences[i].split():
            tdm_frame[j][i] = tdm_frame[j][i] + 1

    # Calculating Term Frequency
    word_freq = pd.DataFrame(0, index = words, columns = ['Frequency'])
    for i in words:
        word_freq['Frequency'][i] = tdm_frame[tdm_frame[i] > 0][i].count()
    word_freq['idf'] = np.log10(len(words)/word_freq['Frequency'])

    # Calulating tf-idf matrix
    np.array(tdm_frame) * np.array(word_freq['idf'])
    tf_idf_frame = pd.DataFrame(np.array(tdm_frame) * np.array(word_freq['idf']), 
                                index = ids, columns = words)
    print('Total time for creating tf-idf matrix: {0:.2f} seconds' .format(time.time() - start))
    return tf_idf_frame

#===================================================================================================
# Reading Files
#===================================================================================================

# Reading files and queries
file_list = os.listdir('input/alldocs')
path_input = 'input/'
path_output = 'output/'
path_docs = path_input + 'alldocs/'
query_input = open(path_input + 'query.txt','r')


cleaned_files_df = pd.read_csv('Cleaned documents')
cleaned_files_df = cleaned_files_df.fillna('')

cleaned_query_df = pd.read_csv('Cleaned Query')
query_ids = cleaned_query_df.FileName
queries = cleaned_query_df.Text

# Creating tfidf vector
#tfidf = TfidfVectorizer()
#tfidf.fit(cleaned_files_df.Text)

file_query_df = cleaned_query_df.append(cleaned_files_df)
file_query_df.reset_index(drop = True, inplace = True)
tfidf = tf_idf_matrix(file_query_df.FileName[0:100], file_query_df.Text[0:100])

#===================================================================================================
# Relevance Feedback
#===================================================================================================

elastic_output_basic = pd.read_csv(path_output + 'elastic_output.csv')
# Extracting top 10 documents for each query as relevant
relevant_files = elastic_output_basic.groupby('QueryID').head(10)
relevant_files = relevant_files.merge(cleaned_files_df, how = 'left', 
                                      left_on = 'DocID', right_on = 'FileName')
# Tfidf vector of original query
#original_query_vector = tfidf.transform(queries).toarray()
original_query_vector = np.array(tfidf.iloc[0:82])

# Calculating the updated queries using algo
#feature_array = np.array(tfidf.get_feature_names())
feature_array = np.array(tfidf.columns.values)
updated_queries = []
for i in range(len(query_ids)):
    #temp_tfidf = tfidf.transform(relevant_files[10*i:10*i+10]).toarray()
    temp_tfidf = np.array(tfidf.loc(relevant_files[10*i:10*i+10]))
    temp_query_vector = original_query_vector[i] + temp_tfidf.sum(axis = 0) * (0.65/10.0)
    tfidf_sorting = np.argsort(temp_query_vector).flatten()[::-1]
    top10_queries = feature_array[tfidf_sorting][:10]
    top10_queries = ' '.join(top10_queries)
    updated_queries.append(top10_queries)

#===================================================================================================
# Running Elastic Search using updated queries
#===================================================================================================

# Initializing elastic search
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

i = 0
query_time = []
file_dict = dict()
for query in updated_queries:
    tic = time.time()
    res = es.search(index = 'assignment3', doc_type = 'text', body = {"from":0, "size":50, 
                                                              "query":{"match":{"content": query}}})
    rest = []
    for doc in res['hits']['hits']:
        if (doc['_source']['name']) not in rest:
            rest.append(doc['_source']['name'])
        if len(rest)>60:
            break
    file_dict[query_ids[i]] = rest
    query_time.append(time.time() - tic)
    i+=1

# Creating output
id_list = []
doc_list = []
for i in query_ids:
    for j in file_dict[i]:
        id_list.append(i)
        doc_list.append(j)
elastic_output_relevance = pd.DataFrame({'QueryID': id_list, 'DocID': doc_list})
elastic_output_relevance.to_csv(path_output + 'elastic_output_relevace.csv', index = False)

elastic_relevance_time = pd.DataFrame({'QueryID': query_ids, 'Time': query_time})
elastic_relevance_time.to_csv(path_output + 'elastic_relevance_time.txt', index=None, sep='\t')
elastic_relevance_time.to_csv(path_output + 'elastic_relevance_time.csv', index=None)

# Formating original output file
true_output = pd.read_table(path_input + 'output.txt', sep = " ", header = None)
true_output.drop(2, axis = 1, inplace = True)
true_output.columns = ['QueryID', 'DocID']
true_output.sort_values(by = ['QueryID', 'DocID'], inplace = True)
true_output.reset_index(drop = True, inplace = True)

# Calculation and Storing Precision and Recall
elastic_score_relevance = calculate_scores(query_ids, actual=true_output, 
                                           predicted=elastic_output_relevance)
mean_score = pd.DataFrame({'QueryID': 'Average', 'Precision': np.mean(elastic_score_relevance.Precision),
                           'Recall': np.mean(elastic_score_relevance.Recall),
                           'F1score': np.mean(elastic_score_relevance.F1score)}, index = [82])

elastic_score_relevance = elastic_score_relevance.append(mean_score)[elastic_score_relevance.columns.tolist()]
elastic_score_relevance[['Precision', 'Recall', 'F1score']] = np.round(elastic_score_relevance[['Precision', 'Recall', 'F1score']], decimals=2)
elastic_score_relevance.to_csv(path_output + 'Task1_deliverables/Performance_after_relevance_feedback.txt', 
                               index = None, sep = '\t')
