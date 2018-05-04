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

def tokenizer(text):
    text = text.replace('\\n',' ')
    text = text.replace('\'','')
    text = re.sub('[^A-Za-z]+', ' ', text)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens        

def text_processing(text):
    tokens = tokenizer(text)
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    new_text = ""
    for token in tokens:
        token = token.lower()
        if token not in stopwords:
            new_text += stemmer.stem(token)
            new_text += " "
    return new_text

# Appending all the sentences in one array
def append_sentences(files):
    sentence_id = []
    sentences_arr = []
    sentence_corpus = ''
    tokenized_sentences_arr = []
    for i in range(len(files)):
        for j in range(len(files[i])):
            sentence_id.append(str(i) + '_' + str(j))
            sentences_arr.append(files[i][j])
            sentence_corpus = sentence_corpus + text_processing(files[i][j]) + ' '
            tokenized_sentences_arr.append(text_processing(files[i][j]))
    return sentence_id, sentences_arr, tokenized_sentences_arr, sentence_corpus

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
# File and Query Processing
#===================================================================================================

# Reading files and queries
file_list = os.listdir('input/alldocs')
path_input = 'input/'
path_output = 'output/'
path_docs = path_input + 'alldocs/'
query_input = open(path_input + 'query.txt','r')

# Processing Files
tic = time.time()
i = 0
file_data = [[]]
for file_name in file_list:
    with open(path_docs + file_name, 'r', encoding="utf8") as file:
        file = file.read()
        file = text_processing(file)
        file_data.append([file_name, file])
        i = i + 1
time_taken = time.time() - tic
cleaned_files_df = pd.DataFrame(file_data, columns = ['FileName', 'Text'])
cleaned_files_df = cleaned_files_df.drop(cleaned_files_df.index[0])
cleaned_files_df.reset_index(drop = True, inplace = True)
#cleaned_files_df.to_csv('Cleaned documents_new.csv', index = False)
#cleaned_files_df = pd.read_csv('Cleaned documents')
#cleaned_files_df = cleaned_files_df.fillna('')
print(time_taken)

# Processing Queries
queries = []
query_ids = []
original_queries = []
for lines in query_input:
    query_ids.append(lines[0:3])
    query_line = lines[5:]
    original_queries.append(query_line)
    query_line = text_processing(query_line)
    queries.append(query_line)

cleaned_query_df = pd.DataFrame({'FileName': query_ids, 'Text': queries})
#cleaned_query_df.to_csv('Cleaned Query.csv', index = False)
#cleaned_query_df = pd.read_csv('Cleaned Query')
#query_ids = cleaned_query_df.FileName
#queries = cleaned_query_df.Text

# Creating tfidf vector
#tfidf = TfidfVectorizer()
#tfidf.fit(cleaned_files_df.Text)
tfidf = tf_idf_matrix(cleaned_files_df.FileName, cleaned_files_df.Text)

#===================================================================================================
# Running Elastic Search on original queries
#===================================================================================================

# Initializing elastic search
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Creating Index using elastic search
try:
    es.indices.delete(index = 'assignment3')
except:
    pass
for i in range(len(cleaned_files_df)):
    res = es.create(index = 'assignment3', id = i, doc_type = 'text',
                    body={'content': cleaned_files_df.iloc[i,1], 
                          'name': cleaned_files_df.iloc[i,0]})

# Searching using elastic search
i = 0
query_time = []
file_dict = dict()
for query in queries:
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
elastic_output_basic = pd.DataFrame({'QueryID': id_list, 'DocID': doc_list})
elastic_output_basic.to_csv(path_output + 'elastic_output.csv', index = False)
#elastic_output_basic = pd.read_csv(path_output + 'elastic_output.csv')

# Formating original output file
true_output = pd.read_table(path_input + 'output.txt', sep = " ", header = None)
true_output.drop(2, axis = 1, inplace = True)
true_output.columns = ['QueryID', 'DocID']
true_output.sort_values(by = ['QueryID', 'DocID'], inplace = True)
true_output.reset_index(drop = True, inplace = True)
true_output['QueryID'] = true_output['QueryID'].astype(str)

elastic_time = pd.DataFrame({'QueryID': query_ids, 'Time': query_time})
elastic_time.to_csv(path_output + 'elastic_time.txt', index=None, sep=' ', mode='a')
elastic_time.to_csv(path_output + 'elastic_time.csv', index=None)

# Calculation and Storing Precision, Recall and F1 Score
elastic_score_basic = calculate_scores(query_ids, actual=true_output, predicted=elastic_output_basic)
mean_score = pd.DataFrame({'QueryID': 'Average', 'Precision': np.mean(elastic_score_basic.Precision),
                           'Recall': np.mean(elastic_score_basic.Recall),
                           'F1score': np.mean(elastic_score_basic.F1score)}, index = [82])

elastic_score_basic = elastic_score_basic.append(mean_score)[elastic_score_basic.columns.tolist()]
elastic_score_basic[['Precision', 'Recall', 'F1score']] = np.round(elastic_score_basic[['Precision', 'Recall', 'F1score']], decimals=2)
elastic_score_basic.to_csv(path_output + 'Task1_deliverables/Performance_before_relevance_feedback.txt', 
                           index = None, sep = '\t')
