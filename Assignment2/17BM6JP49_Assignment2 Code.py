# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:08:17 2018

@author: Shubhank
"""

import pandas as pd
import numpy as np
import os
import re
import time
import rouge
from collections import OrderedDict
import nltk

paths = {'truth': 'input/GroundTruth/', 'topic1' : 'input/Topic1/', 'topic2' : 'input/Topic2/',
         'topic3' : 'input/Topic3/', 'topic4' : 'input/Topic4/', 'topic5' : 'input/Topic5/'}

# ==================================================================================================
# Creating some Important Functions
# ==================================================================================================

def clean_text(text):
    cleaned = text.replace('</P>', ' ')
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('\t', ' ')
    cleaned = cleaned.replace('<P>', ' ')
    cleaned = cleaned.replace('&HT;', ' ')
    cleaned = cleaned.replace('<TEXT>', ' ')
    cleaned = cleaned.strip()
    return cleaned
    
# Function for extracting sentences
def extract_sentences(file):
    if '<P>' in file:
        paras = re.split('<P>', file)
        del paras[0]
        temp = re.split('</P>', paras[-1])
        paras[-1] = temp[0]
        if ('&HT;' in file and len(paras) < 4):
            paras = re.split('&HT;', file)
            del paras[0]
            del paras[-1]
    else:
        if '&HT;' in file:
            paras = re.split('&HT;', file)
            del paras[0]
            del paras[-1]
        elif '<TEXT>' in file:
            paras = re.findall(r'<TEXT>(.*?)</TEXT>',file, re.DOTALL)
            paras = nltk.tokenize.sent_tokenize(paras[0])
        else:
            print('New Format')
    for i in range(len(paras)):
        paras[i] = clean_text(paras[i])
    return paras

# Reading documents
def read_doc(path):
    file_list = os.listdir(path)
    files = []
    for file_name in file_list:
        with open(path + file_name, 'r') as file:
            #print('Reading file: ', file_name)
            file = file.read()
            file = extract_sentences(file)
            file = list(filter(None, file))
            files.append(file)
    return files

def text_processing(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokens = nltk.tokenize.word_tokenize(text)
    #stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    new_text = ""
    for token in tokens:
        token = token.lower()
        if token not in stopwords:
            #new_text += stemmer.stem(token)
            new_text += token
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
def tf_idf_matrix(ids, sentences, corpus):
    start = time.time()
    
    # Creating Term Document Matrix
    words = list(OrderedDict.fromkeys(corpus.split()))
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


# idf modified cosine
def idf_modified_cosine(data_frame):
    tic = time.time()
    x = np.array(data_frame)
    idf_mod_cos = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            num = np.sum(x[i] * x[j])
            den = np.sqrt(np.sum(x[i] * x[i])) * np.sqrt(np.sum(x[j] * x[j]))
            if den == 0:
                idf_mod_cos[i][j] = 0
            else:
                idf_mod_cos[i][j] = num/den
    print('Time for idf-modified-cosine: {0:.2f} seconds' .format(time.time() - tic))
    return idf_mod_cos


#==================================================================================================
# Degree Centrality Summary
#==================================================================================================

def degree_centrality(data_frame, similarity_matrix, threshold = 0.3):
    start = time.time()
    #cos_sim_matrix = idf_modified_cosine(data_frame)
    centrality_matrix = similarity_matrix.copy()

    centrality_matrix[centrality_matrix >= threshold] = 1
    centrality_matrix[centrality_matrix < threshold] = 0

    temp = pd.Series(data_frame.index.values).str.split('_',expand=True)
    temp.columns = ['DocID', 'SentenceID']
    temp = pd.DataFrame({'DocID': temp.DocID, 'SentenceID': temp.SentenceID, 
                         'Degree': np.sum(centrality_matrix, axis = 1)})
    print('Total time for calculating degree centrality: {0:.2f} seconds' .format(time.time() - start))
    return centrality_matrix, temp

def generate_deg_cent_summary(files, centrality, rank, words = 250):
    summary = ''
    while len(summary.split()) < 250:
        a = np.sum(centrality, axis = 1).argmax()
        summary = summary + ' ' + files[int(rank['DocID'][a])][int(rank['SentenceID'][a])]
        for i in range(len(centrality[a])):
            if centrality[a][i] == 1:
                centrality[i][:] = 0
                centrality[:][i] = 0    
        centrality[a][:] = 0
        centrality[:][a] = 0
        if len(summary.split()) > words:
            summary =  ' '.join(summary.split()[:words])
    return summary


def summarize_degree_centrality(files, words = 250, threshold = 0.1):
    start = time.time()
    # Appending all sentences in one array
    ids, sentences, token_sentences, corpus = append_sentences(files)
    
    # Creating tf-idf Matrix
    tf_idf_frame = tf_idf_matrix(ids, token_sentences, corpus)
    
    # idf-modified cosine matrix
    idf_mod_cosine_matrix = idf_modified_cosine(tf_idf_frame)    

    # Calculating degree centrality
    centrality, rank = degree_centrality(tf_idf_frame, idf_mod_cosine_matrix, threshold)
    
    # Generating summary
    summary = generate_deg_cent_summary(files, centrality, rank, words)
    print('Time for generating Degree Centrality Summary: {0:.2f} seconds' .format(time.time() - start))
    return summary



#==================================================================================================
# Text Rank Summary
#==================================================================================================

# Power Method
def power_method(matrix, error_tolerance):
    n = len(matrix)
    power_vector_init = np.full((n), 1.0/n)
    
    t = 0
    error_difference = error_tolerance + 1
    while error_difference > error_tolerance:
        t = t+1
        #if t%10 == 0:
        #    print('Power Loop: ' + str(t))
        #    print('Power error: ' + str(error_difference))
        power_vector = np.dot(np.transpose(matrix), power_vector_init)
        error_difference = np.linalg.norm(power_vector - power_vector_init)        
        power_vector_init = power_vector.copy()
    return power_vector

# Text Rank
def generate_textrank(similarity_matrix, threshold = 0.1):
    start = time.time()
    n = len(similarity_matrix)
    cosine_matrix = similarity_matrix.copy()
    degree = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if cosine_matrix[i][j] >= threshold:
                cosine_matrix[i][j] = 1
                degree[i] = degree[i] + 1
            else:
                cosine_matrix[i][j] = 0
                
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = cosine_matrix[i][j]/(degree[i]+1)
    
    L = power_method(cosine_matrix, 0.0000001)
    print('Time to generate TextRank Summary: {0:.2f} seconds' .format(time.time() - start))
    return L

def generate_textrank_summary(sentences, ranked_matrix, words = 250):
    textrank = ranked_matrix.copy()
    summary = ''
    while len(summary.split()) < words:
        max_index = textrank.argmax()
        summary = summary + ' ' + sentences[max_index]
        textrank[max_index] = 0
        if len(summary.split()) > words:
            summary =  ' '.join(summary.split()[:words])
    return summary


def summarize_TextRank(files, words = 250, threshold = 0.1):
    start = time.time()
    # Appending all sentences in one array
    ids, sentences, token_sentences, corpus = append_sentences(files)
    
    # Creating tf-idf Matrix
    tf_idf_frame = tf_idf_matrix(ids, token_sentences, corpus)
    
    # idf-modified cosine matrix
    idf_mod_cosine_matrix = idf_modified_cosine(tf_idf_frame)    

    # Calculating Text Rank
    textrank = generate_textrank(idf_mod_cosine_matrix, threshold)
    
    # Generating summary
    summary = generate_textrank_summary(sentences, textrank, words)
    print('Total time for generating summary: {0:.2f} seconds\n' .format(time.time() - start))
    return summary


#==================================================================================================
# Function to generate Both Summaries
#==================================================================================================

def generate_summaries(files, words = 250, threshold = 0.1):
    start = time.time()
    # Appending all sentences in one array
    ids, sentences, token_sentences, corpus = append_sentences(files)
    
    # Creating tf-idf Matrix
    tf_idf_frame = tf_idf_matrix(ids, token_sentences, corpus)
    
    # idf-modified cosine matrix
    idf_mod_cosine_matrix = idf_modified_cosine(tf_idf_frame)    

    # Calculating degree centrality
    centrality, rank = degree_centrality(tf_idf_frame, idf_mod_cosine_matrix, threshold)
    
    # Generating degree centrality summary
    degree_summary = generate_deg_cent_summary(files, centrality, rank, words)

    # Calculating Text Rank
    textrank = generate_textrank(idf_mod_cosine_matrix, threshold)
    
    # Generating Text Rank summary
    textrank_summary = generate_textrank_summary(sentences, textrank, words)

    print('Total time for generating summary: {0:.2f} seconds\n' .format(time.time() - start))
    return degree_summary, textrank_summary



#==================================================================================================
# Generating Summaries
#==================================================================================================


truths = dict()
documents = dict()
degree_summaries = dict()
textrank_summaries = dict()
degree_scores = dict()
textrank_scores = dict()
# Automated for all combinations
for i in range(5):
    # Reading truth
    print('============================================================================')
    print('Reading Topic: ' + str(i + 1))
    print('============================================================================')
    with open(paths['truth'] + 'Topic' + str(i+1) + '.1', 'r') as truth:
        truths['Topic' + str(i+1)] = truth.read()
    truths['Topic' + str(i+1)] = truths['Topic' + str(i+1)].replace('\n', '')
    
    # Reading the documents
    documents['Topic' + str(i+1)] = read_doc(paths['topic' + str(i+1)])
    
    '''
    # Checking length of each file
    for j in range(len(documents['Topic' + str(i+1)])):
        print('Length of file ' + str(j+1) + ' is: ' + str(len(documents['Topic' + str(i+1)][j])))
    '''
    
    # Generating Summary
    degree_summaries['Topic' + str(i+1) + '_trsh01'], textrank_summaries['Topic' + str(i+1) + '_trsh01'] = generate_summaries(documents['Topic' + str(i+1)], words=250, threshold=0.01)
    degree_summaries['Topic' + str(i+1) + '_trsh02'], textrank_summaries['Topic' + str(i+1) + '_trsh02'] = generate_summaries(documents['Topic' + str(i+1)], words=250, threshold=0.02)
    degree_summaries['Topic' + str(i+1) + '_trsh03'], textrank_summaries['Topic' + str(i+1) + '_trsh03'] = generate_summaries(documents['Topic' + str(i+1)], words=250, threshold=0.03)
    
    # Evaluation
    score = rouge.Rouge()
    degree_scores['Topic' + str(i+1) + '_trsh01'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              degree_summaries['Topic' + str(i+1) + '_trsh01'])
    degree_scores['Topic' + str(i+1) + '_trsh02'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              degree_summaries['Topic' + str(i+1) + '_trsh02'])
    degree_scores['Topic' + str(i+1) + '_trsh03'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              degree_summaries['Topic' + str(i+1) + '_trsh03'])

    textrank_scores['Topic' + str(i+1) + '_trsh01'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              textrank_summaries['Topic' + str(i+1) + '_trsh01'])
    textrank_scores['Topic' + str(i+1) + '_trsh02'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              textrank_summaries['Topic' + str(i+1) + '_trsh02'])
    textrank_scores['Topic' + str(i+1) + '_trsh03'] = score.get_scores(truths['Topic' + str(i+1)], 
                                                              textrank_summaries['Topic' + str(i+1) + '_trsh03'])
    

# Formatting Results
a = ['rouge-1', 'rouge-2', 'rouge-l']
b = ['f', 'p', 'r']
m = 0
degree_score_frame = pd.DataFrame(columns = ['Topic', 'Threshold', 'Metric1', 'Metric2', 'Value'])
textrank_score_frame = pd.DataFrame(columns = ['Topic', 'Threshold', 'Metric1', 'Metric2', 'Value'])
for i in range(5):
    for j in range(3):
        for k in a:
            for l in b:
                degree_score_frame.loc[m] = [i+1, '0.' + str(j+1), k, l, degree_scores['Topic' + str(i+1) + '_trsh0' + str(j+1)][0][k][l]]
                textrank_score_frame.loc[m] = [i+1, '0.' + str(j+1), k, l, textrank_scores['Topic' + str(i+1) + '_trsh0' + str(j+1)][0][k][l]]
                m = m+1


# Exporting
#scores['Topic1_trsh01'][0]['rouge-1']['f']
degree_score_frame.to_csv('Degree_Scores1.csv', index = False)
textrank_score_frame.to_csv('TextRank_Scores1.csv', index = False)
