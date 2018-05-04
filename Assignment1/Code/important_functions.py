# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:43:53 2018

@author: Shubhank
"""

import re
import nltk

def get_precision_recall(actual, predicted, filename):
    """
    Args:
        output_file: file containing result for queries
    """
    path_input = 'input/'
    query = open(path_input + "query.txt","r")
    query_ID = []
    prerec = open(filename,"a")
    prerec.write("queryID" + " " + "precision" + " " + "recall" + "\n")
    for line in query:
        query_ID.append(line.split()[0])
        
    for q_ID in query_ID:
        estimated = list(predicted[predicted['QueryID'] == q_ID]["DocID"])
        true = list(actual[actual["QueryID"] == q_ID]["DocID"])
        precision = len(list(set(estimated).intersection(set(true))))/float(len(estimated))
        recall = len(list(set(estimated).intersection(set(true))))/float(len(true))
        prerec.write(str(q_ID) + " " + str(round(precision,3)) + " " + str(round(recall,3)) + "\n")
    prerec.close()


def tokenizer(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
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


def unique_list(text):
    l = text.split()
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ' '.join(ulist)
