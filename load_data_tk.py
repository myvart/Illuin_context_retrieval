import os
import numpy as np
import json

# This python module's aim is to propose functions to retrieve data from the json files of fquad data set, and 
# to return those data in a sutable form to address the problem of context retrieval.

def load_data_train_set(json_file, test_size = 0.2):
    # This function retriev the data from the train_set json file of fquad data set and realise a train_val split
    #
    # It takes 1 parameter:
    #   - json_file : (type file) which gives the json file in which to retrieve the data
    #
    # It takes 1 hyperparameter :
    #   - test_size : (type float, default value 0.2) which gives the fraction of instances to consider to build the validation set
    # It returns 5 objects :
    #   - a corpus (list of unique strings) composed of questions of the training set instances
    #   - a corpus (list of unique strings) composed of contexts of the training set instances
    #   - a set (list of strings) composed of questions of the validation set instances
    #   - a corpus (list of unique strings) composed of the contexts of the validation set instances
    #   - a numpy array containing the labels which uniquely identify the context corresponding to a given question of the validation set

    data = json.load(json_file)['data']
    
    questions_corpus_train = []
    contexts_corpus_train = []
    questions_set_val = []
    contexts_corpus_val = []
    
    list_of_indexes = []
    
    for i in range(len(data)):
        for j in range(len(data[i]['paragraphs'])):
            list_of_indexes.append((i,j))
            
    np.random.shuffle(list_of_indexes)
    val_set_indexes = list_of_indexes[ : int(len(list_of_indexes) * test_size)]
    train_set_indexes = list_of_indexes[int(len(list_of_indexes) * test_size) + 1 : ]
    
    val_labels = []
    label = 0
    
    for i, j in val_set_indexes:
        contexts_corpus_val.append(data[i]['paragraphs'][j]['context'])
        for k in range(len(data[i]['paragraphs'][j]['qas'])):
            questions_set_val.append(data[i]['paragraphs'][j]['qas'][k]['question'])
            val_labels.append(label)
        label += 1
        
    for i, j in train_set_indexes:
        contexts_corpus_train.append(data[i]['paragraphs'][j]['context'])
        for k in range(len(data[i]['paragraphs'][j]['qas'])):
            questions_corpus_train.append(data[i]['paragraphs'][j]['qas'][k]['question'])
            
    
    
    return list(set(questions_corpus_train)), list(set(contexts_corpus_train)), questions_set_val, contexts_corpus_val, np.array(val_labels)


def load_data_test_set(json_file):
    # This function retriev the data from the val_set json file of fquad data set in order to build the test set.
    #
    # It takes 1 parameter:
    #   - json_file : (type file) which gives the json file in which to retrieve the data
    #
    # It returns 5 objects :
    #   - a set (list of strings) composed of questions of the test set instances
    #   - a corpus (list of unique strings) composed of the contexts of the test set instances
    #   - a numpy array containing the labels which uniquely identify the context corresponding to a given question of the test set



    data = json.load(json_file)['data']
        
    questions_set = []
    contexts_corpus = []
        
    test_labels = []
    label = 0
        
    for i in range(len(data)):
        for j in range(len(data[i]['paragraphs'])):
            contexts_corpus.append(data[i]['paragraphs'][j]['context'])
            for k in range(len(data[i]['paragraphs'][j]['qas'])):
                questions_set.append(data[i]['paragraphs'][j]['qas'][k]['question'])
                test_labels.append(label)
            label += 1 
                
    return questions_set, contexts_corpus, np.array(test_labels)