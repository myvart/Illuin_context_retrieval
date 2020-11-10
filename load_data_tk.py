import os
import numpy as np
import json

def load_data_train_set(json_file, test_size):
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