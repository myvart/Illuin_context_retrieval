import nltk
import preprocessor
import numpy as np
import math

def computeTF_text(tokenized_text):
    f_dist = nltk.FreqDist(tokenized_text)
    for word in f_dist:
        f_dist[word] /= float(len(tokenized_text))
    return f_dist

def computeIDF(documents, word):
    N = len(documents)
    
    count_texts_with_word = 0 
    
    for document in documents:
        if word in document:
            count_texts_with_word += 1
    
    return math.log(((N + 1) / float(count_texts_with_word + 1)) + 1)

def compute_TF_IDF(documents, words):
    idf = np.zeros(len(words))
    tf_idf = np.zeros((len(words), len(documents)))
    
    for i in range(len(words)):
        idf[i] = computeIDF(documents, words[i])
        
    for j in range(len(documents)):
        fd = computeTF_text(documents[j])
        for i in range(len(words)):
            tf_idf[i, j] = fd[words[i]]*idf[i]
    
    return tf_idf.sum(axis = 0)

def most_likely_context(question, contexts):
    tf_idf_mesures = compute_TF_IDF(contexts, question)
    return np.argmax(tf_idf_mesures)

def predict_new_instances(instances, contexts, preprocessor):
    preprocessed_questions_data_set = preprocessor.transform(instances)
    preprocessed_contexts_data_set = preprocessor.transform(contexts)
    return [most_likely_context(preprocessed_instance, preprocessed_contexts_data_set) for preprocessed_instance in preprocessed_questions_data_set]

def compute_accuracy(predictions, labels):
    return np.sum(predictions == labels)/len(labels)