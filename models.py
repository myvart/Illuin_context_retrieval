import nltk
import preprocessor
import numpy as np
import math
import time

# This python module's aim is to propose a set of models for the context retrieval task.


# Simple TF-IDF model :

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

def tfidf_most_likely_context(question, contexts):
    tf_idf_mesures = compute_TF_IDF(contexts, question)
    return np.argmax(tf_idf_mesures)

def tfidf_predict_new_instances(instances, contexts, preprocessor):
    t1 = time.time()
    preprocessed_questions_data_set = preprocessor.transform(instances)
    preprocessed_contexts_data_set = preprocessor.transform(contexts)
    return [tfidf_most_likely_context(preprocessed_instance, preprocessed_contexts_data_set) for preprocessed_instance in preprocessed_questions_data_set], (time.time() - t1)/len(instances)


# Mixt TF-IDF model (mixing unigrams TFIDF and bigrams TFIDF mesures) using a weighted sum :

def mixt_model_most_likely_context(unigram_instance, unigram_contexts, bigram_instance, bigram_contexts, alpha):
    unigram_tfidf_mesures = compute_TF_IDF(unigram_contexts, unigram_instance)
    bigram_tfidf_mesures = compute_TF_IDF(bigram_contexts, bigram_instance)
    
    aggregated_mesures = unigram_tfidf_mesures + alpha * bigram_tfidf_mesures
    
    return np.argmax(aggregated_mesures)

def mixt_model_predict_new_instances(instances, contexts, unigram_preprocessor, bigram_preprocessor, alpha):
    t1 = time.time()
    unigram_preprocessed_questions_data_set = unigram_preprocessor.transform(instances)
    unigram_preprocessed_contexts_data_set = unigram_preprocessor.transform(contexts)
    
    bigram_preprocessed_questions_data_set = bigram_preprocessor.transform(instances)
    bigram_preprocessed_contexts_data_set = bigram_preprocessor.transform(contexts)
    
    return [mixt_model_most_likely_context(unigram_preprocessed_questions_data_set[i],
                                           unigram_preprocessed_contexts_data_set, bigram_preprocessed_questions_data_set[i],
                                           bigram_preprocessed_contexts_data_set, alpha) for i in range(len(instances))] , (time.time() - t1)/len(instances)


# Mixt TF-IDF model using the harmonic mean of unigrams TFIDF and bigrams TFIDF mesures:

def mixt_model_most_likely_context_2(unigram_instance, unigram_contexts, bigram_instance, bigram_contexts):
    unigram_tfidf_mesures = compute_TF_IDF(unigram_contexts, unigram_instance)
    bigram_tfidf_mesures = compute_TF_IDF(bigram_contexts, bigram_instance)
    
    aggregated_mesures = 2 * unigram_tfidf_mesures * bigram_tfidf_mesures / (unigram_tfidf_mesures + bigram_tfidf_mesures)
    
    return np.argmax(aggregated_mesures)

def mixt_model_predict_new_instances_2(instances, contexts, unigram_preprocessor, bigram_preprocessor):
    t1 = time.time()

    unigram_preprocessed_questions_data_set = unigram_preprocessor.transform(instances)
    unigram_preprocessed_contexts_data_set = unigram_preprocessor.transform(contexts)
    
    bigram_preprocessed_questions_data_set = bigram_preprocessor.transform(instances)
    bigram_preprocessed_contexts_data_set = bigram_preprocessor.transform(contexts)
    
    return [mixt_model_most_likely_context(unigram_preprocessed_questions_data_set[i],
                                           unigram_preprocessed_contexts_data_set, bigram_preprocessed_questions_data_set[i],
                                           bigram_preprocessed_contexts_data_set) for i in range(len(instances))] , (time.time() - t1)/len(instances)

# Compute the accuracy of a set of predictions:

def compute_accuracy(predictions, labels):
    return np.sum(predictions == labels)/len(labels)