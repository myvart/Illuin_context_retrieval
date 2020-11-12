import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import spacy

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
s_words = stopwords.words('french')


# This python module aims to propose a custom scikit-learn transformer class which automate the preprocessing steps for
# the task of context retrieval.
# The constructor's hyperparameters are:
#   - remove_most_frequent : boolean (default value = False) if True, the most common tokens in the corpus to fit are removed
#                            from the vocabulary (after elimination of symbols and stop words)
#   - most_common_threshold : integer (default value = 100) determines the number of common tokens to remove if remove_most_common = True
#   - lemmetize : boolean (default value = False) if True, the preprocessor use a lemmetize the set of tokens before computing the
#                 frequency distribution of the tokens and the vocabulary 


class CorpusPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, remove_most_common = False, most_common_threshold = 100, lemmetize = False, ngrams = 1):
        self.most_common_w = []
        self.remove_most_common = remove_most_common
        self.most_common_threshold = most_common_threshold
        self.lemmetize = lemmetize
        self.ngrams = ngrams
    
    def remove_symbols_from_string(string):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in symbols:
            string = string.replace(i, ' ')
        string = string.replace("'", ' ')
        string = string.replace(",", '')
        return string
    
    def remove_symbols(data_set):
        return [CorpusPreprocessor.remove_symbols_from_string(string) for string in data_set]
    
    def lower_data_set(data_set):
        return [string.lower() for string in data_set]
    
    def tokenize(data_set):
        return [nltk.word_tokenize(string, language='french') for string in data_set]
    
    def remove_stop_words_from_instance(tokenized_instance):
        return  [token for token in tokenized_instance if token not in s_words]

    def remove_stop_words(tokenized_dataset):
        return [CorpusPreprocessor.remove_stop_words_from_instance(instance) for instance in tokenized_dataset]
    
    def compute_most_common_words(tokenize_dataset, threshold):
        concatenated_data_set = [token for list_of_tokens in tokenize_dataset for token in list_of_tokens]
        f_dist = nltk.FreqDist(concatenated_data_set)
        most_freq = f_dist.most_common(threshold)
        most_common_w = [word for word,_ in most_freq]
        return most_common_w
    
    def filter_instance_with_most_common_w(tokenized_instance, most_common_w):
        return [token for token in tokenized_instance if token not in most_common_w]
    
    def filter_data_set_with_most_common_w(tokenized_data_set, most_common_w):
        return [CorpusPreprocessor.filter_instance_with_most_common_w(tokenized_instance, most_common_w) for tokenized_instance in tokenized_data_set]
    
    def lemmetize_instance(tokenized_instance, lemmetizer):
        doc = lemmetizer(' '.join(tokenized_instance))
        return [token.lemma_ for token in doc]

    def lemmetize_data_set(tokenized_data_set, lemmetizer):
        return [CorpusPreprocessor.lemmetize_instance(tokenized_instance, lemmetizer) for tokenized_instance in tokenized_data_set]

    def compute_ngrams_instance(tokenized_instance, n):
        return [ngram for ngram in nltk.ngrams(tokenized_instance, n)]
    
    def compute_ngrams_data_set(tokenized_data_set, n):
        return [CorpusPreprocessor.compute_ngrams_instance(tokenized_instance, n) for tokenized_instance in tokenized_data_set]
    
    def fit(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        X_preprocessed = CorpusPreprocessor.compute_ngrams_data_set(X_preprocessed, self.ngrams)
        self.most_common_w = CorpusPreprocessor.compute_most_common_words(X_preprocessed, self.most_common_threshold)
        
    def transform(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        X_preprocessed = CorpusPreprocessor.compute_ngrams_data_set(X_preprocessed, self.ngrams)
        X_preprocessed = CorpusPreprocessor.filter_data_set_with_most_common_w(X_preprocessed, self.most_common_w)
        return X_preprocessed
        
    def fit_transform(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        X_preprocessed = CorpusPreprocessor.compute_ngrams_data_set(X_preprocessed, self.ngrams)
        self.most_common_w = CorpusPreprocessor.compute_most_common_words(X_preprocessed, self.most_common_threshold)
        X_preprocessed = CorpusPreprocessor.filter_data_set_with_most_common_w(X_preprocessed, self.most_common_w)
        return X_preprocessed