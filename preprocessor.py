import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import spacy

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
s_words = stopwords.words('french')


class CorpusPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, remove_most_frequent = False, most_common_threshold = 100, lemmetize = False):
        self.vocab = None
        self.remove_most_frequent = remove_most_frequent
        self.most_common_threshold = most_common_threshold
        self.lemmetize = lemmetize
    
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
    
    def compute_vocab(tokenize_dataset, remove_most_frequent, threshold):
        concatenated_data_set = [token for list_of_tokens in tokenize_dataset for token in list_of_tokens]
        if remove_most_frequent:
            f_dist = nltk.FreqDist(concatenated_data_set)
            most_freq = f_dist.most_common(threshold)
            most_common_w = [word for word,_ in most_freq]
            vocab = set([token for token in concatenated_data_set if token not in most_common_w])
        else:
            vocab = set(concatenated_data_set)
        return vocab
    
    def filter_instance_with_vocab(tokenized_instance, vocab):
        return [token for token in tokenized_instance if token in vocab]
    
    def filter_data_set_with_vocab(tokenized_data_set, vocab):
        return [CorpusPreprocessor.filter_instance_with_vocab(tokenized_instance, vocab) for tokenized_instance in tokenized_data_set]
    
    def lemmetize_instance(tokenized_instance, lemmetizer):
        doc = lemmetizer(' '.join(tokenized_instance))
        return [token.lemma_ for token in doc]

    def lemmetize_data_set(tokenized_data_set, lemmetizer):
        return [CorpusPreprocessor.lemmetize_instance(tokenized_instance, lemmetizer) for tokenized_instance in tokenized_data_set]

    def fit(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        self.vocab = CorpusPreprocessor.compute_vocab(X_preprocessed, self.remove_most_frequent, self.most_common_threshold)
        
    def transform(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        X_preprocessed = CorpusPreprocessor.filter_data_set_with_vocab(X_preprocessed, self.vocab)
        return X_preprocessed
        
    def fit_transform(self, X, y = None):
        X_preprocessed = CorpusPreprocessor.remove_stop_words(CorpusPreprocessor.tokenize(CorpusPreprocessor.lower_data_set(CorpusPreprocessor.remove_symbols(X))))
        if self.lemmetize:
            lemmetizer = spacy.load('fr_core_news_sm')
            X_preprocessed = CorpusPreprocessor.lemmetize_data_set(X_preprocessed, lemmetizer)
        self.vocab = CorpusPreprocessor.compute_vocab(X_preprocessed, self.remove_most_frequent, self.most_common_threshold)
        X_preprocessed = CorpusPreprocessor.filter_data_set_with_vocab(X_preprocessed, self.vocab)
        return X_preprocessed