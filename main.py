import os
import preprocessor
import load_data_tk
import models

# Retrieve data from the fquad training set to build the training set and the validation set:

train_set_file_path = os.path.join(os.getcwd(), "train_set", "train.json")

with open(train_set_file_path, encoding='utf-8') as json_file:
    questions_corpus_train, contexts_corpus_train, questions_set_val, contexts_corpus_val, val_labels = load_data_tk.load_data_train_set(json_file, test_size = 0.2)

print('\n\n-------------------------------------------------------------------------')
print(f'length questions_corpus_train = {len(questions_corpus_train)}')
print(f'length contexts_corpus_train = {len(contexts_corpus_train)}')
print(f'length questions_set_val = {len(questions_set_val)}')
print(f'length contexts_corpus_val = {len(contexts_corpus_val)}')
print(f'length val_labels = {len(val_labels)}')

# Retrieve data from the fquad validation set to build the test set:

test_set_file_path = os.path.join(os.getcwd(), "validation_set", "valid.json")

with open(test_set_file_path, encoding='utf-8') as json_file:
    questions_set_test, contexts_corpus_test, test_labels = load_data_tk.load_data_test_set(json_file)

print('\n\n-------------------------------------------------------------------------')
print(f'length questions_set_test = {len(questions_set_test)}')
print(f'length contexts_corpus_test = {len(contexts_corpus_test)}')
print(f'length test_labels = {len(test_labels)}')

# fit the custom preprocessors for unigrams and bigrams with the data (questions + contexts) of the training set: 

unigram_preprocessor = preprocessor.CorpusPreprocessor(remove_most_common = False, lemmetize = False, ngrams = 1)
unigram_preprocessor.fit(questions_corpus_train + contexts_corpus_train)

bigram_preprocessor = preprocessor.CorpusPreprocessor(remove_most_common = False, lemmetize = False, ngrams = 2)
bigram_preprocessor.fit(questions_corpus_train + contexts_corpus_train)

# make predictions on the validation set questions using a mixt tf-idf model: 

predictions, t_pred = models.mixt_model_predict_new_instances(questions_set_val, contexts_corpus_val, unigram_preprocessor, bigram_preprocessor, 3, predict_n = 5)

# compute the accuracy mesure for the built tf-idf model :

TOP_1_acc = models.compute_TOP1_accuracy([pred[0] for pred in predictions], val_labels)
TOP_5_acc = models.compute_TOPN_accuracy(predictions, val_labels)

print('\n\n-------------------------------------------------------------------------')
print(f'val TOP1_accuracy ---> {TOP_1_acc}')
print(f'val TOP5_accuracy ---> {TOP_5_acc}')
print(f'mean prediction time ---> {t_pred}')

# Retrain the final model on the whole fquad train data_set :

unigram_preprocessor = preprocessor.CorpusPreprocessor(remove_most_common = False, lemmetize = False, ngrams = 1)
unigram_preprocessor.fit(questions_corpus_train + contexts_corpus_train + list(set(questions_set_val)) + contexts_corpus_val)

bigram_preprocessor = preprocessor.CorpusPreprocessor(remove_most_common = False, lemmetize = False, ngrams = 2)
bigram_preprocessor.fit(questions_corpus_train + contexts_corpus_train + list(set(questions_set_val)) + contexts_corpus_val)


# compute test accuracy:

predictions, t_pred = models.mixt_model_predict_new_instances(questions_set_test, contexts_corpus_test, unigram_preprocessor, bigram_preprocessor, 3, predict_n = 5)
TOP_1_acc = models.compute_TOP1_accuracy([pred[0] for pred in predictions], test_labels)
TOP_5_acc = models.compute_TOPN_accuracy(predictions, test_labels)

print('\n\n-------------------------------------------------------------------------')
print(f'test TOP1_accuracy ---> {TOP_1_acc}')
print(f'test TOP5_accuracy ---> {TOP_5_acc}')
print(f'mean prediction time ---> {t_pred}')