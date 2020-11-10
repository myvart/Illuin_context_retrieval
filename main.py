import os
import preprocessor
import load_data_tk
import models

train_set_file_path = os.path.join(os.getcwd(), "train_set", "train.json")

with open(train_set_file_path, encoding='utf-8') as json_file:
    questions_corpus_train, contexts_corpus_train, questions_set_val, contexts_corpus_val, val_labels = load_data_tk.load_data_train_set(json_file, test_size = 0.2)

print(f'length questions_corpus_train = {len(questions_corpus_train)}')
print(f'length contexts_corpus_train = {len(contexts_corpus_train)}')
print(f'length questions_set_val = {len(questions_set_val)}')
print(f'length contexts_corpus_val = {len(contexts_corpus_val)}')
print(f'length val_labels = {len(val_labels)}')

test_set_file_path = os.path.join(os.getcwd(), "validation_set", "valid.json")

with open(test_set_file_path, encoding='utf-8') as json_file:
    questions_set_test, contexts_corpus_test, test_labels = load_data_tk.load_data_test_set(json_file)

print(f'length questions_set_test = {len(questions_set_test)}')
print(f'length contexts_corpus_test = {len(contexts_corpus_test)}')
print(f'length test_labels = {len(test_labels)}')

preprocessor = preprocessor.CorpusPreprocessor(remove_most_frequent = True, most_common_threshold = 100, lemmetize = True)
preprocessor.fit(questions_corpus_train + contexts_corpus_train)

predictions = models.predict_new_instances(questions_set_val, contexts_corpus_val, preprocessor)

acc = models.compute_accuracy(predictions, val_labels)

print(f'val accuracy ---> {acc}')