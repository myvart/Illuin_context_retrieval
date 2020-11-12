The goal of this project is to produce a context retrieval machine which, given a set of questions and a set of contexts in French language,
associate the right context to a given question. This is a first step in the process of building a Question Answering machine for the French language.

The data structure we consider in this project is the fquad database's one (https://fquad.illuin.tech/).

The code has been split into several python files :
- main.py is the main script which performes the operation of loading the data from the fquad data base, building a model and evaluating the model
- load-data.py is a python module which provides custom functions to load the data from the json file containing the fquad database (downloaded from https://fquad.illuin.tech/)
- preprocessor.py is a python module that provides several custom functions for the preprocessing operations and a custom object based on the scikit learn Transformer model which automates those operations
- models.py is a python module which provides custom functions making predictions (predicting the most likely contexts for a set of questions) based on several models
