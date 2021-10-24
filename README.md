# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Neural-Architecture-Search-NLP
In this repo we provided an implementation to search model for NLP based task using model search framework by google.

## Model Search Framework
Model search (MS) is a framework that implements AutoML algorithms for model architecture search at scale. It aims to help researchers speed up their exploration process for finding the right model architecture for their classification problems (i.e., DNNs with different types of layers).

The library enables you to:

Run many AutoML algorithms out of the box on your data - including automatically searching for the right model architecture, the right ensemble of models and the best distilled models.

Compare many different models that are found during the search.

Create you own search space to customize the types of layers in your neural networks.

The technical description of the capabilities of this framework are found in InterSpeech paper.

While this framework can potentially be used for regression problems, the current version supports classification problems only. Let's start by looking at some classic classification problems and see how the framework can automatically find competitive model architectures.


## Getting started
If you have a text dataset in TSV or CSV file structure and you would like to run model serach to find the best model architecture. Follow the bellow steps to begin you search:

** Note: If you are an notebook geeks then use this **
<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jagan-mathematics/NLP-Neural_Architechture_Search/blob/master/Notebooks/Model%20search%20for%20kick%20start%20experiment.ipynb)


**step 1**
clone the model search repo and move the model_search folder to project directory
```
  git clone https://github.com/jagan-mathematics/NLP-Neural_Architechture_Search.git
  git clone https://github.com/google/model_search.git
  # make sure both the repos are cloned in same path
  mv model_search/model_search NLP-Neural_Architechture_Search/
```

**step 2**
If you want to do model search with your own dataset then do some changes in
```
  /utils/Configuration.py
```
**Note**
As of now it our implementation only supports TSV or CSV file with text in first column and label in second column. We had faced lot of problems while using keras tokenizer, so we decided to create my own. It will do some basic processing like lowering text and removing special charaters.

**step 3**
now let search for the best!!!
```
  python main.py
```


## Summary
 - This notebook provides an complete implementation of model search to find the best fit model for text classification task.
 - We have created our own tokenizer class because we had some hard times with keras tokenizer.
 - Data provider that we implemented has some limitations like we can only able to use tsv or csv file format, we have to provide file as text first label second format.

## Future work
 - We are working on option to train model with pre-trained embedding layer. eg: glove, bert etc.,
 - multiple options to load dataset from provider
 - general aproach for text generation and text segmentation tasks.
 - adding option to pass preprocessing function and stop words removal property in Tokenizer class
