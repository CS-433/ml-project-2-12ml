# CS-433 Machine Learning class project 2: Commonsense causality reasoning

## Introduction

This project examines the use of pre-trained language models (PTMs) for commonsense causality reasoning (CCR).

Models evaluated:

* Discriminative: BERT, DeBERTa-v3

* Autoregressive: GPT-2, GPT-Neo (`gpt-neo-1.3B`), GPT-3 (`text-davinci-003`)

Datasets used:

* COPA

* e-CARE

Please see the report for details and citations.

## Code structure

There are two sets of experiments in this project: fine-tuning and in-context learning.
The code for fine-tuning can be found in the `finetune` directory.
The code for in-context learning can be found in the `incontext-learning` directory.

We mainly work with Jupyter notebooks on Google Colab for their GPUs.
Because of this, each notebook is designed to be as self-contained and independent as possible.
We provide the notebooks in both directories.
The general code flow of a notebook is:

1. It downloads the datasets and installs required packages

2. It preprocesses the dataset

3. It downloads the models and related files.

4. It (trains and) evaluates the models.

Please refer to each aforementioned directory for more details.
Please note that the notebooks might not work smoothly when running locally because they are designed for usage on Colab.

We also provide `requirements.txt`, a list of necessary packages to run the code.
Please note that this is just a reference list since we run the code on Colab, in which there are a lot of pre-installed packages.
So we are not sure whether the list is exhaustive or not.