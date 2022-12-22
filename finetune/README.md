# Fine-tuning pre-trained models for commonsense causality reasoning

There are two notebooks in the directory. They are designed to be as self-contained as possible.
This is why there are some pieces of code that are the same across the notebooks.
But details of the code are different becaue the two classes of models work differently.
In the first code cell in each notebook, there are global variables that can be changed to produce desired results:

* `dataset`: set it to `ecare` or `copa`

* `model_checkpoint`: set it to a HuggingFace model checkpoint, e.g., `"bert-base-cased"` for BERT and `""`

* `global_seed`: set it to a number to control the randomness.

* `epochs`: the number of training epochs (fine-tuning)

* `lr`: the learning rate.

In this project, fine-tuning for CCR is a binary classification problem.

## About `non_gpt2.ipynb`

For models evaluated in this notebook, HuggingFace provides ready-made `...ForSequenceClassification` classes.
But the gist of it is that model computes embeddings for the `[CLS]` token and feeds them to a linear layer for classification.
We use HuggingFace's training utilities (`Trainer` and `TrainingArguments`) for convenience.

## About `gpt2.ipynb`

To use GPT-2 for classification, we use the its embeddings for the first padding token.
So we need to use `GPT2Model` to get those embeddings, and then add a linear layer on top of that for classification.
The whole model pipeline is then fine-tuned via the use of HuggingFace's `Trainer` and `TrainingArguments`.

Note that while there is a `GPT2ForSequenceClassification`, we chose to follow the e-CARE authors' implementation instead
because it yielded bette results.
