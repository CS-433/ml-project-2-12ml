# In-context learning for commonsense causality reasoning

We evaluate three models in this experiment:

* GPT-3 (`text-davinci-003`): `run_gpt3.ipynb`, `run_gpt3.py`, `process_results.py`

* GPT-Neo, GPT-2: `harness.py`

## GPT-3:

The only way to interact with this model (175 billion parameters) is via OpenAI's API, which gives free 18 dollars of credits at the beginning.
So the code should be run with care to avoid running out of free credits (which we did).

For clarity and producibility, the code for running and evaluating GPT-3 is compiled **after** the results in the `results` folder have been collected.
Because we have run out of credits, we are unable to actually test `run_gpt3.ipynb` (`process_results.py` should be able to work correctly).
Nevertheless, the code logic of those three files should be as close as possible to the code we used to collect the results.

### EleutherAI's Language Model Evaluation Harness

For convenience, we use their code to test zero-shot and few-shot learning for GPT-2 and GPT-Neo.
However, for evaluation on e-CARE, some modifications to their code is required. Please see `harness.py` for more details.
Please note that this notebook requires a GPU, and is thus best run on Colab.