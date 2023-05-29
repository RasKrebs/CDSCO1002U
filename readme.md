# Natural Language Processing and Text Analytics Exam
## Exam Project for Course: `CDSC2O1002U`

This is the related repository for the project *Comparing the Effectiveness of Various Classification Techniques for Twitter Sentiment Analysis: A Comparative Study*. The repo is structured as follow:
```
.
├── Bert_SNN_training.ipynb
├── BERT_eval.ipynb
├── SNN_eval.ipynb
├── DataExploration.ipynb
├── LogisticRegression.ipynb
├── MultinomialNB.ipynb
├── RandomForrest.ipynb
├── data/
│   └── .train_tweets_processed.csv
├── model/
│   ├── BERT/
│   └── ShallowNeuralNetwork/
└── modules/
    └── utils.py

```

Within the root folder, all python notebooks for data exploration to data modeling and evaluation can be found.
* `DataExploration.ipynb`: This notebook is used for  performing EDA
* `LogisticRegression.ipynb`: Within this notebook, both logistic regression models are create and evaluated
* `MultinomialNB.ipynb`: In this notebook, a Multinomial Naive Bayes model is built and evaluated
* `RandomForrest.ipynb`: This notebook creates and evaluate both Random Forest models used for this research
* `Bert_SNN_training.ipynb`: This notebook is used to built and train the Shallow Neural Network and fine-tune the language model BERT. The intention is to run this notebook in `Google Colab`, due to their GPU support
* `BERT_eval.ipynb`: This is a notebook used to evaluate the fine-tuned BERT model.
* `SNN_eval.ipynb`: Similarly to the BERT notebook, this notebook is used to evaluate the Shallow Neural Network after having been trained on Google Colab.

Inside the folders of this repo, additional project requirements can be found.
* `── data/` stores the processed data, which is easily extracted using a custom created class
* `── model/` this folder stores all trained models, namely Logit, MNB, Random Forest, SNN and BERT, as well as the models for performing Word2Vec and BERT tokenization.
* `──modules/` This folder contains the python script storing all custom built classes, that have been used across the various notebooks. The purpose of these classes have been to stream line the model building process, by enabling fast and easy extraction of processed data, applying text representation techniques easily and perform common operations on the Scikit-Learn models. In sum, the classes saved here enable us to comply with DRY principles.

*Note, the fine-tuned BERT model have been excluded due to large size, that exceeded Github's size limit of 100mb. Therefore, if BERT was to be evaluated, it is recommended to open this repo from the zipped version that was attached the final exam paper hand-in. This will include the necessary files for BERT, ensuring that the evaluation there is possible*.
