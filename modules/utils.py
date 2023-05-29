# Data handling
import pandas as pd
from datasets import load_dataset
import numpy as np

# Data processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from urllib.parse import urlparse
import re

# General
import os

# Tweet vectorizers
import gensim
from gensim.models import Word2Vec, phrases
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer


# Model Libraries
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# General
from joblib import dump, load
import multiprocessing
import os
import matplotlib.pyplot as plt



ROOT_DIR_PATH = os.path.abspath('.')
DATA_PATH = os.path.join(ROOT_DIR_PATH, 'data')
MODEL_PATH = os.path.join(ROOT_DIR_PATH, 'models')


class SentimentDataset:
    """Dataset class that will perform all the processing and enable easy extraction across our modelling"""
    def __init__(self, raw_input = False, size = 200000, subset = 'train', full_refresh=False):
        """Params:
        * size: Desired size for Dataset (Default 200.000)
        * subset: HuggingFace dataset subset. Either 'train' or 'test' (Default 'train')
        * full_refresh: Instead of using cached pre-processed dataset, extract original version (Default = False)


        Methods: These are only applied if full_refresh or non-cached pre-processed dataset
        * process_data(): Will process raw data to desired and downsize to desired size

        """
        self.subset = subset # Specify the subset
        if self.subset == 'test':
            self.raw_input = raw_input

        elif self.subset == 'train':
            self.size = size # Size of dataset
            if size % 2 != 0:
                raise ValueError('Size must be divisable by 2') # Raise error if size not divisable by 2. Done since there must be equal distribution of sentiments in dataest
            self.processed = False # Wether data is processed or not
            self.ROOT_DIR_PATH = ROOT_DIR_PATH # Current directory path
            self.DATAPATH = DATA_PATH # Path to datafolder (processed data will be stored here)
            self.__processed_file__ = f'.{self.subset}_tweets_processed.csv' # File name for processed file (if any)

            if not os.path.exists(self.DATAPATH):
                os.mkdir(self.DATAPATH)
            self.load_data(full_refresh)
        else:
            raise ValueError('Only takes train or test as subest input')

        self.subset = subset # Specify the subset
        self.stop_words = stopwords.words('english') # Initalizing stopwords that are used for processing
        self.lemmatizer = WordNetLemmatizer() # Initializin lemmatizer that are used for processing

    def load_data(self, full_refresh = False):
        """Method for loading data from HuggingFace datasets API. Will fetch pre_processed data from data folder"""

        print('Loading data...')
        self.dataset = load_dataset("sentiment140", split=self.subset)  # Load data from Huggingface

        if self.__processed_file__ in os.listdir(self.DATAPATH) and not full_refresh: # If processed file in data folder load that instead
            print('\nLoading processed dataset from cache...')
            self.df = pd.read_csv(f'{self.DATAPATH}/{self.__processed_file__}') #load preprocessed data from data folder
            self.df = self.df[['text','sentiment', 'text_processed']] # Remove redundant columns
            self.processed = True # Variable that tracks if processing has been applied
            if len(self.df) != self.size: # Notify if size of dataset is less than desired size specified at initialization
                print(f'\nProcessed DataFrame is not of desired length. To overwrite and re-process, run self.load_data(full_refresh = True)')
        else: # Create pandas dataset from HuggingFace
            print('\nGenerating dataframe from Hugginface dataset')
            self.df = self.dataset.to_pandas() # Transform Huggingface dataset into pandas DataFrame
            self.processed = False
        print('Dataset loaded')
        return self

    def process_data(self, full_refresh = False, remove_stopwords=True):
        """Method for processing raw data.
        Params:
        * full_refresh: Similarly to when initializing class, this enables to extract original un-processed data. If full_refresh is specified during object initialization, then this can be left to default (default = False)
        * remove_stopwords: Inclue stopwords or not (default = True)
        """
        try:
            if self.processed and not full_refresh and self.subset == 'train': # Only process if processing has not been applied
                print(f'Dataset is already processed. To overwrite and re-process, run self.process(full_refresh = True)')
                return self

            elif self.processed and full_refresh and self.subset=='train': # Overwrite processing and re-perform
                self.load_data(full_refresh=True)
                self.process_data()
                return
        except:
            print('Processing')
            if self.subset == 'train':
                self.df = self.df[['text', 'sentiment']] # Exclude irrelevant columns
                self.df.sentiment = self.df.sentiment.replace(4, 1) # Replace positive value 4 with 1
                self.df = self.df.groupby('sentiment').apply(lambda x: x[:int(self.size+2000/2)]).reset_index(drop=True) # Reducing rows to desired size + 1000 (1000 extra since some tweets will be null after processing)
                self.df['text_processed'] = self.df.text.apply(self.__text_processor, remove_stopwords=remove_stopwords) # Applying text processor
                self.df.text_processed.replace('', np.nan, inplace=True) # Some values become empty strings, therefore we'll replace these
                self.df = self.df.dropna() # Removing these rows of empty strings
                self.df = self.df.groupby('sentiment').apply(lambda x: x[:int(self.size/2)]).reset_index(drop=True) # Once again resizing
                self.processed = True

                # Extracting all words from processed data
                self.non_unique_words = ' '.join(str(text) for text in self.df['text_processed'].to_list()).split()

                # Filtering to for only unique words
                self.unique_words = set(self.non_unique_words)
                self.df.to_csv(f'{self.DATAPATH}/{self.__processed_file__}')
            elif self.subset == 'test':
                print(f'Processing test input')
                self.processed_input = [self.__text_processor(text) for text in self.raw_input]
                return self.processed_input
            else:
                pass

        print('Dataset processed')




    def __text_processor(self, text, remove_stopwords = True):
        text = re.sub(r"@\w+\s*|#\w+\s*", "", text) # Removing hastags and at-signs
        text = re.sub(r'[][)(]', ' ', text) # Removing brackets from tweets
        text = re.sub(r'([;:])\w+\s*', ' ', text) # Removing smileys
        text = re.sub(re.compile('<.*?>'), '', text) # Removing text enclosed in angle brachets - typical for HTML tags
        text = [w for w in text.split() if not urlparse(w).scheme] # Removing URLs: List comprehension to remove any 'words' with a HTTP layer
        text = ' '.join(text) # Rejoining after list comprehension
        text = re.sub('[^A-Za-z0-9]+', ' ', text) # Removing everything that is not a number of text
        text = text.lower() # Lowercasing all letters
        text = nltk.word_tokenize(text) # Tokenizing text

        if remove_stopwords:
            text = [word for word in text if word not in self.stop_words] # Removing stopwords

        text = [self.lemmatizer.lemmatize(word) for word in text] # Applying lemmatization
        text = ' '.join(text).strip()  # rejoining text into continous string

        return text

    def __repr__(self):
        if self.subset == 'train':
            return f'Subset type: {self.subset}\nLength: {len(self.df)}\nProcessed: {self.processed}'
        elif self.subset == 'test':
            return self.processed_input
        else:
            pass

    def w2v_vectorize(self):
        raw_input = self.processed_input
        vectors = np.zeros((len(raw_input), 300))

        for tweet in range(len(raw_input)):
            # Initalize empty vector to pass word embeddings
            vector = np.zeros(300).reshape((1, 300))
            counter = 0  # Counter to count words
            for word in raw_input[tweet]:
                try:
                    vector += self.w2v.wv[word].reshape((1, 300))
                    counter += 1
                except:
                    continue

            if counter != 0:
                vector /= counter

            vectors[tweet] = vector

        return vectors

    def tfidf_vectorize(self, raw_input):
        vectors = self.tfidf.transform(raw_input)
        return vectors


# Sklearn Models
class SklearnModels:
    """Class to initalize Sklearn models on. This should help reduce the need for repeated code and enable easy and fast execution of commands across models"""
    def __init__(self, model = None, input_data=False, target_labels = False, data_model = None, model_name:str = None, grid_search:bool = False, model_params:dict={}, split:bool=True, test_size=0.2, cv=5, verbose=2, scale=True): # Enabling model initalization without params (to e.g. load model)
        """Params:
        * model: Sklearn classifier model
        * data_model: Dataset object (SentimentDataset)
        * model_name: Name of classifier model. This is used for saving and loading.
        * grid_search: Boolean of wether or not grid search should be applied.
        * model_parmas: Parameters to test for during grid serach hyperparam tuning
        * split: Boolean of wether or not to split into training and test
        * test_size (Default = 0.2): Test size to use if splitting
        * cv: Number of folds to perform during cross validation
        * verbose: For GridSearchCV. Raises error if iteration fails.
        * scale: Wether or not to apply StandardScaler to data
        """
        self.model = model #Sklearn model
        self.model_params = model_params # Model params for hyper-param tuning
        self.model_name = model_name # Name of model to save
        self.cpu_count = multiprocessing.cpu_count() # Count CPUs
        self.grid_search = grid_search # Bool of wether or not to perform grid_search
        self.tuned = False if grid_search else True # If grid search has been applied, this will be true
        self.cv = cv # number of cv
        self.verbose = verbose # Verbose for crossvalidation
        self.data_model = data_model # Data model (SentimentDataset object)
        self.ROOT_DIR_PATH = os.getcwd() # Current directory path
        self.models_path = MODEL_PATH # Path to model folder (trained models will be stored here)
        self.model_fitted = False # Bool of wether or not model is fitted
        self.test_prediction_done = False # Bool for keeping track of wether test prediction is performed
        self.classification_report = None # Variable for classification_report if done
        self.accuracy = None  # Variable for accuracy if done
        self.confusion_matrix = None  # Variable for confusion matrix if done
        self.scale = scale

        if split:
            self.generate_training_and_test_set(split, test_size)
        else:
            self.X_test = input_data
            self.y_test = target_labels
            self.model_fitted = True

    def generate_training_and_test_set(self, split, test_size): # Method for subsetting data
        self.data = self.data_model.vector

        if self.scale: # Apply StandardScaler to data
            self.data = StandardScaler().fit_transform(self.data)

        self.target = self.data_model.target
        self.test_size = test_size

        if split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=test_size)


    def fit(self): # Method for fitting models
        if self.grid_search:
            print(f'Performing grid search with {self.cv}-fold cross validation\n')
            self.model = GridSearchCV(estimator=self.model,
                                      param_grid=self.model_params,
                                      verbose=self.verbose,
                                      cv = self.cv,
                                      error_score='raise',
                                      n_jobs = self.cpu_count-1,
                                      return_train_score = True)
            self.tuned = True
        self.model.fit(self.X_train, self.y_train) # Fitting model

        if self.grid_search:
            print(f'Best params: {self.model.best_params_}')

        self.model_fitted = True

    def evaluate_on_test(self, accuracy = True, classification_rep = False, confusion_mat = False, print_out = True, normalize='true'): # Method for evaluating on test set
        if self.model_fitted and self.tuned: # Check if is fitted and tune
            if not self.test_prediction_done:
                self.test_prediction = self.model.predict(self.X_test)
                self.test_prediction_done = True

            if classification_rep:
                self.classification_report = classification_report(self.y_test, self.test_prediction)
                if print_out:
                    print('Classification Report:\n', self.classification_report)

            if confusion_mat:
                self.confusion_matrix = confusion_matrix(self.y_test, self.test_prediction, labels=self.model.classes_, normalize=normalize)
                if print_out:
                    print('Confusion Matrix:\n', self.confusion_matrix)

            if accuracy:
                self.accuracy = accuracy_score(self.y_test, self.test_prediction)
                if print_out:
                    print('Test Accuracy: ', self.accuracy)

        else:
            print('Model not fitted. Run self.fit() to fit model with data')

    def plot_confusion_matrix(self, normalize=True):
        """Method for plotting confusion"""

        # Initalize matplotlib figure
        figure = plt.figure(figsize = (10,8))

        # Create display object for confusion matix
        try:
            cmd = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix)
        except:
            print('Confusion matrix has not been calculated. Run self.evaluate_on_test(confusion_mat=True) and then re-run function')
            return


        # Plot
        cmd.plot()
        plt.title(f'Confusion Matrix: {type(self.model).__name__}')
        plt.show()

    def save(self):
        # method for saving trained model
        if not self.model_name:
            print('Model name must be specified. Run self.model_name = "Model Name" and re-try method.')
            return

        dump(self.model, f'{self.models_path}/{self.model_name}.joblib')

    def load_model(self):
        # Method for loading from models folder
        if not self.model_name:
            print('Model name must be specified. Run self.model_name = "Model Name" and re-try method.')
            return

        print('Loading model')
        self.model = load(f'{self.models_path}/{self.model_name}.joblib')
        self.model_fitted = True
        return self




class w2v:
    """Class for performing word Word2Vec embedding on dataset"""
    def __init__(self,
                SentimentDatasetObject,
                return_vector = False,
                target=None,
                w2v_model_name = 'word2vec.model'):

        """Params:
        * SentimentDatasetObject: Data object initiated on SentimentDataset class
        * w2v_model_name: Name of pretrained Word2Vec model (default = 'word2vec.model')
        """
        print('Applying Word2Vec embedding')
        if type(SentimentDatasetObject) == list:
            self.texts = [str(text).split() for text in SentimentDatasetObject]
            self.phrases = Phrases(self.texts, min_count=2, progress_per=1000, connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
            self.bigram_transformer = Phraser(self.phrases)
            self.tweets = self.bigram_transformer[self.texts]
            self.__load_w2v_model__()

        else:
            if not SentimentDatasetObject.processed:
                raise TypeError('Data has not been processed')

            self.SentimentDatasetObject = SentimentDatasetObject  # Save dataset object

            self.texts = [str(text).split() for text in SentimentDatasetObject.df['text_processed'].to_list()]

            self.phrases = Phrases(self.texts, min_count=2, progress_per=1000, connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
            self.bigram_transformer = Phraser(self.phrases)
            self.tweets = self.bigram_transformer[self.texts]
            self.SentimentDatasetObject.tweets = self.tweets
            self.target = self.SentimentDatasetObject.df.sentiment.to_numpy()
            self.__load_w2v_model__()
            self.SentimentDatasetObject.w2v = self.w2v

        self.vector = np.zeros((len(self.tweets.corpus), self.size))
        print(f'Vectorizing {range(len(self.tweets.corpus))} tweets')

        for tweet in range(len(self.tweets.corpus)):
            self.vector[tweet] = self.vectorize(self.tweets.corpus[tweet])


        print('Word2Vec embedding applied')
        try:
            self.SentimentDatasetObject.vector = self.vector
            self.SentimentDatasetObject.target = self.target
            self.SentimentDatasetObject.embedding_type = 'Word2Vec'
        except:
            pass

        if return_vector:
            self.__return_vector()

    def __load_w2v_model__(self):
        try: # load w2v model
            print('Loading Word2Vec model from directory')
            self.w2v = Word2Vec.load(f"{MODEL_PATH}/word2vec.model")
        except:
            raise TypeError(f'No file in {MODEL_PATH}/word2vec.model')

        self.size = self.w2v.wv.vector_size


    def __return_vector(self):
        return self.vector

    def vectorize(self, token):
        vector = np.zeros(self.size).reshape((1, self.size)) # Initalize empty vector to pass word embeddings
        counter = 0 # Counter to count words

        for word in token:
            try:
                vector += self.w2v.wv[word].reshape((1, self.size))
                counter += 1
            except:
                continue

        if counter != 0:
            vector /= counter

        return vector


class tf_idf:
    """Class for performing word tf-idf embedding on dataset"""
    def __init__(self,
                SentimentDatasetObject=False, return_vector=False):

        """Params:
        * SentimentDatasetObject: Data object initiated on SentimentDataset class
        """
        self.SentimentDatasetObject = SentimentDatasetObject  # Save dataset object
        try:
            self.corpus = self.SentimentDatasetObject.df.text_processed.to_list()
            self.target = self.SentimentDatasetObject.df.sentiment.to_numpy()
        except:
            self.corpus = self.SentimentDatasetObject

        self.tfidf = TfidfVectorizer(ngram_range = (1,3), min_df=3)
        self.vector = self.tfidf.fit_transform(self.corpus)

        print('TF-IDF embedding applied')


        self.SentimentDatasetObject.vector = self.vector
        self.SentimentDatasetObject.target = self.target
        self.SentimentDatasetObject.embedding_type = 'TF-IDF'
        self.SentimentDatasetObject.tfidf = self.tfidf

        if return_vector:
            self.__return_vector()

    def __return_vector(self):
        return self.vector





















class WordEmbedder:
    """Class for building word embedding on dataset"""

    def __init__(self,
                SentimentDatasetObject,
                type='Word2Vec',
                w2v_model_name = 'word2vec.model',
                overwrite=False,
                vec_size=200,
                apply_gensim_phrases=True,
                ngram_range:tuple =(1, 3),
                min_document_frequency:int=3):

        """Params:
        * SentimentDatasetObject: Data object initiated on SentimentDataset class
        * type: Type of word embedding. Either 'Word2Vec' or 'tf-idf' (default = 'Word2Vec')
        * w2v_model_name: Name of pretrained Word2Vec model (default = 'word2vec.model')
        * overwrite: Boolean of wether or not to overwrite any pretrained Word2Vec models and re-train on data (default = False)
        * apply_gensim_phrases: Boolean of wether or not to perform Gensim Phrases method on corpus, to detect phrases appearing more common than expected (default = True).
        * ngram_range (tuple): For tf-idf, specifies the lower and upper boundary of the range of n-values for different n-grams to be extracted (default = (1,3))
        * min_document_frequency: For tf-idf specifies the minimum number of documents a word must appear in otherwise it will be dropped.
        """

        # Raising error if Dataset object is not included
        if not SentimentDatasetObject:
            raise TypeError('Word embedder can only be initialized with SentimentDataset object.')

        self.SentimentDatasetObject = SentimentDatasetObject  # Save dataset object
        self.corpus = [str(text) for text in SentimentDatasetObject.df.text_processed.to_list()] # Generate list of tweets
        self.words = [text.split() for text in self.corpus] # Tokenize all lists (for gensim phraser)
        self.gensim_phrases_applied = False # boolean value for knowing if gensim phraser applied
        self.MODEL_PATH = MODEL_PATH  # Path to models folder
        self.w2v_model_name = w2v_model_name  # Name of word2vec model
        self.w2v = None
        self.apply_gensim_phrases = apply_gensim_phrases
        self.SentimentDatasetObject.word_embedder = None
        self.vec_size = vec_size
        if type == 'Word2Vec':
            if apply_gensim_phrases:
                print('Applying Gensim Phrases')
                self.gensim_phrases()
            self.initialize_Word2Vec_model(overwrite=overwrite)
            self.apply_w2v()
        elif type == 'tf-idf':
            self.initialize_tf_idf(ngram_range=ngram_range, min_document_frequency=min_document_frequency)
        else:
            raise ValueError("Embedding type must be one of: 'Word2Vec' or 'tf-idf'")

    def gensim_phrases(self):
        """Method for performing Gensim Phrases on corpus - detects bigrams that are frequently occuring than expected"""

        print('\nApplying Gensim Phrases for detecting common bigram phrases')
        self.phrases = Phrases(self.words, min_count=2, progress_per=1000,
                               connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
        self.bigram_transformer = Phraser(self.phrases)
        self.gensim_phrases = self.bigram_transformer[self.words]
        self.SentimentDatasetObject.gensim_phrases = self.bigram_transformer[self.corpus]
        self.gensim_phrases_applied = True
        self.SentimentDatasetObject.gensim_phrases_applied = True


    def initialize_Word2Vec_model(self, overwrite=False):
        """Method for initalizing Word2vec model.
        Args:
        * overwrite (default = False): Overwrite pre-tained models and re-train
        * apply_gensim_phrases (default = True): Wether or not to apply Gensim Phrases method on corpus
        """

        # If word2vec model exists, load it
        if self.w2v_model_name in os.listdir(self.MODEL_PATH) and not overwrite:
            self.w2v = Word2Vec.load(f'{self.MODEL_PATH}/{self.w2v_model_name}')
            print('Word2Vec model loaded from directory')

        # If model not exists, build one
        else:
            # Training model using gensim phrase
            if self.apply_gensim_phrases:
                # If gensim_phrases have not been applied, apply it to the corpus
                if not self.gensim_phrases_applied:
                    self.gensim_phrases()
                # train a w2v model
                print('Training Word2Vec model')
                self.w2v = Word2Vec(
                    self.gensim_phrases,
                    vector_size=self.vec_size,  # Increased vector size from default, to try and capture more variance
                    window=4,  # context window size
                    # Ignores all words with total frequency lower than 2.
                    min_count=2,
                    sg=1)

                self.w2v.train(self.gensim_phrases, total_examples=len(self.gensim_phrases.corpus), epochs=50)
            else:
                print('Training Word2Vec model')
                self.w2v = Word2Vec(
                    self.corpus,
                    vector_size=self.vec_size,
                    window=4,
                    min_count=2,
                    sg=1)

                self.w2v.train(self.corpus, total_examples=len(self.corpus), epochs=50)

            print('Word2Vec model trained')
            self.w2v.init_sims(replace=True)

            # save model for future use
            self.w2v.save(f'{self.MODEL_PATH}/{self.w2v_model_name}')

        self.SentimentDatasetObject.w2v = self.w2v  # assign model to dataset object

        return self


    def initialize_tf_idf(self, ngram_range, min_document_frequency):
        """  Generate TF-IDF for Corpus - needed for Multinomial Naive Bayes. Arguments:
        * ngram_range:tuple (deafault = (1,3)): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        * min_document_frequency:int (default = 3): Minimum number of documents term must appear in, otherwise ignore
        """

        # Generate tfid object
        self.Tfidf = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_document_frequency)

        # Fit corpus to object
        self.Tfidf_fitted = self.Tfidf.fit_transform(self.corpus)

        # Generate tf-idf dataframe
        self.Tfidf_df = pd.DataFrame(self.Tfidf_fitted.todense(
        ), index=self.corpus, columns=self.Tfidf.get_feature_names_out())

        self.SentimentDatasetObject.Tfidf = self.Tfidf
        self.SentimentDatasetObject.Tfidf_fitted = self.Tfidf_fitted
        self.SentimentDatasetObject.word_embedder = 'tf-idf'

        print('TF-IDF object created')

        return self

    def apply_w2v(self):
        """Apply Word2Vec model on corpus. Returns variable self.vectors containing word embedded tweets"""
        vector_size = self.vec_size

        # Generate empty vector of desired size
        # corpus_size x vector_size
        vectors = np.zeros((len(self.corpus), vector_size))
        corpus = self.gensim_phrases.corpus if self.gensim_phrases_applied else self.corpus

        # Loop through every tweet to generate vector for every tweet
        for tweet in range(len(corpus)):
            # Tweet vector of size equal to vector size
            vector = np.zeros(vector_size).reshape((1, vector_size))
            counter = 0  # word counter

            for word in corpus[tweet]:
                try:
                    # Get word vector from model for every word and multiply it ontop of the existing vector
                    vector += self.w2v.wv[word].reshape((1, vector_size))
                    counter += 1
                except:  # in case word is not in w2v model, we continue without it
                    continue

            if counter != 0:
                vector /= counter  # We aggregate the vector by finding

            # Add vector to array of vectors
            vectors[tweet] = vector

        self.w2v_vectors = vectors
        self.SentimentDatasetObject.w2v_vectors = vectors
        self.SentimentDatasetObject.word_embedder = 'Word2Vec'

        return self
