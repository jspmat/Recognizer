import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = 0
        min_BIC = np.inf
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                p = n** + 2 * n * model.n_features - 1
                BIC = -2 * logL + p * np.log(len(self.X))
                if BIC < min_BIC:
                    min_BIC = BIC
                    best_num_components = n
            except:
                pass
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = 0
        max_DIC = -np.inf
        avgLogL = 0
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                sumLogL = 0                
                model = GaussianHMM(n_components=n).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                for word in self.words:
                    if word != self.this_word:
                        X_temp, lengths_temp = self.hwords[word]
                        logL_temp = model.score(X_temp, lengths_temp)
                        sumLogL = sumLogL + logL_temp
                avgLogL = sumLogL / (len(self.words) - 1)
                DIC = logL - avgLogL
                if DIC > max_DIC:
                    max_DIC = DIC
                    best_num_components = n
            except:
                pass
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = 0
        best_avg = -np.inf
        if len(self.sequences) >= 3:
            for n in range(self.min_n_components, self.max_n_components + 1):
                try:
                    logL_sum = 0
                    split_method = KFold()
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        model = GaussianHMM(n_components=n).fit(X_train, lengths_train)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        logL = model.score(X_test, lengths_test)
                        logL_sum = logL_sum + logL
                    logL_avg = logL_sum / 3
                    if logL_avg > best_avg:
                        best_avg = logL_avg
                        best_num_components = n
                except:
                    pass
        else:
            best_num_components = 3
        return self.base_model(best_num_components)