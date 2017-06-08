import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    logL_max = float('-inf')
    best_word = ''
    X_all = test_set.get_all_Xlengths()
    for key, value in X_all.items():
        prob_dict = {}
        logL_max = float('-inf')
        X, lengths = X_all[key]
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except:
                logL = float('-inf')
            if logL > logL_max:
                logL_max = logL
                best_word = word
            prob_dict[word] = logL
        guesses.append(best_word)
        probabilities.append(prob_dict)
    return probabilities, guesses
