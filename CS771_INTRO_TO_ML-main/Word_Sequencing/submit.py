import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE


class BigramDecisionTree:
    def _init_(self):
        self.tree = None
        self.bigram_list = None
    
    def extract_bigrams(self, word):
        bigrams = set(word[i:i+2] for i in range(len(word)-1))
        return sorted(bigrams)[:5]
    
    def prepare_features(self, words):
        bigram_set = set()
        for word in words:
            bigram_set.update(self.extract_bigrams(word))
        bigram_list = sorted(bigram_set)
        self.bigram_list = bigram_list
        
        features = []
        for word in words:
            bigrams = self.extract_bigrams(word)
            features.append([1 if bigram in bigrams else 0 for bigram in bigram_list])
        
        return np.array(features)
    
    def fit(self, words):
        X = self.prepare_features(words)
        y = np.array(words)
        self.tree = DecisionTreeClassifier()
        self.tree.fit(X, y)
    
    def predict(self, bigrams):
        if self.tree is None or self.bigram_list is None:
            raise ValueError("Model is not trained yet.")
        
        feature_vector = [1 if bigram in bigrams else 0 for bigram in self.bigram_list]
        return self.tree.predict([feature_vector])
    
################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to train your model using the word list provided
    
    model = BigramDecisionTree()
    model.fit(words)
    return model  # Return the trained model

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to predict on a test bigram_list
    # Ensure that you return a list even if making a single guess
    guess_list = model.predict(bigram_list)
    return guess_list  # Return guess(es) as a list



