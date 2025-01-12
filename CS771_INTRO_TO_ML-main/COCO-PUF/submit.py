import numpy as np
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

def compute_tu_k(c):
    k = len(c)
    tu_k = np.zeros(2*k-1)

    for i in range(k):
        tu_k[i] = 1 - 2*c[i]

    for i in range(k, 2*k-1):
        prod = 1
        for j in range(i-k, k):
            prod *= tu_k[j]
        tu_k[i] = prod
    return tu_k

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y0_train, y1_train):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to train your models using training CRPs
    # X_train has 32 columns containing the challenge bits
    # y0_train contains the values for Response0
    # y1_train contains the values for Response1
    
    # THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
    # If you do not wish to use a bias term, set it to 0

    # Map the training data
    X_train_mapped = my_map(X_train)
    model0 = LinearSVC(C=10, loss="hinge", max_iter=10000, tol=1e-6)
    model1 = LinearSVC(C=10, loss="squared_hinge", max_iter=10000, tol=1e-6)
    
    # Fit the models
    model0.fit(X_train_mapped, y0_train)
    model1.fit(X_train_mapped, y1_train)

    # Extract coefficients and intercepts
    w0, b0 = model0.coef_, model0.intercept_
    w1, b1 = model1.coef_, model1.intercept_

    return w0, b0, w1, b1

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to create features.
    # It is likely that my_fit will internally call my_map to create features for train points
    n_samples = X.shape[0]
    k = X.shape[1]
    feat = np.zeros((n_samples, 2*k-1))

    for i in range(n_samples):
        c = X[i]
        feat[i] = compute_tu_k(c)
    
    return feat
