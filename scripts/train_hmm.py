#!/usr/bin/env python
import numpy as np
from hmmlearn import hmm 
import pickle

def bic_score(sample_size, log_likelihood, n_param):
    # BIC = k * log(n) - 2*log(L)
    # k = number of indipendent variables to build model
    # L = maximum likelihood estimate of model
    # n = sample size so length of train_data = sample_size
    value = (n_param * np.log(sample_size) - (2 * log_likelihood))
    return value

def train_hmm(filename):
    # the data in filename are already preprocessed so we can just train the network with several states and select the best model
    train_data = np.loadtxt(filename, delimiter=',')
    BICs = []
    models = []
    min_states = 2
    max_states = 20
    for state in range(min_states, max_states + 1):
        model = hmm.GaussianHMM(n_components=state, covariance_type="diag")
        model.fit(train_data)
        # for hmmlearn 0.2.2 in Python 2.7 I do not have the bic attribute.
        features = model.n_features
        free_param = 2 * (features * state) + state * (state - 1) + (state - 1)
        sample_size = len(train_data)
        bic = bic_score(sample_size, model.score(train_data), free_param)
        BICs.append(bic)
        models.append(model)

    best_model_idx = np.argmin(BICs)
    best_model = models[best_model_idx]

    # TRY TO REPLICATE THE SAME HMM WITH 9 STATES USED IN THE OFFLINE PROCEDURE WITH CSV FILES
    best_model = hmm.GaussianHMM(n_components=9, covariance_type="diag")
    best_model.fit(train_data)
    print("Best model has a number of states equal to: " + str(best_model_idx + 2))
    filename = "hmm_berardo.pkl"
    with open(filename, "wb") as file: 
        pickle.dump(best_model, file)
    
if __name__ == '__main__':
    # filename is in the catkin_ws folder
    # REMEMBER TO DELETE FIRST ROW WITH TITLES IF NEEDED, OTHERWISE ERROR
    filename = 'src/anomaly_detection/data/csv/preprocess_data_ros/nominal_0.csv'
    train_hmm(filename)
