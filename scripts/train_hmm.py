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


# def train_hmm(filename):
#     # the data in filename are already preprocessed so we can just train the network with several states and select the best model
#     train_data = np.loadtxt(filename, delimiter=',')
#     BICs = []
#     models = []
#     min_states = 2
#     max_states = 20
#     for state in range(min_states, max_states + 1):
#         model = hmm.GaussianHMM(n_components=state, covariance_type="diag")
#         model.fit(train_data)
#         # for hmmlearn 0.2.2 in Python 2.7 I do not have the bic attribute.
#         features = model.n_features
#         free_param = 2 * (features * state) + state * (state - 1) + (state - 1)
#         sample_size = len(train_data)
#         bic = bic_score(sample_size, model.score(train_data), free_param)
#         BICs.append(bic)
#         models.append(model)
#
#     best_model_idx = np.argmin(BICs)
#     best_model = models[best_model_idx]
#
#     # TRY TO REPLICATE THE SAME HMM WITH 9 STATES USED IN THE OFFLINE PROCEDURE WITH CSV FILES
#     best_model = hmm.GaussianHMM(n_components=9, covariance_type="diag")
#     best_model.fit(train_data)
#     print("Best model has a number of states equal to: " + str(best_model_idx + 2))
#     filename = "hmm_berardo.pkl"
#     with open(filename, "wb") as file:
#         pickle.dump(best_model, file)

def train_hmm(files, test_file):
    assert len(files) > 0, "No files to train the hmm"
    assert test_file is not None, "No test file to test the hmm"
    test_data = np.loadtxt(test_file, delimiter=',')
    train_data = np.loadtxt(files[0], delimiter=',')
    lengths = [len(train_data)]
    for f in files[1:]:
        data = np.loadtxt(f, delimiter=',')

        lengths.append(len(data))
        # add each element of data to train_data
        train_data = np.concatenate((train_data, data), axis=0)

    # add some shift to the first two columns, caused by the bag files not well synchronized with the real data
    shift = np.array([9.60, 2.46])
    train_data[:, :2] = train_data[:, :2] + shift
    # number of states to try
    min_states, max_states = 15, 22
    best_features, best_bic, best_model = 0, None, None
    for state in range(min_states, max_states + 1):
        model = hmm.GaussianHMM(n_components=state, covariance_type="diag")
        model.fit(X=train_data, lengths=lengths)
        # for hmmlearn 0.2.2 in Python 2.7 I do not have the bic attribute.
        features = model.n_features
        free_param = 2 * (features * state) + state * (state - 1) + (state - 1)
        bic = bic_score(len(test_data), model.score(test_data), free_param)
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_features = state
            best_model = model

    return best_model, best_bic, best_features


def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    # filename is in the catkin_ws folder
    # REMEMBER TO DELETE FIRST ROW WITH TITLES IF NEEDED, OTHERWISE ERROR
    filename = ''
    best_model, best_bic, best_features = False, False, False
    for i in range(5):
        model, bic, features = train_hmm(
            files=[
                "../data/bag_files/train/02_nominal_stack.csv",
                # "../data/bag_files/train/03_nominal.csv",
                "../data/bag_files/train/04_nominal.csv",
                "../data/bag_files/train/07_nominal.csv",
                # "../data/bag_files/train/09_nominal.csv",
                "../data/bag_files/train/10_nominal_true.csv",
                "../data/bag_files/train/11_nominal_true.csv",
            ],
            test_file="../data/bag_files/train/11_nominal_true.csv"
        )
        print("BIC: " + str(bic) + " with " + str(features) + " features")
        if not best_model or bic < best_bic:
            best_model, best_bic, best_features = model, bic, features
    print("Best model has a number of states equal to: " + str(best_features))

    save_model(best_model, "hmm_best.pkl")
