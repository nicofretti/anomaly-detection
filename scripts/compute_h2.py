import numpy as np


def compute_h2_score(files):
    h2_data = np.loadtxt(files[0], delimiter=',', skiprows=0)
    for filename in files[1:]:
        data = np.loadtxt(filename, delimiter=',')
        np.concatenate((h2_data, data), axis=0)
    # for each column, compute the mean
    standard_deviation = np.std(h2_data, axis=0)
    mean = np.mean(h2_data, axis=0)
    standard_deviation_error_of_mean = np.std(h2_data, axis=0) / np.sqrt(len(h2_data))
    standard_deviation_weights = np.array([2, 2, 0, 0, 0, 0])
    standard_deviation_error_of_mean_weights = np.array([0, 0, 1, 1, 1, 1])
    return mean + (standard_deviation_error_of_mean * standard_deviation_error_of_mean_weights) + (standard_deviation * standard_deviation_weights)


def compute_hellinger_score(files):
    h_data = np.loadtxt(files[0], delimiter=',')
    for filename in files[1:]:
        data = np.loadtxt(filename, delimiter=',')
        np.concatenate((h_data, data), axis=0)
    # for each column, compute the mean
    standard_deviation_error_of_mean = np.std(h_data, axis=0) / np.sqrt(len(h_data))
    mean = np.mean(h_data, axis=0)
    standard_deviation = np.std(h_data, axis=0)
    return mean + abs(standard_deviation_error_of_mean) + standard_deviation


if __name__ == "__main__":
    h2_scores = compute_h2_score([
        # "../data/bag_files/train/scores/03_h2.csv",
        #"../data/bag_files/train/scores/02_h2.csv",
        "../data/bag_files/train/scores/04_h2.csv",
        "../data/bag_files/train/scores/07_h2.csv",
        # "../data/bag_files/train/scores/09_h2.csv",
    ])
    h_score = compute_hellinger_score([
        # "../data/bag_files/train/scores/03_he.csv",
        #"../data/bag_files/train/scores/02_he.csv",
        "../data/bag_files/train/scores/04_he.csv",
        "../data/bag_files/train/scores/07_he.csv",
        # "../data/bag_files/train/scores/09_he.csv",
    ])
    print(",".join([str(x) for x in h2_scores]))
    #print(str(h_score))
