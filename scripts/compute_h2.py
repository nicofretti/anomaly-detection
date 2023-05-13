import numpy as np


def compute_h2_scores(files):
    h2_data = np.loadtxt(files[0], delimiter=',', skiprows=0)
    for filename in files[1:]:
        data = np.loadtxt(filename, delimiter=',')
        np.concatenate((h2_data, data), axis=0)
    # for each column, compute the mean
    standard_deviation = np.std(h2_data, axis=0)
    mean = np.mean(h2_data, axis=0)
    return mean + 5 * standard_deviation

if __name__ == "__main__":
    h2_scores = compute_h2_scores([
        "../data/bag_files/train/scores/03_h2.csv",
        "../data/bag_files/train/scores/04_h2.csv",
        "../data/bag_files/train/scores/07_h2.csv",
        "../data/bag_files/train/scores/09_h2.csv",
    ])
    print(",".join([str(x) for x in h2_scores]))
