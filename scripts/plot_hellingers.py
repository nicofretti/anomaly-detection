import numpy as np
import matplotlib.pyplot as plt


def plot_decomposition_from_np_file(filename, title):
    labels = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
    with open(filename, 'rb') as f:
        decomposition = np.load(f)
        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle(title, fontsize=18)
        for i in range(0, 6):
            ax = plt.subplot(6, 1, i + 1)
            ax.plot(decomposition[:, i], label=labels[i])
            ax.legend()
        plt.savefig(title + ".png", bbox_inches='tight', dpi=100)
        plt.close()


def plot_hellinger(filename, title):
    with open(filename, 'rb') as f:
        hellinger = np.load(f)
        plt.figure(figsize=(20, 10))
        plt.plot(hellinger, label=title)
        plt.title(title)
        plt.legend()
        plt.savefig(title + ".png", bbox_inches='tight', dpi=100)
        plt.close()


if __name__ == "__main__":

    # PUT THE PATH IN FILES
    files = ["nominal_0_decomposition.npy",
             "nominal_1_decomposition.npy",
             "anomaly_0_decomposition.npy",
             "anomaly_1_decomposition.npy"]

    for filename in files:
        print("Processing:" + filename[:-4])
        plot_decomposition_from_np_file(filename=filename, title=filename[:-4])

    # DATASET NOMNALE 2
    h2_thr = [1.26020238, 6.67861522, 0.4251171, 0.70920265, 0.94272347, 0.89692743]

    for filename in files:
        print("Processing binary threshold for " + filename[:-4])
        with open(filename, 'rb') as f:
            decomposition = np.load(f)
            decomposition_binary = np.where(decomposition > h2_thr, 1, 0)
            plt.figure(figsize=(20, 10))
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle(filename[:-4] + "_binary", fontsize=18)
            for i in range(0, 6):
                ax = plt.subplot(6, 1, i + 1)
                ax.plot(decomposition_binary[:, i], label=filename[:-4] + "_binary")
                ax.legend()
            plt.savefig(filename[:-4] + "_binary.png", bbox_inches='tight', dpi=100)
            plt.close()

    threshold_nominal_1_data = 0.9994715267699249
    files_hellingers = [
        "hellingers_nominal_0.npy",
        "hellingers_nominal_1.npy",
        "hellingers_anomaly_0.npy",
        "hellingers_anomaly_1.npy"
    ]

    for filename in files_hellingers:
        print("Processing: " + filename[:-4])
        plot_hellinger(filename=filename, title=filename[:-4])

    for filename in files_hellingers:
        print("Processing binary threshold for " + filename[:-4])
        with open(filename, 'rb') as f:
            hellinger = np.load(f)
            hellinger_binary = np.where(hellinger > threshold_nominal_1_data, 1, 0)
            plt.figure(figsize=(20, 10))
            plt.plot(hellinger_binary, linewidth=2, label=filename[:-4])
            plt.title(filename[:-4] + " BINARY THRESHOLD")
            plt.legend()
            plt.savefig(filename[:-4] + "_binary_thr.png", bbox_inches='tight', dpi=100)
            plt.close()
