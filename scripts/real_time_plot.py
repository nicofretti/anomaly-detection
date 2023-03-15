# Plot the data in real time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg # PLOT THE ICE LAB BACKGROUND 
import matplotlib.lines as mlines

# CSV library
import pandas as pd
import numpy as np

def update_plots(i):
    
    points_copy = pd.read_csv(map_filename).to_numpy()
    h2_variables = pd.read_csv(h2_filename).to_numpy()

    map.cla()
    decomposition.cla()
    
    # plot map
    map.set_title('ICE Lab Map')
    map.plot(X_nominal_0, Y_nominal_0, zorder=1, c = 'orange')
    map.plot(X_nominal_1, Y_nominal_1, zorder=2, c = 'c')
    # map.plot(X_sample[:, 0], X_sample[:, 1], ".-", label="observations", ms=6, mfc="orange", alpha=0.7)
    # Indicate the component numbers
    # means = model.means_
    # for i, m in enumerate(means):
    #     map.text(m[0], m[1], 's%i' % (i + 1), size=10, horizontalalignment='center', bbox=dict(boxstyle='circle', alpha=.5, facecolor='w'))

    img_backgroung = mpimg.imread('src/anomaly_detection/data/images/ICE_lab.png')
    map.imshow(img_backgroung, extent=(-1.5, 2.5, -13, 1), cmap='gray')
    
    map.set_xlim([-1.5, 2.5])
    map.set_ylim([-5, 1])
    
    # points contains X, Y, anomaly
    n_points = np.shape(points_copy)[0]
    if n_points > 0:
        x0 = points_copy[0, 0]
        y0 = points_copy[0, 1]
        X_plot = points_copy[1:, 0]
        Y_plot = points_copy[1:, 1]
        anomaly = points_copy[1:, 2]
        orange_patch = mpatches.Patch(color='orange', label='nominal_0')
        cyan_patch = mpatches.Patch(color='cyan', label='nominal_1')
        green_patch = mpatches.Patch(color='green', label='start')
        red_patch = mpatches.Patch(color='red', label='anomaly')
        blue_patch = mpatches.Patch(color='blue', label='normal')
        color = np.where(anomaly, 'r', 'b')
        map.scatter(x0, y0, c='g', s=10, zorder=4)
        map.scatter(X_plot, Y_plot, c=color, s=7, zorder=3)
        map.legend(handles=[orange_patch, cyan_patch, green_patch, blue_patch, red_patch], loc=1)

    decomposition.set_title('Hellinger Distance Decomposition')
    x_rows = np.shape(h2_variables)[0]
    if x_rows!=0:
        X_dec = h2_variables[:, 0]
        Y_dec = h2_variables[:, 1]
        O_dec = h2_variables[:, 2]
        LS_dec = h2_variables[:, 3]
        LC_dec = h2_variables[:, 4]
        LD_dec = h2_variables[:, 5]
        decomposition.plot(X_dec, 'k', linestyle = '-', label = 'X')
        decomposition.plot(Y_dec, 'g', linestyle = '-', label = 'Y')
        decomposition.plot(O_dec, 'y', linestyle = '-', label = 'O')
        decomposition.plot(LS_dec, 'c', linestyle = '-', label = 'LS')
        decomposition.plot(LC_dec, 'm', linestyle = '-', label = 'LC')
        decomposition.plot(LD_dec, 'b', linestyle = '-', label = 'LD')

        # PLOT A RED LINE IF THERE IS AN ANOMALY ON THE CURRENT SENSORS
        anomalies = np.where(h2_variables > h2_thr, h2_variables, np.nan)
        X_an = anomalies[:, 0]
        Y_an = anomalies[:, 1]
        O_an = anomalies[:, 2]
        LS_an = anomalies[:, 3]
        LC_an = anomalies[:, 4]
        LD_an = anomalies[:, 5]
        decomposition.plot(X_an, 'kx')
        decomposition.plot(Y_an, 'gx')
        decomposition.plot(O_an, 'yx')
        decomposition.plot(LS_an, 'cx')
        decomposition.plot(LC_an, 'mx')
        decomposition.plot(LD_an, 'bx')
        labels = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
        colors = np.where(h2_variables[-1] > h2_thr, 'red', 'green')
        lights = []
        for i in range(0, len(labels)):
            light = mlines.Line2D([], [], color=colors[i], marker='.', linestyle='None', markersize=20, label=labels[i])
            lights.append(light)
        semaphore_legend = decomposition.legend(handles=lights, loc='upper left')
        decomposition.legend(loc='upper right')
        decomposition.add_artist(semaphore_legend)

if __name__ == "__main__":

    map_filename = "map_data.csv"
    h2_filename = "h2_decomposition_data.csv"
    h2_thr = [1.26020238, 6.67861522, 0.4251171,  0.70920265, 0.94272347, 0.89692743]
    
    # PLOTTING REAL TIME SENSOR DATA
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0

    
    nominal_0_csv = pd.read_csv('./src/anomaly_detection/data/csv/preprocess_data_ros/nominal_0.csv')
    nominal_1_csv = pd.read_csv('./src/anomaly_detection/data/csv/preprocess_data_ros/nominal_1.csv')

    nominal_0_csv = nominal_0_csv.to_numpy()
    nominal_1_csv = nominal_1_csv.to_numpy()
    
    fig = plt.figure(num=1, facecolor='#DEDEDE')
    map = plt.subplot(121)
    decomposition = plt.subplot(122)
    map.set_facecolor('#DEDEDE')
    decomposition.set_facecolor('#DEDEDE')
    #animate
    ani = FuncAnimation(fig, update_plots, interval = 5)
    X_nominal_0 = nominal_0_csv[:, 0]
    Y_nominal_0 = nominal_0_csv[:, 1]
    X_nominal_1 = nominal_1_csv[:, 0]
    Y_nominal_1 = nominal_1_csv[:, 1]

    plt.show(block = True)