import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x = np.linspace(-4,4,1000)
N = 100
z1 = np.random.randint(1, 3, N) * np.random.uniform(0, .4, N)
z2 = np.random.uniform(0, 1, N)

R_sq = -2 * np.log(z1)
theta = 2 * np.pi * z2
z1 = np.sqrt(R_sq) * np.cos(theta)
z2 = np.sqrt(R_sq) * np.sin(theta)

fig = plt.figure(figsize=(12,4))
for ind_subplot, zi, col in zip((1, 2), (z1, z2), ('lightgreen', 'orange')):
    ax = fig.add_subplot(1, 2, ind_subplot)
    ax.hist(zi, bins=40, range=(-4, 4), color=col, label='nominal' if ind_subplot == 1 else 'observed')
    ax.set_xlabel("Value of the variable")
    ax.set_ylabel("Frequency")

    binwidth = 8 / 40
    scale_factor = len(zi) * binwidth

    gaussian_kde_zi = stats.gaussian_kde(z1)
    # ax.plot(x, gaussian_kde_zi(x)*scale_factor, color='springgreen', linewidth=3, label='kde')

    std_zi = np.std(zi)
    mean_zi = np.mean(zi)
    ax.plot(x, stats.norm.pdf((x-mean_zi)/std_zi)*scale_factor, color='blue', linewidth=2, label='normal')
    ax.legend()

plt.show()