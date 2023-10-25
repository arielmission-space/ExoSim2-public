import numpy as np

from exosim.utils.binning import rebin

xp = np.linspace(1, 10, 50)
fp = np.sin(xp)

x_bin = np.linspace(1, 10, 10)
f_bin = rebin(x_bin, xp, fp)

x_inter = np.linspace(1, 10, 100)
f_inter = rebin(x_inter, xp, fp)

import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(xp, fp, label="original array", alpha=0.5, c="r")
ax0.scatter(
    x_inter, f_inter, marker="X", label="interpolated array", alpha=0.5
)
ax0.scatter(x_bin, f_bin, marker="v", label="binned array", alpha=0.5)
ax0.legend()

ax1.axhline(0, c="r", alpha=0.5)
ax1.scatter(
    x_inter,
    (f_inter - np.sin(x_inter)) / np.sin(x_inter),
    marker="X",
    label="interpolated array",
    alpha=0.5,
)
ax1.scatter(
    x_bin,
    (f_bin - np.sin(x_bin)) / np.sin(x_bin),
    marker="v",
    label="binned array",
    alpha=0.5,
)
ax1.legend()
plt.show()
