import astropy.units as u
import numpy as np

from exosim.utils.psf import create_psf

img = create_psf(1 * u.um, (40, 40), 6 * u.um, shape="gauss")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(4, 1),
    height_ratios=(1, 4),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.05,
)
ax = fig.add_subplot(gs[1, 0])
ax.imshow(img)

ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
ax_y = fig.add_subplot(gs[1, 1], sharey=ax)


axis_x = np.arange(0, img.shape[1])
ax_x.plot(axis_x, img.sum(axis=0))
ax_x.set_xticks([], [])
axis_y = np.arange(0, img.shape[0])
ax_y.plot(img.sum(axis=1), axis_y)
ax_y.set_yticks([], [])

plt.show()
