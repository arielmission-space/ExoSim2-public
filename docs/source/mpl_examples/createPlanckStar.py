import astropy.units as u
import numpy as np

from exosim.tasks.sed import CreatePlanckStar

createPlanckStar = CreatePlanckStar()
wl = np.linspace(0.5, 7.8, 10000) * u.um
T = 6086 * u.K
R = 1.18 * u.R_sun
D = 47 * u.au
sed = createPlanckStar(wavelength=wl, T=T, R=R, D=D)

import matplotlib.pyplot as plt

plt.plot(sed.spectral, sed.data[0, 0])
plt.ylabel(sed.data_units)
plt.xlabel(sed.spectral_units)
plt.show()
