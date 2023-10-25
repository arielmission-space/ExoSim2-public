import astropy.units as u
import numpy as np

from exosim.tasks.parse import ParseSource

parseSource = ParseSource()
wl = np.linspace(0.5, 7.8, 10000) * u.um
tt = np.linspace(0.5, 1, 10) * u.hr

source_in = {
    "value": "HD 209458",
    "source_type": "planck",
    "R": 1.18 * u.R_sun,
    "D": 47 * u.pc,
    "T": 6086 * u.K,
}
source_out = parseSource(parameters=source_in, wavelength=wl, time=tt)

import matplotlib.pyplot as plt

plt.plot(source_out["HD 209458"].spectral, source_out["HD 209458"].data[0, 0])
plt.ylabel(source_out["HD 209458"].data_units)
plt.xlabel(source_out["HD 209458"].spectral_units)
plt.show()
