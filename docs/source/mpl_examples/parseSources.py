from collections import OrderedDict

import astropy.units as u
import numpy as np

from exosim.tasks.parse import ParseSources

wl = np.linspace(0.5, 7.8, 10000) * u.um
tt = np.linspace(0.5, 1, 10) * u.hr
sources_in = OrderedDict(
    {
        "HD 209458": {
            "value": "HD 209458",
            "source_type": "planck",
            "R": 1.18 * u.R_sun,
            "D": 47 * u.pc,
            "T": 6086 * u.K,
        },
        "GJ 1214": {
            "value": "GJ 1214",
            "source_type": "planck",
            "R": 0.218 * u.R_sun,
            "D": 13 * u.pc,
            "T": 3026 * u.K,
        },
    }
)

parseSources = ParseSources()
sources_out = parseSources(parameters=sources_in, wavelength=wl, time=tt)

import matplotlib.pyplot as plt

for key in sources_out.keys():
    plt.plot(sources_out[key].spectral, sources_out[key].data[0, 0], label=key)
plt.ylabel(sources_out[key].data_units)
plt.xlabel(sources_out[key].spectral_units)
plt.legend()
plt.show()
