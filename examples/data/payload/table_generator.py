import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import QTable

wl = np.arange(0.65, 4.5, 0.05) * u.um

D1, M1, QE = QTable(), QTable(), QTable()

M1["Wavelength"] = wl
M1["Reflectivity"] = np.ones(wl.size) * 0.9 * u.Unit("")
M1["Emissivity"] = np.ones(wl.size) * 0.03 * u.Unit("")
ascii.write(M1, "M1.ecsv", format="ecsv", delimiter=",", overwrite=True)


D1["Wavelength"] = wl
D1["Reflectivity"] = np.ones(wl.size) * 0.9 * u.Unit("")
D1["Reflectivity"][wl < 0.7 * u.um] = 0 * u.Unit("")
D1["Reflectivity"][wl > 1.05 * u.um] = 0 * u.Unit("")

D1["Transmission"] = np.ones(wl.size) * 0.9 * u.Unit("")
D1["Transmission"][wl < 1 * u.um] = 0 * u.Unit("")
D1["Transmission"][wl > 3.55 * u.um] = 0 * u.Unit("")
D1["Emissivity"] = np.ones(wl.size) * 0.03 * u.Unit("")
ascii.write(D1, "D1.ecsv", format="ecsv", delimiter=",", overwrite=True)


QE["Wavelength"] = wl
QE["Photometer"] = np.ones(wl.size) * 0.7 * u.Unit("")
QE["Spectrometer"] = np.ones(wl.size) * 0.8 * u.Unit("")

ascii.write(QE, "QE.ecsv", format="ecsv", delimiter=",", overwrite=True)


R = 40.0
Fnum_x = 20
delta_pix = 18 * u.um

# Calculate ideal wavelength dispersion
x = np.linspace(-80.0, 80.0, 100) * delta_pix
y = np.zeros_like(x)

# Wavelength Dispersion
wl_center = 0.5 * (1 + 3.5) * u.um
DwlDx = 1.0 / (1.22 * Fnum_x * R)
wl = DwlDx * x
wl -= wl[0] - 0.95 * u.um
data = QTable([wl, y, x], names=["Wavelength", "y", "x"])

fname = "spec-wl_sol.ecsv"
ascii.write(data, fname, format="ecsv", delimiter=",", overwrite=True)
