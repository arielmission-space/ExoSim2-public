import astropy.units as u

from exosim.utils.psf import create_psf

img = create_psf(1 * u.um, (60, 40), 6 * u.um, shape="gauss")
import matplotlib.pyplot as plt

plt.imshow(
    img,
    aspect="equal",
)
plt.show()
