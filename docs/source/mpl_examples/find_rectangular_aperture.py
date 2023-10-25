import astropy.units as u
import matplotlib.pyplot as plt
import photutils

from exosim.utils.aperture import find_rectangular_aperture
from exosim.utils.psf import create_psf

img = create_psf(1 * u.um, (60, 40), 6 * u.um)
size, area, ene = find_rectangular_aperture(img, 0.85)
positions = [(img.shape[1] // 2, img.shape[0] // 2)]
aperture = photutils.aperture.RectangularAperture(positions, size[0], size[1])

plt.imshow(img)
aperture.plot(
    color="r",
    lw=2,
)
plt.show()
