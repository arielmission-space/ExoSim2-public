import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

tab = ascii.read("dp_map_Photometer.csv")
print(tab)

focal = np.zeros((32, 32))

for (y, x) in tab:
    focal[y, x] = 1.0

plt.imshow(focal)
plt.savefig("dp_map.png")
