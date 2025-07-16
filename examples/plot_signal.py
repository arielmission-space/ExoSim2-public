import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

input = "./test_freq.h5"


def main():
    with h5py.File(input, "r") as f:
        ch_list = list(f.keys())
        ch_list.sort()
        ch_list.remove("time")
        widths = []
        for ch in ch_list:
            size_x = f[ch].shape[1]
            size_y = f[ch].shape[0]
            widths += [int(np.ceil(size_x / size_y)), 0.1]

        heights = [1]
        scale = np.ceil(len(widths) / len(heights))
        size_y_fig = 10
        size_x_fig = size_y_fig * scale
        fig = plt.figure(
            constrained_layout=True, dpi=100, figsize=(size_x_fig, size_y_fig)
        )

        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            wspace=0.1,
            hspace=0.1,
        )
        for time in f["time"][()]:
            i = 0
            t = np.argmin(abs(time - f["time"][()]))
            for ch in ch_list:
                ax = fig.add_subplot(spec[0, i])
                ax.set_title(ch)
                im = ax.imshow(f[ch][t], interpolation="none")
                i += 1
                ax = fig.add_subplot(spec[0, i])
                plt.colorbar(im, ax=ax, cax=ax)
                i += 1

            plt.savefig("plots/iter/{}.png".format(t))


if __name__ == "__main__":
    main()
