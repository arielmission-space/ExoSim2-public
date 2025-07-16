import astropy.units as u
import numpy as np
import scipy

from exosim.tasks.task import Task


class ForegroundsToFocalPlane(Task):
    """
    It adds the foreground contribution to the focal plane

    Returns
    -------
     :class:`~exosim.models.signal.Signal`
        focal plane array
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        focal_plane: :class:`~exosim.models.signal.Signal`
            focal plane array (with time evolution)
        paths:  dict
            dictionary containing :class:`~exosim.models.signal.Radiance`
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("focal_plane", "focal plane")
        self.add_task_param("path", "path")

    def execute(self):
        self.info("adding foreground to focal plane")
        parameters = self.get_task_param("parameters")
        focal_plane = self.get_task_param("focal_plane")
        path = self.get_task_param("path")

        # if check focal plane units
        if focal_plane.data_units == "":
            focal_plane.data_units = u.ct / u.s

        # check if sub focal planes are needed
        if self._look_for_isolate(path):
            self.debug("found isolated contributions")
            sub_focal_plane = focal_plane.copy()
        else:
            sub_focal_plane = None
        sub_focal_planes = {}

        # populate the focal plane with path
        for rad in [k for k in path.keys() if "radiance" in k]:
            if sub_focal_plane:
                tmp_focal_plane = sub_focal_plane.copy()
                tmp_focal_plane.dataset_name = "sub_focal_planes/{}".format(
                    rad
                )

            if "slit_width" in path[rad].metadata.keys():
                # TODO need to perform the convolution on a cube to make it faster
                for t in range(path[rad].time.size):
                    npix = (
                        path[rad].metadata["slit_width"].to(u.um)
                        / parameters["detector"]["delta_pix"]
                        * focal_plane.metadata["oversampling"]
                    )

                    conv = scipy.signal.convolve(
                        path[rad].data[t, 0], np.ones(int(npix)), "same"
                    )
                    focal_plane.data[t, :] += conv
                    if sub_focal_plane:
                        tmp_focal_plane.data[t, :] += conv

            else:
                focal_plane.data[:] += np.nansum(path[rad].data[:, 0])
                if sub_focal_plane:
                    tmp_focal_plane.data[:] += np.nansum(path[rad].data[:, 0])

            if sub_focal_plane:
                sub_focal_planes[rad] = tmp_focal_plane

        self.set_output([focal_plane, sub_focal_planes])

    @staticmethod
    def _look_for_isolate(path):
        rad_list = [rad for rad in path if "radiance" in rad]
        for rad in rad_list:
            if not rad.split("_")[-1].isdigit():
                return True
        return False
