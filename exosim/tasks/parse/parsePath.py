import re
from collections import OrderedDict

import astropy.units as u
import numpy as np

import exosim.models.signal as signal
import exosim.output as output
from exosim.tasks.parse import ParseOpticalElement
from exosim.tasks.task import Task


class ParsePath(Task):
    """
    Given the optical path descrition, it parses the optical elements and return an ordered dictionary.


    Returns
    -------
    dict
        dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
        represeting the radiance and efficiency of the path.

    Note
    ------
    The user can force the parser to isolate contribution by addind to the description dictionary the key 'isolate' set to ``True``.

    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            wavelength grid. If no units are attached is considered as expressed in `um`.
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        group_name: str (optional)
            group name in output
        light_path: `~collections.OrderedDict` (optional)
            dictionary of contributes
        """
        self.add_task_param("parameters", "optical element parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)
        self.add_task_param(
            "group_name", "group name in output", "contributes"
        )
        self.add_task_param("light_path", "previous light path", None)

        self.wl, self.tt, self.out, self.radiance_key = None, None, None, None

    def execute(self):
        parameters = self.get_task_param("parameters")
        self.info("parsing optical path")

        self.wl = self.get_task_param("wavelength")
        self.tt = self.get_task_param("time")

        # prepare the output radiance key
        self.radiance_key = None
        # if a previous light path is provided
        self.out = {}
        light_path = self.get_task_param("light_path")
        if light_path:
            for key in light_path.keys():
                self.out[key] = light_path[key]
            self.radiance_key = self._find_last_radiance_key(light_path.keys())
        else:
            efficiency = signal.Dimensionless(
                spectral=self.wl,
                data=np.ones((self.tt.size, 1, self.wl.size)),
                time=self.tt,
            )
            # prepare output dict
            self.out["efficiency"] = efficiency

        output_file = self.get_task_param("output")
        group_name = self.get_task_param("group_name")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                output_file = output_file.create_group(group_name)

        # if a list of optical elements are provided we parse all of them and we sum them up
        if isinstance(parameters["opticalElement"], OrderedDict):
            for element in parameters["opticalElement"].keys():
                self._add_new_element(
                    parameters["opticalElement"][element], output_file
                )

        # else, we load the only source available
        else:
            self._add_new_element(parameters["opticalElement"], output_file)

        # output preparation
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                if "final contributes" in output_file._entry.keys():
                    del output_file._entry["final contributes"]
                og = output_file.create_group("final contributes")
                for key in self.out:
                    self.out[key].write(og, key)
        self.set_output(self.out)

    def _add_new_element(self, parameters, output_file):
        """
        This method select the appropriate parser for the optical element.
        It updates the output dictionary.

        Parameters
        ----------
        parameters: dict
            parameters dict
        output_file: :class:`~exosim.output.output.Output`
            output file
        """
        # if slit, it splits the output
        if (
            "type" in parameters.keys()
            and parameters["type"].lower() == "slit"
        ):
            self._parse_slit(parameters)
        # if solid angle
        elif "solid_angle" in parameters.keys():
            self._parse_angle(parameters, output_file)
        # if isolate, store it in its own way
        elif "isolate" in parameters.keys() and parameters["isolate"]:
            # parse the optical element to get radiance and efficiency
            parsed_out = self._parse_optical_element(parameters, output_file)
            # updates efficiency
            self._update_efficiency(parsed_out)
            # check if isolate
            # updates the radiance key
            self._update_radiance_key(parameters["value"])
            # start a new radiance output
            self.out[self.radiance_key] = parsed_out["radiance"]
        # otherwise it updates it
        else:
            # if it is the first element, initialise the output dict
            if self.radiance_key is None:
                self._parse_new_radiance()
            if not self.radiance_key.split("_")[-1].isdigit():
                self._parse_new_radiance()
            if "slit_width" in self.out[self.radiance_key].metadata.keys():
                self._parse_new_radiance()
            # parse the optical element to get radiance and efficiency
            parsed_out = self._parse_optical_element(parameters, output_file)
            # updates efficiency
            self._update_efficiency(parsed_out)
            # increase current radiance
            self.out[self.radiance_key] += parsed_out["radiance"]

    def _parse_slit(self, parameters):
        """
        This method parses the slit. If a slit is found the output dict is split into two different radiances.
        It updates the output dictionary.

        Parameters
        ----------
        parameters: dict
            parameters dict
        """

        self.debug("parsing: {}".format(parameters["value"]))
        # it appends the slith width to the metadata of all the previous radiances
        for rad in [x for x in self.out.keys() if "radiance" in x]:
            self.out[rad].metadata["slit_width"] = parameters["width"]

    def _parse_angle(self, parameters, output_file):
        """
        This method parses the solid angle. If a solid angle is found the output dict is split into two different radiances.
        It updates the output dictionary.

        Parameters
        ----------
        parameters: dict
            parameters dict
        """

        self.debug("solid angle found")

        # start a new radiance output
        parseOpticalElement = ParseOpticalElement()
        parsed_out = parseOpticalElement(
            parameters=parameters,
            wavelength=self.wl,
            time=self.tt,
            output=output_file,
        )

        # updates efficiency
        self._update_efficiency(parsed_out)
        # updates the radiance key
        if "isolate" in parameters.keys() and parameters["isolate"]:
            self._update_radiance_key(parameters["value"])
        else:
            self._update_radiance_key()

        # adds the new radiance to the data
        self.out[self.radiance_key] = parsed_out["radiance"]
        # it appen the solid angle to the metadata of the previous radiance
        self.out[self.radiance_key].metadata["solid_angle"] = parameters[
            "solid_angle"
        ]

    def _parse_optical_element(self, parameters, output_file):
        if parameters["value"].lower() in ["zodiacal", "zodi"]:
            import sys  # avoid circular import

            if "ParseZodi" not in sys.modules:
                from exosim.tasks.parse.parseZodi import ParseZodi

            parseZodi = ParseZodi()
            parsed_out = parseZodi(
                parameters=parameters,
                wavelength=self.wl,
                time=self.tt,
                output=output_file,
            )
        else:
            parseOpticalElement = ParseOpticalElement()
            parsed_out = parseOpticalElement(
                parameters=parameters,
                wavelength=self.wl,
                time=self.tt,
                output=output_file,
            )
        return parsed_out

    @staticmethod
    def _find_last_radiance_key(keys_list):
        """Finds the last radiance key in the output dictionary"""
        rad_list = [rad for rad in keys_list if "radiance" in rad]
        rad_list.sort()
        return rad_list[-1]

    @property
    def radiance_keys_list(self):
        dat_re = re.compile(r"\d+")
        rad_list = [
            int(dat_re.search(rad).group())
            for rad in self.out.keys()
            if dat_re.search(rad)
        ]
        return rad_list

    def _update_radiance_key(self, name=None):
        """
        This method updates the output dictionary key for the radiance.
        """

        if self.radiance_keys_list:
            self.radiance_keys_list.sort()
            i = int(self.radiance_keys_list[-1]) + 1
        else:
            i = 0
        if name:
            self.radiance_key = "radiance_{}_{}".format(i, name)
        else:
            self.radiance_key = "radiance_{}".format(i)

        self.debug(
            "output radiance key updated to {}".format(self.radiance_key)
        )

    def _update_efficiency(self, parsed_out):
        """
        It updates the path dictionary by multiplying each efficiency and radiance by the new efficiency

        Parameters
        ----------
        parsed_out: dict
            dictionary of :class:`~exosim.models.signal.Radiance` and :class:`~exosim.models.signal.Dimensionless`,
            represeting the radiance and efficiency of the optical element.


        """
        for k in self.out.keys():
            self.out[k] *= parsed_out["efficiency"].data

    def _parse_new_radiance(self):
        self._update_radiance_key()
        radiance = signal.Radiance(
            spectral=self.wl,
            data=np.zeros((self.tt.size, 1, self.wl.size))
            * u.W
            / u.m**2
            / u.um
            / u.sr,
            time=self.tt,
        )
        # prepare output dict
        self.out[self.radiance_key] = radiance
