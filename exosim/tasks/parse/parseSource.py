from collections import OrderedDict

import astropy.units as u
import numpy as np
import requests
from astropy import constants as cc

import exosim.output as output
from exosim.tasks.sed import CreateCustomSource
from exosim.tasks.sed import PrepareSed
from exosim.tasks.task import Task
from exosim.utils.klass_factory import find_task


class ParseSources(Task):
    """
    Given the source descrition, it parses the sources elements and return a dictionary.
    It also applyes the time variation if provided.

    Returns
    -------
    dict
        dictionary containing :class:`~exosim.models.signal.Sed`

    Examples
    ---------

    >>> import astropy.units as u
    >>> import numpy as np
    >>> from exosim.tasks.parse import ParseSources
    >>> from collections import OrderedDict
    >>>
    >>> wl = np.linspace(0.5, 7.8, 10000) * u.um
    >>> tt = np.linspace(0.5, 1, 10) * u.hr
    >>>
    >>> sources_in = OrderedDict({'HD 209458': {'value': 'HD 209458',
    >>>                                         'source_type': 'planck',
    >>>                                         'R': 1.18 * u.R_sun,
    >>>                                         'D': 47 * u.pc,
    >>>                                         'T': 6086 * u.K,
    >>>                                         },
    >>>                           'GJ 1214': {'value': 'GJ 1214',
    >>>                                       'source_type': 'planck',
    >>>                                       'R': 0.218 * u.R_sun,
    >>>                                       'D': 13 * u.pc,
    >>>                                       'T': 3026 * u.K,
    >>>                                       }, })
    >>>
    >>> parseSources = ParseSources()
    >>> sources_out = parseSources(parameters=sources_in,
    >>>                            wavelength=wl,
    >>>                            time=tt)

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.plot(source_out['HD 209458'].spectral, source_out['HD 209458'].data[0,0])
    >>> plt.ylabel(source_out['HD 209458'].data_units)
    >>> plt.xlabel(source_out['HD 209458'].spectral_units)
    >>> plt.show()

    .. plot:: mpl_examples/parseSources.py

    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.add_task_param("parameters", "sources parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)

    def execute(self):
        parameters = self.get_task_param("parameters")
        self.info("parsing sources")

        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        output_file = self.get_task_param("output")

        parseSource = ParseSource()
        out = {}
        # if a list of sources are provided we parse all of them and we sum them up
        if isinstance(parameters, OrderedDict):
            for source_name in parameters.keys():
                source_ = parseSource(
                    parameters=parameters[source_name],
                    wavelength=wl,
                    time=tt,
                    output=output_file,
                )

                out = {**out, **source_}
        # else, we load the only source available
        else:
            out = parseSource(
                parameters=parameters,
                wavelength=wl,
                time=tt,
                output=output_file,
            )

        self.set_output(out)


class ParseSource(Task):
    """
    Given the source parameters, it parses the source element and returns a dictionary.
    It also applyes the time variation if provided.

    Returns
    -------
    dict
        dictionary containing :class:`~exosim.models.signal.Sed`

    Examples
    ---------

    >>> from exosim.tasks.parse import ParseSource
    >>> import astropy.units as u
    >>> import numpy as np
    >>> parseSource = ParseSource()
    >>> wl = np.linspace(0.5, 7.8, 10000) * u.um
    >>> tt = np.linspace(0.5, 1, 10) * u.hr
    >>> source_in = {
    >>>     'value': 'HD 209458',
    >>>     'source_type': 'planck',
    >>>     'R': 1.18 * u.R_sun,
    >>>     'D': 47 * u.pc,
    >>>     'T': 6086 * u.K,
    >>>     }
    >>> source_out = parseSource(parameters=source_in,
    >>>                          wavelength=wl,
    >>>                          time=tt)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(source_out['HD 209458'].spectral, source_out['HD 209458'].data[0,0])
    >>> plt.ylabel(source_out['HD 209458'].data_units)
    >>> plt.xlabel(source_out['HD 209458'].spectral_units)
    >>> plt.show()

    .. plot:: mpl_examples/parseSource.py

    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.add_task_param("parameters", "sources parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)

    def execute(self):
        parameters = self.get_task_param("parameters")
        self.info("parsing source: {}".format(parameters["value"]))

        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        sed = self._parse_sed(parameters, wl)

        sed.metadata["name"] = parameters["value"]
        sed.metadata["parsed_parameters"] = parameters

        sed.spectral_rebin(wl)
        sed.temporal_rebin(tt)

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                og = output_file.create_group("sources")
                sed.write(og, sed.metadata["name"])

        out = {parameters["value"]: sed}
        self.set_output(out)

    def _parse_sed(self, parameters, wl):
        kwards = {"wavelength": wl}

        if "source_task" in parameters.keys():
            # if custom source is created by the user skip the automatic part
            source_task = find_task(
                parameters["source_task"], CreateCustomSource
            )
            task_instance = source_task()
            sed = task_instance(parameters=parameters)
            sed.metadata = {**sed.metadata, **kwards}
            return sed

        # initialize the keyword dictionary to the value into the xml or to None
        k_list = [
            "source_type",
            "R",
            "D",
            "logg",
            "T",
            "z",
            "path",
            "filename",
        ]
        for k in k_list:
            kwards[k] = None
            if k in list(parameters.keys()):
                kwards[k] = parameters[k]
                self.debug(
                    "source {} found in parameters: {}".format(
                        k, parameters[k]
                    )
                )

        # estimate the log from the mass if is not in the parameters but M and R are
        if "logg" not in list(parameters.keys()):
            try:
                g = (cc.G * parameters["M"].si / parameters["R"].si ** 2).to(
                    u.cm / u.s**2
                )
                kwards["logg"] = np.log10(g.value)
                self.debug("logg estimated: {}".format(kwards["logg"]))
            except KeyError:
                self.warning(
                    "Both mass (M) and radius (R) must be indicated to estimate logg"
                )

        # if an online data base is indicated it grab the data
        if "online_database" in list(parameters.keys()):
            # if parameters[
            #     'online_database'] == 'https://exodb.space/api/v1/star':
            response = requests.post(
                url=parameters["online_database"]["url"],
                headers=parameters["online_database"],
                json={"star_name": parameters["value"]},
            )
            star = response.json()["data"]["Properties"]
            kwards["R"] = star["Radius"]["value"] * u.Unit(
                star["Radius"]["unit"]
            )
            kwards["T"] = star["Effective Temperature"]["value"] * u.Unit(
                star["Effective Temperature"]["unit"]
            )
            kwards["D"] = star["Distance from Earth"]["value"] * u.Unit(
                star["Distance from Earth"]["unit"]
            )
            kwards["z"] = star["Metallicity"]["value"]
            M = star["Mass"]["value"] * u.Unit(star["Mass"]["unit"])
            g = (cc.G * M.si / kwards["R"].si ** 2).to(u.cm / u.s**2)
            kwards["logg"] = np.log10(g.value)
            self.debug("source data loaded from ExoDB")

        prepareSed = PrepareSed()
        sed = prepareSed(**kwards)
        sed.metadata = {**sed.metadata, **kwards}
        return sed
