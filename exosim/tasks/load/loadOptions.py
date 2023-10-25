import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

from astropy import units as u
from astropy.table import Table

from exosim.tasks.task import Task
from exosim.utils.runConfig import RunConfig


class LoadOptions(Task):
    """
    Reads the xml file with payload parameters and return an object with attributes related to the input data

    Attributes
    ----------
    configPath: str
        configuration path


    Returns
    -------
    dict:
        parsed xml input file

    Raises
    ------
        IOError
            if the indicated file is not found or the format is not supported

    Examples
    --------
    >>> loadOptions = LoadOptions()
    >>> options = loadOptions(filename = 'path/to/file.xml')
    """

    def __init__(self):
        """
        Parameters
        ----------
        filename: str
            input xml file location
        config_path: string (optional)
                on-run setting for ConfigPat. Default is None.

        """
        self.add_task_param("filename", "input option file name")
        self.add_task_param(
            "config_path", "on-run setting for ConfigPath", None
        )

    def _compactString(self, string):
        return string.replace("\n", "").strip()

    def execute(self):
        self._filename = self.get_task_param("filename")
        configPath = self.get_task_param("config_path")

        RunConfig.config_file_list += [os.path.abspath(self._filename)]

        if configPath:
            self.configPath = configPath
        else:
            self.configPath = "__ConfigPath__"

        self.__check_format__()
        root = self.__get_root__()
        self._dict = self.__parser__(root)
        self.__finalise__(self._dict)
        # self.debug('loaded options: {}'.format(self._dict))
        self._dict = _clean_dict(self._dict)
        self.set_output(self._dict)

    def __finalise__(self, dictionary):
        """
        Parse the dictionary tree and search for xml entries to load
        """
        xml_entry = dictionary.pop("config", None)
        if xml_entry:
            xmlfile = xml_entry["value"].replace(
                "__ConfigPath__", self.configPath
            )
            xmlfile = os.path.expanduser(xmlfile)
            lo = LoadOptions()
            sub_system_dict = lo(filename=xmlfile, config_path=self.configPath)
            for key, item in sub_system_dict.items():
                dictionary[key] = item

        for key, item in dictionary.items():
            if isinstance(item, dict):
                self.__finalise__(item)

    def __check_format__(self):
        if not self._filename.endswith(".xml"):
            self.error("wrong input format")
            raise OSError

    def __get_root__(self):
        try:
            self.debug("input option file found %s" % self._filename)
            return ET.parse(self._filename).getroot()
        except OSError:
            self.error("No input option file found")
            raise OSError

    def __parser__(self, root):
        root_dict = {}

        for ch in root:
            retval = self.__parser__(ch)

            # parse all attributes
            for key in list(ch.attrib.keys()):
                retval[key] = ch.attrib[key]

            value = self._compactString(ch.text)

            if (value is not None) and (value != ""):
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                if "unit" in retval:
                    unitName = retval["unit"]
                    if unitName == "dimensionless":
                        unitName = ""
                    value = value * u.Unit(unitName)

                if value == "True":
                    value = bool(True)

                if value == "False":
                    value = bool(False)
                if isinstance(value, str):
                    if "__ConfigPath__" in value:
                        value = value.replace(
                            "__ConfigPath__", self.configPath
                        )
                if ch.tag == "ConfigPath":
                    self.configPath = value

                # if isinstance(value, str):
                #     retval = value
                # else:
                retval["value"] = value

            if ch.tag in root_dict:
                # if an other instance of same tag exists, transform into a dict
                attr = root_dict[ch.tag]
                if isinstance(attr, OrderedDict):
                    attr[value] = retval
                else:
                    if not isinstance(attr, str):
                        dtmp = OrderedDict(
                            [(attr["value"], attr), (value, retval)]
                        )
                        root_dict[ch.tag] = dtmp
            else:
                # othewise, set new attr
                root_dict[ch.tag] = retval

        if "datafile" in root_dict:
            datafile = root_dict["datafile"]
            datafile = datafile["value"].replace(
                "__ConfigPath__", self.configPath
            )

            if not (os.path.exists(datafile)):
                self.error("Datafile not found {}".format(datafile))
                raise FileNotFoundError(
                    "Datafile not found {}".format(datafile)
                )

            try:
                data = self.__read_datatable__(datafile)
                root_dict["data"] = data
            except OSError:
                self.error("Cannot read input file {}".format(datafile))
                raise OSError

        return root_dict

    def __getopt__(self):
        return self.__obj__

    def __read_datatable__(self, datafile):
        data_type = os.path.splitext(datafile)[1]

        try:
            data = Table.read(
                os.path.expanduser(datafile),
                fill_values=[("#N/A", "0"), ("N/A", "0"), ("", "0")],
                format="ascii" + data_type,
            )
        except Exception as exc:
            raise Exception(
                "{} caused the exception".format(datafile)
            ) from exc

        for col in data.columns:
            if hasattr(data[col], "fill_value"):
                data[col].fill_value = 0.0

        return data


def _clean_dict(input_dict):
    """
    It cleans an input dictionary by removing the "value" notation and comments.
    It can be applied recursively

    Parameters
    ----------
    input_dict

    Returns
    -------
    dict:
        cleaned dictionary

    """
    # remove the comments
    if "comment" in input_dict:
        input_dict.pop("comment")

    # if only values are in the dict key, they are converted
    list(input_dict.keys())
    for k in list(input_dict.keys()):
        if isinstance(input_dict[k], dict):
            k_list = list(input_dict[k].keys())
            if k_list == ["unit", "value"]:
                input_dict[k] = input_dict[k][
                    "value"
                ]  # * input_dict[k]['unit']
            elif k_list == ["value"]:
                input_dict[k] = input_dict[k]["value"]
            else:
                _clean_dict(input_dict[k])

    return input_dict
