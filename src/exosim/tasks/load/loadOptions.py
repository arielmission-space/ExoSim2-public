import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

from astropy import units as u
from astropy.table import Table

from exosim.tasks.task import Task
from exosim.utils.runConfig import RunConfig


class LoadOptions(Task):
    """
    Reads the XML file with payload parameters and returns an object with attributes related to the input data.

    Attributes
    ----------
    config_path : str
        Configuration path.

    Returns
    -------
    dict
        Parsed XML input file.

    Raises
    ------
    IOError
        If the indicated file is not found or the format is not supported.

    Examples
    --------
    >>> load_options = LoadOptions()
    >>> options = load_options(filename='path/to/file.xml')
    """

    def __init__(self):
        super().__init__()  # Explicitly call the parent class constructor
        self.add_task_param("filename", "Input option file name")
        self.add_task_param(
            "config_path", "On-run setting for ConfigPath", None
        )

    def execute(self):
        self._filename = self.get_task_param("filename")
        self.config_path = (
            self.get_task_param("config_path") or "__ConfigPath__"
        )

        RunConfig.config_file_list.append(os.path.abspath(self._filename))

        self._check_format()
        root = self._get_root()
        options_dict = self._parse_xml(root)
        self._finalise(options_dict)
        options_dict = self._substitute_keywords(options_dict)
        options_dict = _clean_dict(options_dict)
        self.set_output(options_dict)

    def _check_format(self):
        if not self._filename.endswith(".xml"):
            self.error("Wrong input format: Expected an XML file.")
            raise OSError(
                "Unsupported file format. Please provide an XML file."
            )

    def _get_root(self):
        try:
            self.debug(f"Input option file found: {self._filename}")
            return ET.parse(self._filename).getroot()
        except FileNotFoundError:
            self.error(f"No input option file found: {self._filename}")
            raise FileNotFoundError(f"Input file not found: {self._filename}")

    def _parse_xml(self, root):
        root_dict = {}
        for child in root:
            child_dict = self._parse_xml(child)
            child_dict.update(child.attrib)

            value = self._compact_string(child.text)
            if value:
                value = self._convert_value(value)
                if "unit" in child_dict:
                    unit_name = child_dict.pop("unit")
                    if unit_name == "dimensionless":
                        unit_name = ""
                    value = value * u.Unit(unit_name)
                if isinstance(value, str) and "__ConfigPath__" in value:
                    value = value.replace("__ConfigPath__", self.config_path)
                child_dict["value"] = value

            if child.tag == "ConfigPath":
                self.config_path = value

            if child.tag in root_dict:
                existing_attr = root_dict[child.tag]
                if isinstance(existing_attr, OrderedDict):
                    existing_attr[value] = child_dict
                else:
                    dtmp = OrderedDict(
                        [
                            (existing_attr.get("value"), existing_attr),
                            (value, child_dict),
                        ]
                    )
                    root_dict[child.tag] = dtmp
            else:
                root_dict[child.tag] = child_dict

        if "datafile" in root_dict:
            datafile = root_dict["datafile"]["value"].replace(
                "__ConfigPath__", self.config_path
            )
            if not os.path.exists(datafile):
                self.error(f"Datafile not found: {datafile}")
                raise FileNotFoundError(f"Datafile not found: {datafile}")
            try:
                root_dict["data"] = self._read_data_table(datafile)
            except OSError:
                self.error(f"Cannot read input file: {datafile}")
                raise OSError("Error reading the input data file.")

        return root_dict

    def _substitute_keywords(self, root_dict):
        """
        Substitutes keywords in the root_dict with their corresponding values.

        Parameters
        ----------
        root_dict : dict
            The dictionary parsed from the XML.

        Returns
        -------
        dict
            Updated dictionary with substituted keywords.
        """
        # Extract keywords (elements starting with '__')
        keywords = {
            key.strip("_"): value["value"]
            for key, value in root_dict.items()
            if isinstance(value, dict)
            and "value" in value
            and key.startswith("__")
        }

        def substitute_value(value):
            if isinstance(value, str):
                for keyword, replacement in keywords.items():
                    value = value.replace(f"__{keyword}__", replacement)
            return value

        def recursive_substitute(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    recursive_substitute(value)
                elif isinstance(value, str):
                    d[key] = substitute_value(value)

        recursive_substitute(root_dict)
        return root_dict

    def _finalise(self, dictionary):
        xml_entry = dictionary.pop("config", None)
        if xml_entry:
            xmlfile = xml_entry["value"].replace(
                "__ConfigPath__", self.config_path
            )
            xmlfile = os.path.expanduser(xmlfile)
            if not os.path.exists(xmlfile):
                self.error(f"Referenced config file not found: {xmlfile}")
                raise FileNotFoundError(
                    f"Referenced config file not found: {xmlfile}"
                )
            sub_system_dict = LoadOptions()(
                filename=xmlfile, config_path=self.config_path
            )
            dictionary.update(sub_system_dict)

        for key, item in dictionary.items():
            if isinstance(item, dict):
                self._finalise(item)

    def _compact_string(self, string):
        return string.replace("\n", "").strip() if string else ""

    def _convert_value(self, value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return self._convert_boolean(value)

    def _convert_boolean(self, value):
        if value == "True":
            return True
        elif value == "False":
            return False
        return value

    def _read_data_table(self, datafile):
        data_type = os.path.splitext(datafile)[1]
        try:
            data = Table.read(
                os.path.expanduser(datafile),
                fill_values=[("#N/A", "0"), ("N/A", "0"), ("", "0")],
                format="ascii" + data_type,
            )
        except Exception as exc:
            raise Exception(f"{datafile} caused the exception") from exc

        for col in data.columns:
            if hasattr(data[col], "fill_value"):
                data[col].fill_value = 0.0

        return data


def _clean_dict(input_dict):
    """
    Cleans an input dictionary by removing the "value" notation and comments.
    It can be applied recursively.

    Parameters
    ----------
    input_dict : dict
        The dictionary to clean.

    Returns
    -------
    dict
        Cleaned dictionary.
    """
    input_dict.pop("comment", None)

    for key in list(input_dict.keys()):
        if isinstance(input_dict[key], dict):
            keys_list = list(input_dict[key].keys())
            if keys_list == ["unit", "value"]:
                input_dict[key] = input_dict[key]["value"]
            elif keys_list == ["value"]:
                input_dict[key] = input_dict[key]["value"]
            else:
                _clean_dict(input_dict[key])

    return input_dict
