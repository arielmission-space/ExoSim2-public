import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

import yaml
from astropy import units as u
from astropy.table import Table

from exosim.tasks.task import Task
from exosim.utils.run_config import RunConfig


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
    >>> options = load_options(filename="path/to/file.xml")
    """

    def __init__(self):
        super().__init__()  # Explicitly call the parent class constructor
        self.add_task_param("filename", "Input option file name")
        self.add_task_param("config_path", "On-run setting for ConfigPath", None)

    def execute(self):
        self._filename = self.get_task_param("filename")
        config_path_param = self.get_task_param("config_path")
        self.config_path = config_path_param or "__ConfigPath__"
        # Track if config_path was explicitly set (not default)
        self._config_path_from_param = config_path_param is not None

        RunConfig.config_file_list.append(os.path.abspath(self._filename))

        self._check_format()
        root = self._get_root()
        options_dict = self._parse_data(root)
        self._finalise(options_dict)
        options_dict = self._substitute_keywords(options_dict)
        options_dict = _clean_dict(options_dict)
        self.set_output(options_dict)

    def _parse_data(self, root):
        """Parse XML or YAML data structure into a unified dictionary format.

        Parameters
        ----------
        root : ElementTree.Element or dict
            Root element from XML or dict from YAML

        Returns
        -------
        dict
            Parsed data in unified format
        """
        if isinstance(root, dict):
            # YAML input
            return self._parse_yaml(root)
        # XML input
        return self._parse_xml(root)

    def _parse_yaml(self, data):
        """Parse YAML dict to match the XML format.

        Parameters
        ----------
        data : dict
            YAML data structure

        Returns
        -------
        dict
            Converted data structure
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                child_dict = self._parse_yaml(value)
                if "value" in value and "unit" in value:
                    unit_name = value["unit"]
                    val = value["value"]
                    if isinstance(val, str):
                        val = self._convert_value(val)
                    if unit_name == "dimensionless":
                        unit_name = ""
                    if unit_name:
                        val = val * u.Unit(unit_name)
                    child_dict["value"] = val
                result[key] = child_dict
            else:
                result[key] = {"value": value}
                if isinstance(value, str):
                    result[key]["value"] = self._convert_value(value)
        return result

    def _check_format(self):
        if not (
            self._filename.endswith(".xml")
            or self._filename.endswith(".yaml")
            or self._filename.endswith(".yml")
        ):
            self.error("Wrong input format: Expected an XML or YAML file.")
            raise OSError("Wrong input format: Expected an XML or YAML file.")

    def _get_root(self):
        try:
            self.debug(f"Input option file found: {self._filename}")
            if self._filename.endswith(".yaml") or self._filename.endswith(".yml"):
                with open(self._filename) as f:
                    return yaml.safe_load(f)
            else:
                return ET.parse(self._filename).getroot()
        except FileNotFoundError:
            self.error(f"No input option file found: {self._filename}")
            raise
        except (yaml.YAMLError, ET.ParseError) as e:
            self.error(f"Error parsing {self._filename}: {e!s}")
            raise OSError(f"Error parsing {self._filename}: {e!s}") from e

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

            # Only update config_path from XML if it wasn't explicitly set as parameter
            if child.tag == "ConfigPath" and not self._config_path_from_param:
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
                raise OSError("Error reading the input data file.") from None

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
            if isinstance(value, dict) and "value" in value and key.startswith("__")
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
            xmlfile = xml_entry["value"].replace("__ConfigPath__", self.config_path)
            xmlfile = os.path.expanduser(xmlfile)
            if not os.path.exists(xmlfile):
                self.error(f"Referenced config file not found: {xmlfile}")
                raise FileNotFoundError(f"Referenced config file not found: {xmlfile}")
            sub_system_dict = LoadOptions()(
                filename=xmlfile, config_path=self.config_path
            )
            dictionary.update(sub_system_dict)

        for item in dictionary.values():
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
        if value == "False":
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
            if keys_list == ["unit", "value"] or keys_list == ["value"]:
                input_dict[key] = input_dict[key]["value"]
            else:
                _clean_dict(input_dict[key])

    return input_dict
