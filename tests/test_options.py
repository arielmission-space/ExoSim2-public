import logging
import os
import unittest

from inputs import example_dir
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.tasks.load.loadOptions import LoadOptions

setLogLevel(logging.DEBUG)


def payload_file(source, destination):
    payload_file = os.path.join(example_dir, "main_example.xml")
    new_configPath = "    <ConfigPath> {}\n".format(source)
    tmp = os.path.join(destination, "payload_test.xml")
    try:
        os.remove(tmp)
    except OSError:
        pass
    with open(tmp, "w+") as new_file:
        with open(payload_file) as old_file:
            for line in old_file:
                if "<ConfigPath>" in line:
                    new_file.write(new_configPath)
                else:
                    new_file.write(line)
    return tmp


class LoadOptionsTest(unittest.TestCase):
    loadOption = LoadOptions()

    def test_loadFile(self):
        self.loadOption(
            filename=payload_file(source=example_dir, destination=test_dir)
        )
