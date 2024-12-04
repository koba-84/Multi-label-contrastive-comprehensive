import json
from typing import Dict


class ConfigParser:
    """ Basic parser
    :class: Class to parse the first json
    :param file_name: _description_
    :type file_name: str
     """
    def __init__(self, file_name: str):
        self.config = self.initialisation(file_name=file_name)

    def initialisation(self, file_name: str):
        with open(file_name, 'r') as f:
            config = json.load(f)
        return config

if __name__ == '__main__':
    pass
