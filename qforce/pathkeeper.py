"""keep track of basic pathways in qforce"""
from collections import UserDict
import os
from pathlib import Path
from string import Template


class Pathlib(UserDict):

    """UserDict to easy access local paths in qforce

    the given dict should have the form:

        'name': ['parent', 'name']

        or

        'name': 'name'

    where parent is the parent diretory of the path
    """

    def __getitem__(self, key):
        value = self.data[key]
        if isinstance(value, str):
            return value
        if value[0] == '':
            return value[1]
        return os.path.join(self.data[value[0]], value[1])


class Pathways:
    """Basic class to keep track of all important file paths in qforce"""

    pathways = Pathlib({
        # folders
        'preopt': '0_preopt',
        'hessian': '1_hessian',
        'hessian_charge': ['hessian', 'charge'],
        'fragments': '2_fragments',
        'fragment': ['fragments', '${frag_id}'],
        'fragment_scans': ['fragments', '${frag_id}_scans'],
        # files
        'init.xyz': 'init.xyz',
        'preopt.xyz': ['preopt', 'preopt.xyz'],
        'calculations.json': '_calculations.json',
        'settings.ini': 'settings.ini',
    })

    def __init__(self, jobdir, name=None):
        if name is None:
            name = jobdir
        self.jobdir = Path(jobdir)
        self.name = name

    def initxyz(self, *, only=False):
        return self._path(self.pathways['init.xyz'], only)

    def preoptxyz(self, *, only=False):
        return self._path(self.pathways['preopt.xyz'], only)

    def calculationsjson(self, *, only=False):
        return self._path(self.pathways['calculations.json'], only)

    def settingsini(self, *, only=False):
        return self._path(self.pathways['settings.ini'], only)

    def preopt_dir(self, *, only=False, create=False):
        return self._dirpath(self.pathways['preopt'], only, create)

    def hessian_dir(self, *, only=False, create=False):
        return self._dirpath(self.pathways['hessian'], only, create)

    def hessian_charge_dir(self, *, only=False, create=False):
        return self._dirpath(self.pathways['hessian_charge'], only, create)

    def fragments_dir(self, *, only=False, create=False):
        return self._dirpath(self.pathways['fragments'], only, create)

    def frag_dir(self, id, *, only=False, create=False):
        return self._dirpath(self.pathways['fragments'], only, create)

    def __getitem__(self, key):
        options = {
                'preopt': self.preopt_dir,
                'hessian': self.hessian_dir,
                'hessian_charge': self.hessian_charge_dir,
                'fragments': self.fragments_dir,
                # files
                'init.xyz': self.initxyz,
                'preopt.xyz': self.preoptxyz,
                'calculations.json': self.calculationsjson,
                'settings.ini': self.settingsini,
                }
        value = options.get(key, None)
        if value is None:
            raise ValueError(f"Option can only be one of '{','.join(options)}'")
        return value()

    def _path(self, path, only):
        """Path to the folder"""
        if only is True:
            return Path(path)
        return self.jobdir / path

    def _dirpath(self, path, only, create):
        path = self._path(path, only)
        if create is True:
            os.makedirs(path, exist_ok=True)
        return path

    def basename(self, software, charge, mult):
        return f'{self.name}_{software.hash(charge, mult)}'

    def hessian_name(self, software, charge, mult):
        return f'{self.basename(software, charge, mult)}_hessian'

    def hessian_charge_name(self, software, charge, mult):
        return f'{self.basename(software, charge, mult)}_hessian_charge'

    def charge_name(self, software, charge, mult):
        return f'{self.basename(software, charge, mult)}_charge'

    def scan_sp_name(self, software, charge, mult, i):
        return f'{self.basename(software, charge, mult)}_sp_step_{i:02d}'
