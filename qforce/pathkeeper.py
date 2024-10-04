"""keep track of basic pathways in qforce"""
import os
import shutil
from pathlib import Path
from string import Template


def _resort(orders, final_orders):
    missing = {}
    for name, order in orders.items():
        if isinstance(order, int):
            final_orders[name] = order
        else:
            if order in final_orders:
                final_orders[name] = final_orders[order]+1
            else:
                missing[name] = order
    return missing


def _compute_order(data):
    orders = {}
    for key, value in data.items():
        if isinstance(value, str) or value[0] == '':
            orders[key] = 0
        else:
            orders[key] = value[0]

    final_orders = {}
    while True:
        orders = _resort(orders, final_orders)
        if len(orders) == 0:
            break
    return [name for name, _ in sorted(final_orders.items(), key=lambda x: x[1])]


def get_template_arguments(string):
    allvalues = [value[1] or value[2] for value in Template.pattern.findall(string)]
    out = []
    for value in allvalues:
        if value not in out:
            out.append(value)
    return out


class Pathlib:
    """Easy access local paths in qforce

    the given dict should have the form:

        'name': ['parent', 'name']

        or

        'name': 'name'

    where parent is the parent diretory of the path, formates everything with string Templates

    so placeholder can be put according to:

        'name': ['parent', '${name}_number']

        or

        'name': '${name}_number'

    do not use placeholders in the parent!
    """

    def __init__(self, data):
        self._data = self._format(data)

    def __getitem__(self, key):
        return self._data[key]

    def files(self):
        return [name for name in self._data
                if os.path.splitext(name)[1] != '']

    def dirs(self):
        return [name for name in self._data
                if os.path.splitext(name)[1] == '']

    @staticmethod
    def _format(data):
        keys = _compute_order(data)
        result = {}
        for key in keys:
            value = data[key]
            if isinstance(value, str):
                result[key] = value
            else:
                if value[0] == '':
                    result[key] = value[1]
                else:
                    result[key] = os.path.join(result[value[0]], value[1])

        return {name: (Template(value), get_template_arguments(value))
                for name, value in result.items()}

    def __setitem__(self, key, value):
        raise ValueError("Cannot set item for Pathways")


class Pathways:
    """Basic class to keep track of all important file paths in qforce"""

    pathways = Pathlib({
        # folders
        'preopt': '0_preopt',
        'hessian': '1_hessian',
        'addstruct': '5_additional',
        'hessian_new': 'hessian_new',
        'hessian_charge': ['hessian', 'charge'],
        'hessian_energy': ['hessian', '${idx}_en_conformer'],
        'hessian_gradient': ['hessian', '${idx}_grad_conformer'],
        'hessian_step': ['hessian', '${idx}_conformer'],
        'fragments': '2_fragments',
        'frag': ['fragments', '${frag_id}'],
        'frag_charge': ['frag', 'charge'],
        'frag_step': ['frag', 'step_${idx}'],
        'frag_mm': ['frag', 'mm'],
        # files should have ending!
        'settings.ini': 'settings.ini',
        'init.xyz': 'init.xyz',
        'preopt.xyz': ['preopt', 'preopt.xyz'],
        'calculations.json': '_calculations.json',
    })

    _files = pathways.files()
    _dirs = pathways.dirs()

    def __init__(self, jobdir, name=None):
        if name is None:
            name = jobdir
        self.jobdir = Path(jobdir)
        self.name = name

    def __getitem__(self, args):
        if isinstance(args, str):
            return self.get(args)
        return self.get(*args)

    def getdir(self, key, *args, only=False, create=False, remove=False):
        if key not in self._dirs:
            raise ValueError(f"Option can only be one of '{', '.join(self._dirs)}'")
        path, kwargs = self._get_path_args(key, *args)
        return self._dirpath(path, only, create, remove, **kwargs)

    def getfile(self, key, *args, only=False):
        if key not in self._files:
            raise ValueError(f"Option can only be one of '{', '.join(self._files)}'")
        path, kwargs = self._get_path_args(key, *args)
        return self._path(path, only, **kwargs)

    def get(self, key, *args, only=False):
        if key in self._dirs:
            return self.getdir(key, *args, only=only)
        else:
            return self.getfile(key, *args, only=only)

    def basename(self, software, charge, mult):
        return f'{self.name}_{software.hash(charge, mult)}', software.fileending

    def hessian_filename(self, software, charge, mult):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_hessian.{ending}'

    def hessian_charge_filename(self, software, charge, mult):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_hessian_charge.{ending}'

    def charge_filename(self, software, charge, mult):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_charge.{ending}'

    def scan_filename(self, software, charge, mult):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_sp.{ending}'

    def scan_sp_filename(self, software, charge, mult, i):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_sp_step_{i:02d}.{ending}'

    def preopt_filename(self, software, charge, mult):
        base, ending = self.basename(software, charge, mult)
        return f'{base}_preopt.{ending}'

    def _path(self, path, only, **kwargs):
        """Path to the folder"""
        if isinstance(path, Template):
            path = path.substitute(**kwargs)
        if only is True:
            return Path(path)
        return self.jobdir / path

    def _dirpath(self, path, only, create, remove, **kwargs):
        path = self._path(path, only, **kwargs)
        if remove is True:
            if os.path.exists(path):
                shutil.rmtree(path)
        if create is True:
            os.makedirs(path, exist_ok=True)
        return path

    def _get_path_args(self, key, *args):
        path, arguments = self.pathways[key]
        #
        if len(args) != len(arguments):
            if len(args) > len(arguments):
                raise TypeError(f"getdir() takes {len(arguments)} "
                                f"positional arguments but {len(args)} were given")
            else:
                raise TypeError(f"getdir() missing {len(arguments)} "
                                "required positional arguments: "
                                f"""{" and ".join(f"'arg'" for arg in arguments[len(args):])}""")
        return path, {arg: value for arg, value in zip(arguments, args)}
