import pkg_resources
qforce_data = pkg_resources.resource_filename('qforce', 'data')

from .fit import fit_forcefield
__all__ = ['fit_forcefield']
