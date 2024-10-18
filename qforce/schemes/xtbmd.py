import os
from shutil import copy2 as copy
import numpy as np
#
from .helper import coords_ids_iter, write_xyz
from .creator import CalculationStorage, CustomStructureCreator
from .generalcreators import EnergyCalculationIterCreator, GradientCalculationIterCreator


class XTBMolecularDynamics(CustomStructureCreator):

    _user_input = """
    weight = 1 :: float
    # What to compute:
    #   none: do not do metadynamics, default
    #     en: do metadynamics and do single point energy calculations
    #   grad: do metadynamics and do single point energy+ gradient calculations
    #
    compute = none :: str :: [none, en, grad]
    # Number of frames used for fitting
    n_fit = 100 :: int
    # Number of frames used for validation
    n_valid = 0 :: int
    # temperature (K)
    temp = 800 :: float :: >0
    # interval for trajectory printout (fs)
    dump = 500.0 :: float :: >0
    # time step for propagation (fs)
    step = 2.0 :: float :: >0
    # Bond constraints (0: none, 1: H-only, 2: all)
    shake = 0
    """

    def __init__(self, folder, structures, weight, compute, config):
        super().__init__(weight, folder=folder)
        self.compute = compute

        self.n_total_fit = config['n_fit']
        self.n_total_valid = config['n_valid']
        self.n_traj = len(structures)
        self.n_fit_dist = self.distribute_sim_count(config['n_fit'])
        self.n_valid_dist = self.distribute_sim_count(config['n_valid'])
        self.total_frame_dist = self.n_fit_dist + self.n_valid_dist
        self.time_dist = config['dump'] * self.total_frame_dist / 1e3

        self.xtbinput = {}
        for item in ['temp', 'dump', 'step', 'shake']:
            self.xtbinput[item] = config[item]

        self._calcs = []
        self._init_structs = structures
        self._mds = CalculationStorage()

    @classmethod
    def from_config(cls, config, folder, structures):
        if config['compute'] == 'none':
            return None
        return cls(folder, structures, config['weight'], config['compute'], config)

    def setup_pre(self, qm):
        """setup md calculation"""
        mdinput = "$md\n"
        for key, value in self.xtbinput.items():
            mdinput += f"{key} = {value}\n"

        calcs = []
        for i, (coords, ids) in enumerate(self._init_structs):
            folder = self.folder / f'{i}_md'
            os.makedirs(folder, exist_ok=True)
            calc = qm.Calculation('xtb.inp',
                                  {'traj': ['xtb.trj']},
                                  folder=folder,
                                  software='xtb')

            write_xyz(folder / 'xtb.xyz', ids, coords, comment=f"Structure {i}")
            with open(folder / 'md.inp', 'w') as fh:
                fh.write(mdinput)
                fh.write(f'time = {self.time_dist[i]}\n')
                fh.write('$end\n')

            with open(folder / 'xtb.inp', 'w') as fh:
                fh.write("xtb xtb.xyz --input md.inp --md --ceasefiles > md.log ")

            calcs.append(calc)
        #
        self._mds.calculations = calcs

    def check_pre(self):
        """check that the md calculation was finished"""
        return self._check(self._mds.calculations)

    def parse_pre(self, qm):
        results = []
        for calc in self._mds.calculations:
            traj = calc.check()['traj']
            result = calc.folder / 'mdresult.xyz'
            copy(traj, result)
            results.append({'file': result})
        self._mds.results = results

    def setup_main(self, qm):
        # setup
        parent = self.folder
        if self.compute == 'en':
            ComputeCls = EnergyCalculationIterCreator
        elif self.compute == 'grad':
            ComputeCls = GradientCalculationIterCreator
        else:
            raise ValueError("do not know compute method!")
        # currently only one result there

        for i, mdrun in enumerate(self._mds.results):
            filename = mdrun['file']

            data = coords_ids_iter(filename, f':{self.total_frame_dist[i]}')
            folder = parent / f'{i}_{self.compute}_structs'
            self._calcs.append(ComputeCls(folder, self.weight, data))

        # actually setup the calculations
        for calc in self._calcs:
            calc.setup_main(qm)

    def check_main(self):
        for calc in self._calcs:
            res = calc.check_main()
            if res is not None:
                return res
        return None

    def parse_main(self, qm):
        for calc in self._calcs:
            calc.parse_main(qm)

    def enouts(self, select='all'):
        res = []
        start, end = self.do_selection(select)
        for i, calc in enumerate(self._calcs):
            res += calc.enouts()[start[i]:end[i]]
        return res

    def gradouts(self, select='all'):
        res = []
        start, end = self.do_selection(select)
        for i, calc in enumerate(self._calcs):
            res += calc.gradouts()[start[i]:end[i]]
        return res

    def hessouts(self, select='all'):
        res = []
        start, end = self.do_selection(select)
        for i, calc in enumerate(self._calcs):
            res += calc.hessouts()[start[i]:end[i]]
        return res

    def distribute_sim_count(self, n_total):
        n = np.full(self.n_traj, n_total // self.n_traj)
        n[:n_total % self.n_traj] += 1
        return n.astype(int)

    def do_selection(self, select):
        if select == 'all':
            start, end = np.zeros(self.n_traj, dtype=int), self.total_frame_dist
        elif select == 'fit':
            start, end = np.zeros(self.n_traj, dtype=int), self.n_fit_dist
        elif select == 'valid':
            start, end = self.n_fit_dist, self.total_frame_dist
        else:
            raise ValueError(f'Wrong selection with: {select}')
        return start, end
