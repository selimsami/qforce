# examples/basic_colt.py
from colt import Colt


class Example(Colt):
    """Basic Example of Colt class

    _questions: is a class string and contains
                the questions to be ask from commandline 
                or read from the input file
    
    from_config: classmethod, used to initialize the 
                 class after reading the data from user input
    """

    _questions = """
    natoms = :: int :: >1
    nstates = :: int :: >1
    [options]
    do_force = True :: bool
    """

    @classmethod
    def from_config(cls, config):
        """The main function to be implemented to use
           Colt's features, it uses config data to
           initialize the class"""
        return cls(config['natoms'], config['nstates'], config['options'])

    def __init__(self, natoms, nstates, options):
        self.natoms = natoms
        self.nstates = nstates
        self.options = options
        print("initialized Example")

if __name__ == '__main__':
    Example.from_commandline()
