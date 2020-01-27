import re
#
from .dihedral_terms import DihedralTerms
from .non_dihedral_terms import BondTerm, AngleTerm, UrayAngleTerm
#
from .baseterms import MappingIterator
#
from .topology import QM, Topology


class Terms(MappingIterator):

    _term_factories = {
            'bond': BondTerm,
            'angle': AngleTerm,
            'uray': UrayAngleTerm,
            'dihedral': DihedralTerms,
    }

    regex_substring = re.compile(r"(?P<key>\w+)\((?P<subkey>.*)\)")

    def __init__(self, topo, ignore=['dihedral(rigid)']):
        _terms = {name: factory.get_terms(topo)
                  for name, factory in self._term_factories.items()}
        ignore = self._set_ignore(_terms, ignore)
        # enable iteration
        MappingIterator.__init__(self, _terms, ignore)

    def _set_ignore(self, terms, ignore):
        regular_ignore = []

        for ign in ignore:
            key, subkey = self._get_substring(ign)
            if subkey is None:
                regular_ignore.append(subkey)
                continue
            if key not in terms:
                print(f"WARNING: '{key}' not known, therefore ignored!")
                continue
            iterm = terms[key]
            if not isinstance(iterm, MappingIterator):
                print(f"WARNING: '{key}({subkey})' cannot be modified therefore ignored!")
            iterm.add_ignore_key(subkey)

        return regular_ignore

    @classmethod
    def _get_substring(cls, string):
        if '(' not in string and ')' not in string:
            return string, None
        match = cls.regex_substring.match(string)
        if match is None:
            raise ValueError("Could not pass '%s'" % string)
        return match.group('key'), match.group('subkey')


if __name__ == '__main__':
    qm = QM('freq', 'thio3.fchk', 'thio3.out')
    topo = Topology(qm.atomids, qm.coords, qm, 5)
    for term in Terms(topo):
        print(term)
