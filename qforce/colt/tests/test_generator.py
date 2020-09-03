import pytest
#
from colt.generator import Generator


@pytest.fixture
def dict_generator():

    class DictGenerator(Generator):

        leafnode_type = str

        def leaf_from_string(self, name, value, parent):
            """Create a leaf from an entry in the config file

            Args:
                name (str):
                    name of the entry

                value (str):
                    value of the entry in the config

            Kwargs:
                parent (str):
                    identifier of the parent node

            Returns:
                A leaf node

            Raises:
                ValueError:
                    If the value cannot be parsed
            """
            return value.strip()
    return DictGenerator


def test_dict_generator_basic(dict_generator):

    string = """
    a = 100
    b = 200
    c = name
    [system]
    natoms = 8
    methods = tddft
    """
    out = dict_generator(string)
    assert out.tree == {
            'a': "100",
            'b': "200",
            'c': "name",
            "system": {
                'natoms': '8',
                'methods': 'tddft',
                }
            }


def test_dict_generator_tree_from_tree(dict_generator):

    string = """
    a = 100
    b = 200
    c = name
    [system]
    natoms = 8
    methods = tddft
    """
    out = dict_generator(string)
    out = dict_generator(out)
    assert out.tree == {
            'a': "100",
            'b': "200",
            'c': "name",
            "system": {
                'natoms': '8',
                'methods': 'tddft',
                }
            }


def test_dict_generator_from_int(dict_generator):
    with pytest.raises(TypeError):
        dict_generator(1)


def test_dict_generator_from_float(dict_generator):
    with pytest.raises(TypeError):
        dict_generator(8.88)


def test_dict_generator_fail_branching(dict_generator):
    string = """
    a = 100
    b = 200
    c = name
    [system(a)]
    natoms = 8
    methods = tddft
    """
    with pytest.raises(NotImplementedError):
        dict_generator(string)
