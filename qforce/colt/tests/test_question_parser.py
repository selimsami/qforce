"""Test all Parser Types and there corresponding values

currently the following options are checked:

    - str
    - float
    - int
    - bool
    - list
    - ilist
    - ilist_np
    - flist
    - flist_np


"""
import pytest
import numpy.random as random
import numpy as np
#
import colt.validator as LineParser
from colt.validator import Validator


def check_solution(values, solutions, is_array=False):

    assert type(values) == type(solutions)

    if is_array is True:
        assert all(value == solution for value, solution in zip(values, solutions))
    else:
        assert values == solutions


@pytest.fixture
def check_uniform_types():
    def check_solutions(typ, solution, solution_string, choices=None, is_array=False):
        validator = Validator(typ, choices=choices)
        answer = validator.validate(solution_string)
        check_solution(answer, solution, is_array)
    return check_solutions


# simple tests
def test_parser_bool_failure():
    with pytest.raises(ValueError):
        LineParser.bool_parser("hamster")


def test_parser_bool_True():
    assert LineParser.bool_parser("True") is True


def test_parser_bool_true():
    assert LineParser.bool_parser("true") is True


def test_parser_bool_yes():
    assert LineParser.bool_parser("yes") is True


def test_parser_bool_y():
    assert LineParser.bool_parser("y") is True


def test_parser_bool_False():
    assert LineParser.bool_parser("False") is False


def test_parser_bool_false():
    assert LineParser.bool_parser("false") is False


def test_parser_bool_no():
    assert LineParser.bool_parser("no") is False


def test_parser_bool_n():
    assert LineParser.bool_parser("n") is False


def test_parser_list_csv():
    check_solution(LineParser.list_parser("a, b, c"), ['a', 'b', 'c'], is_array=True)


def test_parser_list_ssv():
    check_solution(LineParser.list_parser("a  b  c"), ['a', 'b', 'c'], is_array=True)


def test_parser_ilist_csv():
    check_solution(LineParser.ilist_parser("1, 11, 8"), [1, 11, 8], is_array=True)


def test_parser_ilist_ssv():
    check_solution(LineParser.ilist_parser("1 11 8"), [1, 11, 8], is_array=True)


def test_parser_ilist_np_string_error():
    with pytest.raises(ValueError):
        check_solution(LineParser.ilist_np_parser("hallo"), None, is_array=True)


def test_parser_ilist_np_csv():
    check_solution(LineParser.ilist_np_parser("1, 11, 8"), np.array([1, 11, 8]), is_array=True)


def test_parser_ilist_np_ssv():
    check_solution(LineParser.ilist_np_parser("1 11 8"), np.array([1, 11, 8]), is_array=True)


def test_parser_flist_string_error():
    with pytest.raises(ValueError):
        check_solution(LineParser.flist_parser("hallo"), None, is_array=True)


def test_parser_flist_csv():
    check_solution(LineParser.flist_parser("1.1, 11.23, 8.1231"),
                   [1.1, 11.23, 8.1231], is_array=True)


def test_parser_flist_ssv():
    check_solution(LineParser.flist_parser("1.1 11.23 8.1231"),
                   [1.1, 11.23, 8.1231], is_array=True)


def test_parser_flist_np_string():
    with pytest.raises(ValueError):
        check_solution(LineParser.flist_np_parser("hallo"),
                       np.array([1.1, 11.23, 8.1231]), is_array=True)


def test_parser_flist_np_csv():
    check_solution(LineParser.flist_np_parser("1.1, 11.23, 8.1231"),
                   np.array([1.1, 11.23, 8.1231]), is_array=True)


def test_parser_flist_np_ssv():
    check_solution(LineParser.flist_np_parser("1.1 11.23 8.1231"),
                   np.array([1.1, 11.23, 8.1231]), is_array=True)


# tests via questions module
def test_question_generator_bool_True(check_uniform_types):
    solution = True
    solution_string = True
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_False(check_uniform_types):
    solution = False
    solution_string = False
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_true(check_uniform_types):
    solution = True
    solution_string = "true"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_false(check_uniform_types):
    solution = False
    solution_string = "false"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_yes(check_uniform_types):
    solution = True
    solution_string = "yes"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_no(check_uniform_types):
    solution = False
    solution_string = "no"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_y(check_uniform_types):
    solution = True
    solution_string = "y"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_bool_n(check_uniform_types):
    solution = False
    solution_string = "n"
    check_uniform_types("bool", solution, solution_string)


def test_question_generator_strings(check_uniform_types):
    solution = "hallo hand"
    solution_string = solution
    check_uniform_types("str", solution, solution_string)


def test_question_generator_integers(check_uniform_types):
    solution = 10
    solution_string = str(solution)
    check_uniform_types("int", solution, solution_string)


def test_question_generator_floats(check_uniform_types):
    solution = random.random()
    solution_string = str(solution)
    check_uniform_types("float", solution, solution_string)


def test_question_generator_lists(check_uniform_types):
    solution = ['halo', 'qchem', 'gaussian']
    solution_string = str(solution)
    check_uniform_types("list", solution, solution_string, is_array=True)


def test_question_generator_ilists_list(check_uniform_types):
    solution = [random.randint(100) for _ in range(random.randint(100))]
    solution_string = solution
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_range(check_uniform_types):
    solution = list(range(10, 101))
    solution_string = "10~100"
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_multiple_range(check_uniform_types):
    solution = [1, 8, 12, 13, 14, 15] + list(range(10, 101))
    solution_string = "1, 8, 12~15, 10~100"
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_range_inversed_bonds(check_uniform_types):
    solution = [1, 8, 12, 13, 14, 15] + list(range(10, 101))
    solution_string = "1, 8, 12~15, 100~10"
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_range_negative(check_uniform_types):
    solution = [1, 8, 12, 13, 14, 15] + list(range(-10, -4))
    solution_string = "1, 8, 12~15, -10~-5"
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_list_string(check_uniform_types):
    solution = [random.randint(100) for _ in range(random.randint(100))]
    solution_string = str(solution)
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_string_csv(check_uniform_types):
    solution = [random.randint(100) for _ in range(random.randint(100))]
    solution_string = str(solution).replace("[", "").replace("]", "")
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_string_ssv(check_uniform_types):
    solution = [random.randint(100) for _ in range(random.randint(100))]
    solution_string = str(solution).replace(",", " ")
    check_uniform_types("ilist", solution, solution_string, is_array=True)


def test_question_generator_ilists_np(check_uniform_types):
    solution = np.array([random.randint(100) for _ in range(random.randint(100))])
    solution_string = solution
    check_uniform_types("ilist_np", solution, solution_string, is_array=True)


def test_question_generator_ilists_np_string_csv(check_uniform_types):
    solution = np.array([random.randint(100) for _ in range(random.randint(100))])
    solution_string = str(solution)
    check_uniform_types("ilist_np", solution, solution_string, is_array=True)


def test_question_generator_ilists_np_string_ssv(check_uniform_types):
    solution = np.array([random.randint(100) for _ in range(random.randint(100))])
    solution_string = str(solution).replace(",", " ")
    check_uniform_types("ilist_np", solution, solution_string, is_array=True)


def test_question_generator_flists(check_uniform_types):
    solution = [1.2, 2.1, 3.3, 4.8, 5.1, 6.10]
    solution_string = solution
    check_uniform_types("flist", solution, solution_string, is_array=True)


def test_question_generator_flists_string_csv(check_uniform_types):
    solution = [1.2, 2.1, 3.3, 4.8, 5.1, 6.10]
    solution_string = str(solution)
    check_uniform_types("flist", solution, solution_string, is_array=True)


def test_question_generator_flists_string_ssv(check_uniform_types):
    solution = [1.2, 2.1, 3.3, 4.8, 5.1, 6.10]
    solution_string = str(solution).replace(",", " ")
    check_uniform_types("flist", solution, solution_string, is_array=True)


def test_question_generator_flists_np(check_uniform_types):
    solution = np.array([1.2, 2.1, 3.3, 4.8, 5.1, 6.10])
    solution_string = solution
    check_uniform_types("flist_np", solution, solution_string, is_array=True)


def test_question_generator_flists_np_string_csv(check_uniform_types):
    solution = np.array([1.2, 2.1, 3.3, 4.8, 5.1, 6.10])
    solution_string = str(solution)
    check_uniform_types("flist_np", solution, solution_string, is_array=True)


def test_question_generator_flists_np_string_ssv(check_uniform_types):
    solution = np.array([1.2, 2.1, 3.3, 4.8, 5.1, 6.10])
    solution_string = str(solution).replace(",", " ")
    check_uniform_types("flist_np", solution, solution_string, is_array=True)


def test_question_generator_python_nparray(check_uniform_types):
    solution = np.array([1.2, 2.1, 3.3, 4.8, 5.1, 6.10])
    solution_string = "[1.2, 2.1, 3.3, 4.8, 5.1, 6.10]"
    check_uniform_types("python(np.array)", solution, solution_string, is_array=True)


def test_question_generator_python_dict(check_uniform_types):
    solution = {'hi': 1.2, 'du': {1: 2, 3: 4}}
    solution_string = "{'hi': 1.2, 'du': {1: 2, 3: 4}}"
    check_uniform_types("python(dict)", solution, solution_string)


def test_question_generator_python_list(check_uniform_types):
    solution = [1, 2, 'hi', 3]
    solution_string = "[1, 2, 'hi', 3]"
    check_uniform_types("python(list)", solution, solution_string)
