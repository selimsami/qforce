import pytest
#
import os
#
from colt.ask import AskQuestions


path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def configini():
    return os.path.join(path, "config.ini")


@pytest.fixture
def askini():
    return os.path.join(path, "askq.ini")


@pytest.fixture
def configiniout():
    return os.path.join(path, "configout.ini")


def get_content(filename):
    with open(filename, "r") as f:
        txt = f.read()
    return txt


@pytest.fixture
def questions():
    return """
      value = 2 :: int :: [1, 2, 3]
      # hallo ihr
      # ihr auch
      name = :: str :: [hallo, du]
      ilist = :: ilist
      flist = 1.2 3.8 :: flist
      optional = :: str, optional

      [qm]
      nqm = 100 :: int
      nmm = 200 :: int
      optional = :: str, optional

      [examplecase(yes)]
      a = 10
      [examplecase(no)]
      a = 666

      [examplecase(no)::further]
      a = 666

      [examplecase(no)::further::andmore]
      a = 666
      select = :: str

      [examplecase(no)::further::andmore::select(yes)]
      a = yes

      [examplecase(no)::further::andmore::select(no)]
      a = no

      [examplecase(no)::further::andmore::select(maybe)]
      a = maybe :: str :: :: What was the question?
    """


# def test_basic_ask_questions(questions):
#     questions = AskQuestions(questions)
#     assert type(questions.questions) == Questions


def test_basic_ask_questions_from_configfile(questions, askini):
    questions = AskQuestions(questions, config=askini)
    answers = questions.ask()
    assert answers['value'] == 2


def test_basic_ask_questions_from_config(questions, configini):
    questions = AskQuestions(questions, config=configini)
    answers = questions.ask()
    assert answers['qm']['nqm'] == 100
    assert answers['qm']['optional'] is None
    assert answers['qm']['nmm'] == 200
    assert answers['examplecase']['a'] == '666'
    assert answers['examplecase']['further']['a'] == '131'
    assert answers['examplecase']['further']['andmore']['select'] == 'maybe'
    assert answers['examplecase']['further']['andmore']['select']['a'] == 'maybe'


def test_basic_ask_questions_from_config_file(questions, configini, configiniout):
    questions = AskQuestions(questions, config=configini)
    answers = questions.ask(configiniout)
    assert answers['optional'] is None
    assert answers['qm']['nqm'] == 100
    assert answers['qm']['nmm'] == 200
    assert answers['qm']['optional'] is None
    assert answers['examplecase']['a'] == '666'
    assert answers['examplecase']['further']['a'] == '131'
    assert answers['examplecase']['further']['andmore']['select'] == 'maybe'
    assert answers['examplecase']['further']['andmore']['select']['a'] == 'maybe'
    # assert get_content(configini) == get_content(configiniout)


def test_basic_ask_questions_from_config_checkonly_pass(questions, configini):
    questions = AskQuestions(questions, config=configini)
    answers = questions.check_only()
    assert answers['qm']['nqm'] == 100
    assert answers['qm']['nmm'] == 200
    assert answers['examplecase']['a'] == '666'
    assert answers['examplecase']['further']['a'] == '131'
    assert answers['examplecase']['further']['andmore']['select'] == 'maybe'
    assert answers['examplecase']['further']['andmore']['select']['a'] == 'maybe'


def test_basic_ask_questions_from_config_checkonly_pass_file(questions, configini, configiniout):
    questions = AskQuestions(questions, config=configini)
    answers = questions.check_only()
    assert answers['qm']['nqm'] == 100
    assert answers['qm']['nmm'] == 200
    assert answers['examplecase']['a'] == '666'
    assert answers['examplecase']['further']['a'] == '131'
    assert answers['examplecase']['further']['andmore']['select'] == 'maybe'
    assert answers['examplecase']['further']['andmore']['select']['a'] == 'maybe'
    # assert get_content(configini) == get_content(configiniout)


def test_basic_ask_questions_from_config_checkonly_fail_default_set(questions, configini):
    questions = questions.replace("nqm = 200 :: int", "nqm = yes :: bool")
    questions = AskQuestions(questions, config=configini)
    answers = questions.check_only()
    assert answers['qm']['nqm'] == 100
    assert answers['qm']['nmm'] == 200
    assert answers['examplecase']['a'] == '666'
    assert answers['examplecase']['further']['a'] == '131'
    assert answers['examplecase']['further']['andmore']['select'] == 'maybe'
    assert answers['examplecase']['further']['andmore']['select']['a'] == 'maybe'
