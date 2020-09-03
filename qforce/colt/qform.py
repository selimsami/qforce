from abc import abstractmethod, ABC
from collections import UserDict, UserString
from collections.abc import Mapping
from contextlib import contextmanager
#
from .answers import AnswersBlock, SubquestionsAnswer
from .config import ConfigParser
from .generator import GeneratorNavigator
#
from .questions import QuestionASTGenerator
from .questions import QuestionASTVisitor
from .questions import Component
#
from .presets import PresetGenerator
from .validator import Validator, NOT_DEFINED, file_exists
from .validator import ValidatorErrorNotChoicesSubset, ValidatorErrorNotInChoices
from .validator import Choices, RangeExpression


join_case = GeneratorNavigator.join_case
join_keys = GeneratorNavigator.join_keys
rsplit_keys = GeneratorNavigator.rsplit_keys


def split_keys(name):
    block, key = rsplit_keys(name)
    if key is None:
        key = block
        block = ""
    return block, key


def is_existing_file(config):
    try:
        config = file_exists(config)
        return True
    except ValueError:
        return False


class _QuestionsContainerBase(Component):
    """Base class to contain question containers"""

    __slots__ = ('name', 'parent_name', '_name')

    def __init__(self, name, qform):
        #
        self.name = name
        #
        self.parent_name, self._name = split_keys(name)
        # register block
        qform.blocks[name] = self

    @property
    def label(self):
        return self._name


class _ConcreteQuestionBase(Component):
    """Logic of each actual question"""

    __slots__ = ("name", "parent_name", "_name", "is_set")

    def __init__(self, name):
        self.name = name
        self.parent_name, self._name = split_keys(name)
        self.is_set = False

    @property
    def answer(self):
        """Used to set/get user input, needs to be overwritten"""
        return ""

    @property
    def accept_empty(self):
        return False

    @property
    def label(self):
        return self._name

    @property
    def id(self):
        return self.name

    @abstractmethod
    def get_answer(self):
        """get answer"""

    @abstractmethod
    def get_answer_as_string(self):
        """get string of answer"""

    @abstractmethod
    def preset(self, value, choices):
        """preset new value and choices!"""

    def set(self, answer, on_empty_entry=lambda answer, self: None,
            on_value_error=lambda answer, self: None, on_wrong_choice=lambda answer, self: None):
        """ Handle all set events """
        if answer == "":
            if self.accept_empty is False:
                on_empty_entry(answer, self)
            return self.accept_empty
        #
        try:
            self.answer = answer
            return True
        except ValueError:
            on_value_error(answer, self)
        except ValidatorErrorNotInChoices:
            on_wrong_choice(answer, self)
        return False

    def get(self):
        return self.answer


class LiteralBlockString(UserString):
    """UserString to contain a literalblock, the string can also be empty"""

    def __init__(self, string):
        if string is None:
            self.is_none = True
            string = ''
        elif isinstance(string, LiteralBlockString):
            self.is_none = string.is_none
        else:
            self.is_none = False
        #
        UserString.__init__(self, string)


class LiteralBlock(_ConcreteQuestionBase):
    """LiteralBlock Node"""

    __slots__ = ("_answer",)

    def __init__(self, name, qform):
        #
        _ConcreteQuestionBase.__init__(self, name)
        # register self
        qform.literals[name] = self
        #
        self._answer = LiteralBlockString(None)
        #

    @property
    def is_optional(self):
        return True

    @property
    def answer(self):
        answer = self.get_answer()
        if answer is NOT_DEFINED:
            return ""
        return LiteralBlockString(answer)

    @answer.setter
    def answer(self, value):
        self._answer = LiteralBlockString(value)
        self.is_set = True

    @property
    def accept_empty(self):
        return True

    def preset(self, value, choices):
        raise Exception(f"preset not defined for Literalblock")

    def get_answer(self):
        if self._answer.is_none is True:
            return None
        return self._answer.data

    def get_answer_as_string(self):
        """get string of answer"""
        if self._answer.is_none is True:
            return None
        return self._answer

    def accept(self, visitor):
        return visitor.visit_literal_block(self)


class ConcreteQuestion(_ConcreteQuestionBase):
    """Concrete question"""

    __slots__ = ("_value", "_comment", "is_subquestion_main",
                 "question", "typ", "is_optional")

    def __init__(self, name, question, is_subquestion=False):
        _ConcreteQuestionBase.__init__(self, name)
        #
        self._value = Validator(question.typ, default=question.default, choices=question.choices)
        #
        if question.comment is NOT_DEFINED:
            self._comment = None
        else:
            self._comment = question.comment
        #
        self.question = question.question
        self.typ = question.typ

        self.is_optional = question.is_optional
        self.is_subquestion_main = is_subquestion

    @property
    def accept_empty(self):
        return self.is_optional or self._value.get() is not NOT_DEFINED

    def get_answer(self):
        """get answer back, if is optional, return None if NOT_DEFINED"""
        answer = self._value.get()
        if self.is_optional is True and answer is NOT_DEFINED:
            return None
        return answer

    def get_answer_as_string(self):
        """get answer back, if is optional, return None if NOT_DEFINED"""
        return self._value.answer_as_string()

    def accept(self, visitor):
        if isinstance(self.choices, Choices):
            return visitor.visit_concrete_question_select(self)
        return visitor.visit_concrete_question_input(self)

    @property
    def has_only_one_choice(self):
        return len(self.choices) == 1

    @property
    def comment(self):
        return self._comment

    @property
    def placeholder(self):
        choices = self.choices
        if isinstance(choices, RangeExpression):
            return f"{self.typ}, {choices.as_str()}"
        return self.typ

    @property
    def answer(self):
        answer = self._value.get()
        if answer is NOT_DEFINED:
            return ""
        return answer

    @answer.setter
    def answer(self, value):
        self._value.set(value)
        self.is_set = True

    @property
    def choices(self):
        return self._value.choices

    def set_answer(self, value):
        self._value.set(value)
        self.is_set = True

    def preset(self, value, choices):
        """preset new value, choices:
        important: first update choices to ensure that default in choices!
        """
        if choices is not None:
            self._value.choices = choices
            #
            answer = self._value.get()
            #
            if answer is NOT_DEFINED:
                self.is_set = False
        if value is not None:
            self._value.set(value)


class QuestionBlock(_QuestionsContainerBase, UserDict):
    """Store a question block"""

    def __init__(self, name, concrete, blocks, qform):
        _QuestionsContainerBase.__init__(self, name, qform)
        #
        UserDict.__init__(self)
        #
        self.concrete = concrete
        self.blocks = blocks
        #
        self.data = concrete

    @property
    def is_set(self):
        return all(question.is_set for question in self.concrete.values())

    @property
    def answer(self):
        raise Exception("Answer not available for QuestionBlock")

    def accept(self, visitor):
        return visitor.visit_question_block(self)

    def get_blocks(self):
        return sum((block.get_blocks() for block in self.blocks.values()),
                   [self.name])


class SubquestionBlock(_QuestionsContainerBase):
    """Container for the cases spliting"""

    def __init__(self, name, main_question, cases, parent):
        _QuestionsContainerBase.__init__(self, name, parent)
        #
        self.main_question = main_question
        #
        self.cases = cases

    @property
    def is_optional(self):
        return self.main_question.is_optional

    def generate_setup(self):
        answer = self.answer
        if answer in ("", None):
            return {}
        return self.cases[answer].generate_setup()

    def accept(self, visitor):
        return visitor.visit_subquestion_block(self)

    @property
    def answer(self):
        return self.main_question.answer

    @answer.setter
    def answer(self, value):
        self.main_question.answer = value

    def get_blocks(self):
        answer = self.answer
        if answer in ("", None):
            return []
        return self.cases[answer].get_blocks()

    def get_delete_blocks(self):
        return {block: None for block in self.get_blocks()}


def generate_string(name, value):
    if value is None:
        return f"{name} ="
    return f"{name} = {value}"


def answer_iter(name, dct, default_name):
    if isinstance(dct, LiteralBlockString):
        if dct.is_none is True:
            return
    else:
        if len(dct) == 0:
            return

    if name != default_name:
        yield f'[{name}]'
    else:
        yield ''

    if isinstance(dct, LiteralBlockString):
        yield dct.data
    else:
        for _name, _value in dct.items():
            yield generate_string(_name, _value)
        yield ''


class QuestionVisitor(ABC):
    """Base class to define visitors for the question form
    the entry point is always the `QuestionForm`"""

    __slots__ = ()

    def visit(self, qform, **kwargs):
        """Visit a question form"""
        return qform.accept(self, **kwargs)

    @abstractmethod
    def visit_qform(self, qform, **kwargs):
        pass

    @abstractmethod
    def visit_question_block(self, block):
        pass

    @abstractmethod
    def visit_concrete_question_select(self, question):
        pass

    @abstractmethod
    def visit_concrete_question_input(self, question):
        pass

    @abstractmethod
    def visit_literal_block(self, block):
        pass

    def visit_subquestion_block(self, block):
        """visit subquestion blocks"""
        answer = block.answer
        if answer in ("", None):
            return {}
        #
        return block.cases[answer].accept(self)

    def on_empty_entry(self, answer, question):
        pass

    def on_value_error(self, answer, question):
        pass

    def on_wrong_choice(self, answer, question):
        pass

    def set_answer(self, question, answer):
        return question.set(answer, on_empty_entry=self.on_empty_entry,
                            on_value_error=self.on_value_error,
                            on_wrong_choice=self.on_wrong_choice)

    def _visit_block(self, block):
        """visit a block, first only concrete questions, then the subblocks"""
        for question in block.concrete.values():
            question.accept(self)
        #
        for subblock in block.blocks.values():
            subblock.accept(self)


def block_not_set(block_name, keys):
    if block_name == '':
        txt = '\n'
    else:
        txt = f"\n[{block_name}]\n"
    txt += "\n".join(f"{key} = NotSet" for key in keys)
    return txt + "\n"


class ColtErrorAnswerNotDefined(Exception):
    """Error if answer is not defined"""

    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __repr__(self):
        return f"ColtErrorAnswerNotDefined:\n{self.msg}"

    def __str__(self):
        return f"ColtErrorAnswerNotDefined:\n{self.msg}"


class AnswerVistor(QuestionVisitor):
    """Visitor to collect answers from a given qform"""

    __slots__ = ('check', 'not_set')

    def __init__(self):
        self.not_set = None
        self.check = False

    def visit_qform(self, qform, check=False):
        """Visit the qform

        Raises
        ------
        ColtErrorAnswerNotDefined
            in case an answer is not defined
        """
        self.not_set = {}
        self.check = check
        answer = qform.form.accept(self)
        if check is True:
            if self.not_set != {}:
                raise ColtErrorAnswerNotDefined(self._create_exception(self.not_set))
        self.not_set = {}
        return answer

    def visit_question_block(self, block):
        """Visit the question block and store all results in a Mapping"""
        out = AnswersBlock({question.label: question.accept(self)
                            for question in block.concrete.values()})
        #
        if self.check is True:
            not_set = tuple(name for name, value in out.items() if value is NOT_DEFINED)
            if len(not_set) > 0:
                self.not_set[block.name] = not_set
        #
        out.update({name: block.accept(self) for name, block in block.blocks.items()})
        #
        return out

    def visit_subquestion_block(self, block):
        """visit subquestion blocks"""
        answer = block.main_question.get_answer()
        if answer is NOT_DEFINED:
            return SubquestionsAnswer(block.label, None, {})
        #
        return SubquestionsAnswer(block.label, answer, block.cases[answer].accept(self))

    def visit_concrete_question_select(self, question):
        return question.get_answer()

    def visit_concrete_question_input(self, question):
        return question.get_answer()

    def visit_literal_block(self, block):
        return block.get_answer()

    @staticmethod
    def _create_exception(not_set):
        return "\n".join(block_not_set(block, values) for block, values in not_set.items())


class QuestionGeneratorVisitor(QuestionASTVisitor):
    """QuestionASTVisitor, to fill a qform"""

    __slots__ = ('question_id', 'qname', 'qform', 'concrete', 'blocks')

    def __init__(self):
        """Set basic defaults to None"""
        # current question id, id is blockname::qname
        self.question_id = None
        # current question name within that block
        self.qname = None
        # qform the questions get saved in
        self.qform = None
        # current concrete question container
        self.concrete = None
        # current block container
        self.blocks = None

    def reset(self):
        """set data to initial form"""
        # current question id, id is blockname::qname
        self.question_id = None
        # current question name within that block
        self.qname = None
        # qform the questions get saved in
        self.qform = None
        # current concrete question container
        self.concrete = None
        # current block container
        self.blocks = None

    def visit_question_ast_generator(self, qgen, qform=None):
        """When visiting an ast generator"""
        # save qform in self.qform
        self.qform = qform
        # set block_name and block_id
        self.question_id = ''
        # set concrete and blocks to None
        self.concrete = None
        self.blocks = None
        # start visiting the blocks
        output = qgen.tree.accept(self)
        # reset
        self.reset()
        #
        return output

    def visit_question_container(self, block):
        """when visiting a question container"""
        # save qname
        qname = self.qname
        #
        with self.question_block() as qid:
            for key, question in block.items():
                # set qname to current key
                self.qname = key
                # set question_id
                self.question_id = join_keys(qid, key)
                # visit next item
                question.accept(self)
            # create block
            block = QuestionBlock(qid, self.concrete, self.blocks, self.qform)
        # if in main form, or within subquestions block, return the block
        if self.blocks is None:
            return block
        # else set the block
        self.blocks[qname] = block

    def visit_conditional_question(self, question):
        """visit conditional question form"""
        # create concrete_question and save it in the concrete_question block
        concrete_question = ConcreteQuestion(self.question_id, question.main, is_subquestion=True)
        self.concrete[self.qname] = concrete_question
        # enter the subquestion_block mode
        with self.subquestion_block() as (qid, block_name):
            # create empty cases dictionary
            cases = {}
            #
            for qname, quest in question.items():
                # set question_id
                self.question_id = join_case(qid, qname)
                # there are no ids for these, so question.id does not need
                # to be set, here only subblocks can be inside!
                cases[qname] = quest.accept(self)
        # save subquestion block
        self.blocks[block_name] = SubquestionBlock(qid, concrete_question, cases, self.qform)

    def visit_literal_block(self, question):
        """block needs to be in a concrete section"""
        self.concrete[self.qname] = LiteralBlock(self.question_id, self.qform)

    def visit_question(self, question):
        """question needs to be in concrete section"""
        self.concrete[self.qname] = ConcreteQuestion(self.question_id, question)

    @contextmanager
    def subquestion_block(self):
        """helper function to set defaults and reset them
        for SubquestionBlock"""
        blocks = self.blocks
        #
        self.blocks = None
        #
        yield self.question_id, self.qname
        #
        self.blocks = blocks

    @contextmanager
    def question_block(self):
        """helper function to set defaults and reset them
        for QuestionBlock"""
        # save old concrete, and blocks
        concrete = self.concrete
        blocks = self.blocks
        # create empty new ones
        self.concrete = {}
        self.blocks = {}
        #
        yield self.question_id
        # restore the old ones
        self.concrete = concrete
        self.blocks = blocks


class ErrorSettingAnswerFromFile(Exception):
    """errors setting answers from file"""
    __slots__ = ('filename', 'msg')

    def __init__(self, filename, msg):
        super().__init__()
        self.filename = filename
        self.msg = msg

    def __repr__(self):
        return f"ErrorSettingAnswerFromFile: file = '{self.filename}'\n{self.msg}"

    def __str__(self):
        return f"ErrorSettingAnswerFromFile: file = '{self.filename}'\n{self.msg}"


class ErrorSettingAnswerFromDict(Exception):
    """Error when trying to read answers from a dict"""
    __slots__ = ('msg',)

    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __repr__(self):
        return f"ErrorSettingAnswerFromDict:\n{self.msg}"

    def __str__(self):
        return f"ErrorSettingAnswerFromDict:\n{self.msg}"


class QuestionForm(Mapping, Component):
    """Main interface to the question forms"""
    #
    __slots__ = ('blocks', 'literals', 'unset', 'form')
    # visitor to generate answers
    answer_visitor = AnswerVistor()
    # visitor to generate question forms
    question_generator_visitor = QuestionGeneratorVisitor()

    def __init__(self, questions, config=None, presets=None):
        #
        self.blocks = {}
        # literal blocks
        self.literals = {}
        # not set variables
        self.unset = {}
        # generate Question Forms
        self.form = self._generate_forms(questions)
        #
        self.set_answers_and_presets(config, presets)

    def _generate_forms(self, questions):
        questions = QuestionASTGenerator(questions)
        return self.question_generator_visitor.visit(questions, qform=self)

    def accept(self, visitor, **kwargs):
        return visitor.visit_qform(self, **kwargs)

    @property
    def is_all_set(self):
        """check if all questions are answered"""
        return all(block.is_set for block in self.values())

    def set_answer(self, name, answer):
        """Set the answer of a question"""
        if answer == "":
            return False
        #
        block, key = self._split_keys(name)
        #
        try:
            block.concrete[key].answer = answer
            is_set = True
        except ValueError:
            is_set = False
        except ValidatorErrorNotInChoices:
            is_set = False
        #
        return is_set

    def get_answers(self, check=True):
        """Get the answers from the forms

        Parameter
        ---------

        check: bool
            if False, raise no exception in case answers are not set
                      missing answers are given as empty strings

        Returns
        -------
        AnswersBlock
            dictionary with the parsed and validated userinput

        Raises
        ------
        ColtErrorAnswerNotDefined
            if `check` is True, raises error in case answers are not given
        """
        return self.answer_visitor.visit(self, check=check)

    def get_blocks(self):
        """return blocks"""
        return self.form.get_blocks()

    def write_config(self, filename):
        """ get a linear config and write it to the file"""
        config = {}
        for blockname in self.get_blocks():
            # normal blocks
            config[blockname] = {key: question.get_answer_as_string()
                                 for key, question in self.blocks[blockname].concrete.items()
                                 if not isinstance(question, LiteralBlock)}
            # add literal blocks
            config.update({question.id: question.get_answer_as_string()
                           for _, question in self.blocks[blockname].concrete.items()
                           if isinstance(question, LiteralBlock)})

        default_name = ''
        with open(filename, 'w') as fhandle:
            fhandle.write("\n".join(answer for key, answers in config.items()
                                    for answer in answer_iter(key, answers, default_name)))

    def set_answers_from_file(self, filename, raise_error=True):
        errmsg = self._set_answers_from_file(filename)
        if raise_error is True and errmsg is not None:
            raise ErrorSettingAnswerFromFile(filename, errmsg)

    def set_answers_from_dct(self, dct, raise_error=True):
        errmsg = self._set_answers_from_dct(dct)
        if raise_error is True and errmsg is not None:
            raise ErrorSettingAnswerFromDict(errmsg)

    def set_answers_and_presets(self, config=None, presets=None, raise_error=True):
        """set both presets and answers"""
        if presets is not None:
            self.set_presets(presets)

        if config is not None:
            if isinstance(config, Mapping):
                self.set_answers_from_dct(config, raise_error=raise_error)
            elif is_existing_file(config):
                self.set_answers_from_file(config, raise_error=raise_error)

    def set_presets(self, presets):
        """reset some of the question possibilites"""
        presets = PresetGenerator(presets).tree
        #
        for blockname, fields in presets.items():
            if blockname not in self.blocks:
                print(f"Unknown block {blockname} in presets, continue")
                continue
            block = self.blocks[blockname]
            for key, preset in fields.items():
                if key not in block:
                    print(f"Unknown key {key} in {blockname} in presets, continue")
                    continue
                try:
                    block[key].preset(preset.default, preset.choices)
                except ValidatorErrorNotChoicesSubset:
                    print((f"Could not update choices in '{blockname}' entry '{key}' as choices ",
                           "not subset of previous choices, continue"))

    def __iter__(self):
        return iter(self.get_blocks())

    def __len__(self):
        return len(self.get_blocks())

    def __getitem__(self, key):
        return self.blocks[key]

    def _set_answers_from_file(self, filename):
        """Set answers from a given file"""
        #
        try:
            parsed, literals = ConfigParser.read(filename, self.literals)
        except FileNotFoundError:
            return f"File '{filename}' not found!"
        # set literal blocks
        for key, value in literals.items():
            if value in (None, ''):
                continue
            self.literals[key].answer = value
        #
        return self._set_answers_from_dct(parsed)

    def _set_answers_from_dct(self, dct):
        """Set the answers from a dictionary"""
        #
        errstr = ""
        #
        for blockname, answers in dct.items():
            if blockname == ConfigParser.base:
                blockname = ""

            if blockname not in self.blocks:
                if blockname in self.literals:
                    self.literals[blockname].answer = answers
                    continue
                print(f"""Section = {blockname} unknown, maybe typo?""")
                continue

            errstr += self._set_block_answers(blockname, answers)
        #
        if errstr == "":
            return None
        return errstr

    def _set_block_answers(self, blockname, answers):
        if blockname != "":
            error = f"[{blockname}]"
        else:
            error = ""

        errmsg = ""
        block = self.blocks[blockname]
        for key, answer in answers.items():
            if key not in block:
                print("key not known")
                continue
            question = block[key]
            if answer == "":
                if question.is_optional:
                    question.is_set = True
                continue
            #
            try:
                question.answer = answer
            except ValueError:
                errmsg += f"\n{key} = {answer}, ValueError"
            except ValidatorErrorNotInChoices as err_choices:
                errmsg += (f"\n{key} = {answer}, Wrong Choice: can only be "
                           f"{err_choices}")
        if errmsg != "":
            return error + errmsg
        return ""

    def _split_keys(self, name):
        block, key = split_keys(name)

        if block not in self.blocks:
            raise Exception("block unknown")

        return self.blocks[block], key
