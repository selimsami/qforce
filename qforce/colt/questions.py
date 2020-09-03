"""Definitions of all Question Classes"""
from abc import abstractmethod, ABC
from collections import UserDict
#
from .generator import Generator, BranchingNode
#
from .validator import NOT_DEFINED
from .slottedcls import slottedcls


class Component(ABC):
    """Basic Visitor Component"""

    @abstractmethod
    def accept(self, visitor):
        """accept a visitor"""


class QuestionASTVisitor(ABC):
    """Basic class to visit the nodes of the QuestionASTGenerator"""

    __slots__ = ()

    def visit(self, qgen, **kwargs):
        """start to visit the QuestionASTGenerator"""
        return qgen.accept(self, **kwargs)

    @abstractmethod
    def visit_question_ast_generator(self, qgen, **kwargs):
        """visit the ast generator"""

    @abstractmethod
    def visit_question_container(self, block):
        """visit the question container"""

    @abstractmethod
    def visit_literal_block(self, question):
        """visit a literal block"""

    @abstractmethod
    def visit_question(self, block):
        """visit a concrete question"""

    @abstractmethod
    def visit_conditional_question(self, block):
        """visit a conditional question"""


class Question(Component):
    """Question Node in the QuestionASTGenerator"""

    __slots__ = ('question', 'typ', 'default', 'choices', 'comment', 'is_optional')

    def __init__(self, question="", typ="str", default=NOT_DEFINED,
                 choices=None, comment=NOT_DEFINED, is_optional=False):
        self.question = question
        self.typ = typ
        self.default = default
        self.choices = choices
        self.comment = comment
        self.is_optional = is_optional

    def __eq__(self, other):
        if not isinstance(other, Question):
            return False
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__slots__)

    def accept(self, visitor):
        return visitor.visit_question(self)


class LiteralBlockQuestion(Component):
    """LiteralBlock Node in the QuestionASTGenerator"""

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Question):
            return False
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__slots__)

    def accept(self, visitor):
        return visitor.visit_literal_block(self)


class ConditionalQuestion(Component, BranchingNode):  # pylint: disable=too-many-ancestors
    """ConditionalQuestion Node in the QuestionASTGenerator
       is a branching node, used to store decissions """

    def __init__(self, name, main, subquestions):
        super().__init__(name, main, subquestions)
        self.main = self.leaf
        self.subquestions = self.subnodes
        # updatable view!
        self.main.choices = self.subquestions.keys()
        #

    @property
    def main_choices(self):
        """get back the choices for the main question"""
        return list(self.subquestions.keys())

    def __str__(self):
        return (f"ConditionalQuestion(name = {self.name},"
                f" main = {self.main}, subquestions = {self.subquestions}")

    def __repr__(self):
        return (f"ConditionalQuestion(name = {self.name},"
                f" main = {self.main}, subquestions = {self.subquestions}")

    def accept(self, visitor):
        return visitor.visit_conditional_question(self)


class QuestionContainer(Component, UserDict):
    """QuestionContainer Node in the QuestionASTGenerator"""

    def __init__(self, data=None):
        if data is None:
            data = {}
        UserDict.__init__(self, data)

    def concrete_items(self):
        """Loop only over the concrete items, and not over containers"""
        types = (ConditionalQuestion, QuestionContainer)
        for key, question in self.items():
            if not isinstance(question, types):
                yield key, question
            if isinstance(question, ConditionalQuestion):
                yield key, question.main

    def accept(self, visitor):
        return visitor.visit_question_container(self)


class QuestionASTGenerator(Component, Generator):
    """Contains all tools to automatically generate questions from
       a given file
    """

    comment_char = "###"
    default = '__QUESTIONS__'
    _allowed_choices_types = ['int', 'str', 'float', 'bool']
    # for tree generator
    leafnode_type = Question
    branching_type = ConditionalQuestion
    node_type = QuestionContainer

    LeafString = slottedcls("LeafString", {"default": NOT_DEFINED,
                                           "typ": "str",
                                           "choices": NOT_DEFINED,
                                           "question": NOT_DEFINED,
                                           "is_optional": False})

    def __init__(self, questions):
        """Main Object to generate questions from string

        Args:
            questions:  Questions object, can
                        1) Question Object, just save questions
                        2) file, read file and parse input

        Kwargs:
            isfile (bool): True, `questions` is a file
                           False, `questions` is a string

        """
        Generator.__init__(self, questions)
        #
        self.questions = self.tree

    @classmethod
    def new_branching(cls, name, leaf=None):
        """Create a new empty branching"""
        if leaf is None:
            return ConditionalQuestion(name, Question(name), QuestionContainer())
        return ConditionalQuestion(name, leaf, QuestionContainer())

    @staticmethod
    def new_node():
        return QuestionContainer()

    @staticmethod
    def tree_container():
        return QuestionContainer()

    def leaf_from_string(self, name, value, parent=None):
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
        original_value = value
        # handle comment
        value, comment = self._parse_comment(value)
        # try to parse line
        try:
            value = self._parse_string(value)
        except ValueError:
            raise ValueError(f"Cannot parse value `{original_value}`") from None
        # check for literal block
        if value.typ == 'literal':
            return LiteralBlockQuestion(name)
        # get default
        default = self._parse_default(value.default)
        # get question
        if value.question is NOT_DEFINED:
            question = name
        else:
            question = value.question
        # get choices
        choices = self._parse_choices(value.choices)
        # return leaf node
        return Question(question, value.typ, default, choices, comment, value.is_optional)

    def _parse_string(self, string):
        # set default parameters
        default = NOT_DEFINED
        typ = "str"
        choices = NOT_DEFINED
        question = NOT_DEFINED
        #
        value = tuple(ele.strip() for ele in string.split(self.seperator))
        if len(value) == 1:
            default = value[0]
        elif len(value) == 2:
            default, typ = value
        elif len(value) == 3:
            default, typ, choices = value
        elif len(value) == 4:
            default, typ, choices, question = value
        else:
            raise ValueError(f"Cannot parse string {string}")
        #
        typ, optional = self._parse_typ(typ)
        #
        return self.LeafString(default=default, typ=typ, choices=choices,
                               question=question, is_optional=optional)

    @staticmethod
    def _parse_typ(typ):
        if "," not in typ:
            return typ, False
        #
        typ, setting = tuple(ele.strip() for ele in typ.split(","))

        if setting == 'optional':
            return typ, True
        raise ValueError(f"Dont understand setting {setting}")

    def generate_cases(self, key, subquestions, block=None):
        """Register `subquestions` at a given `key` in given `block`

        Args:
            key (str): name of the variable that should be overwritten as a subquestion

            subquestions (dict): Dict of Questions corresponding to the subquestions
                                 one wants to register

        Kwargs:
            block (str):  The name of the block, the given `key` is in

        Example:
            >>> _question = "sampling = "
            >>> questions.generate_cases("sampling", {name: sampling.questions for name, sampling
                                                      in cls._sampling_methods.items()})
        """
        #
        subquestions = {name: QuestionASTGenerator(questions)
                        for name, questions in subquestions.items()}
        #
        self.add_branching(key, subquestions, parentnode=block)

    def add_questions_to_block(self, questions, block=None, overwrite=True):
        """add questions to a particular block """
        questions = QuestionASTGenerator(questions)
        self.add_elements(questions, parentnode=block, overwrite=overwrite)

    def generate_block(self, name, questions, block=None):
        """Register `questions` at a given `key` in given `block`

        Args:
            name (str):
                name of the block

            questions (string, tree):
                questions of the block

        Kwargs:
            block (str):  The name of the block, the given `key` is in

        Raises:
            ValueError: If the `key` in `block` already exist it raises an ValueError,
                        blocks can only be new created, and cannot overwrite existing
                        blocks!

        Example:
            >>> _question = "sampling = "
            >>> questions.generate_block("software", {name: software.questions for name, software
                                                      in cls._softwares.items()})
        """
        questions = QuestionASTGenerator(questions)
        self.add_node(name, questions, parentnode=block)

    @classmethod
    def questions_from_file(cls, filename):
        """generate questions from file"""
        with open(filename, "r") as fhandle:
            string = fhandle.read()
        return cls(string)

    def accept(self, visitor, **kwargs):
        return visitor.visit_question_ast_generator(self, **kwargs)

    @staticmethod
    def _parse_default(default):
        """Handle default value"""
        if default in ('NOT_DEFINED', ""):
            return NOT_DEFINED
        return default

    @classmethod
    def _parse_comment(cls, line):
        """Handle Comment section"""
        line, _, comment = line.partition(cls.comment_char)
        if comment == "":
            comment = NOT_DEFINED
        else:
            comment = comment.replace("#n", "\n")
        return line, comment

    @staticmethod
    def _preprocess_string(string):
        """Basic Preprocessor to handle in file comments!"""

        parsed_string = []
        comment_lines = []
        for line in string.splitlines():
            line = line.strip()
            if line == "":
                continue
            if line.startswith('#'):
                comment_lines.append(line[1:])
                continue
            if comment_lines != []:
                line += "###" + "#n".join(comment_lines)
                comment_lines = []
            parsed_string.append(line)
        return "\n".join(parsed_string)

    @staticmethod
    def _parse_choices(line):
        """Handle choices"""
        if line == "":
            return None
        if line is NOT_DEFINED:
            return None
        return line
