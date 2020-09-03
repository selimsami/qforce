import argparse
from argparse import Action
#
from .qform import QuestionForm, QuestionVisitor, join_case
from .qform import ValidatorErrorNotInChoices


def get_config_from_commandline(questions, description=None, presets=None):
    """Create the argparser from a given questions object and return the answers

    Parameters
    ----------
    questions: str or QuestionASTGenerator
        questions object to generate commandline arguments from

    description: str, optional
        description used for the argument parser

    presets: str, optional
        presets used for the questions form

    Returns
    -------
    AnswersBlock
        User input
    """
    # Visitor object
    visitor = CommandlineParserVisitor()
    #
    qform = QuestionForm(questions, presets=presets)
    #
    parser = visitor.visit(qform, description=description)
    # parse commandline args
    parser.parse_args()
    #
    return qform.get_answers()


class CommandlineParserVisitor(QuestionVisitor):
    """QuestionVisitor to create Commandline arguments"""

    __slots__ = ('parser', 'block_name')

    def __init__(self):
        """ """
        self.parser = None
        self.block_name = None

    def visit_qform(self, qform, description=None):
        """Create basic argument parser with `description` and RawTextHelpFormatter"""
        parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.RawTextHelpFormatter)
        self.parser = parser
        # visit all forms
        qform.form.accept(self)
        # return the parser
        return parser

    def visit_question_block(self, block):
        """visit all subquestion blocks"""
        for question in block.concrete.values():
            if question.is_subquestion_main is False:
                question.accept(self)
        #
        for subblock in block.blocks.values():
            subblock.accept(self)

    def visit_concrete_question_select(self, question):
        """create a concrete parser and add it to the current parser"""
        if question.has_only_one_choice is True:
            self.set_answer(question, question.choices[0])
        else:
            self.add_concrete_to_parser(question)

    def visit_concrete_question_input(self, question):
        """create a concrete parser and add it to the current parser"""
        self.add_concrete_to_parser(question)

    def visit_literal_block(self, block):
        """do nothing when visiting literal blocks"""

    def visit_subquestion_block(self, block):
        """When visiting subquestion block create subparsers using
           the `SubquestionAction`"""
        # save current parser
        parser = self.parser
        block_name = self.block_name
        #
        comment = self.get_comment(block.main_question)
        # create subparser
        subparser = parser.add_subparsers(action=SubquestionAction, question=block.main_question,
                                          help=f'{comment}')
        for case, subblock in block.cases.items():
            self.block_name = join_case(block.name, case)
            # overwrite parser with new subparser
            self.parser = subparser.add_parser(case)
            subblock.accept(self)
        # restore old parser
        self.parser = parser
        # restore old block_name
        self.block_name = block_name

    def _get_default_and_name(self, question):
        """get the name and default value for the current question"""
        # get id_name
        id_name = question.id
        #
        if self.block_name is not None:
            # remove block_name from the id
            id_name = id_name.replace(self.block_name, '')
            if id_name[:2] == '::':
                id_name = id_name[2:]
        #
        default = question.answer
        #
        if default in ('', None) and not question.is_optional:
            # default does not exist -> Positional Argument
            name = f"{id_name}"
            default = None
        else:
            # default exists -> Optional Argument
            name = f"-{id_name}"
        #
        return default, name

    @staticmethod
    def get_comment(question):
        """get the comment string"""
        choices = question.choices
        if choices is None:
            choices = ''
        #
        comment = f"{question.typ}, {choices}"
        if question.comment is not None:
            comment += f"\n{question.comment}"
        #
        return comment

    def add_concrete_to_parser(self, question):
        """adds a concrete question to the current active parser"""
        default, name = self._get_default_and_name(question)
        #
        comment = self.get_comment(question)
        #
        self.parser.add_argument(name, metavar=question.label, type=_QuestionType(question),
                                 default=default, help=comment)


class SubquestionAction(Action):
    """Create Subparser that reacts to subquestions adopted from argparse._SubParsersAction"""

    def __init__(self, option_strings, prog, parser_class,
                 required=True, help=None, question=None):
        """Initialize new action using questions object"""
        #
        if question is None:
            raise Exception("Need question set for SubquestionAction")
        # actual main question
        self.question = question
        #
        self._prog_prefix = prog
        self._parser_class = parser_class
        #
        self._subquestion_cases = {}
        #
        Action.__init__(self,
                        option_strings=option_strings,
                        dest=argparse.SUPPRESS,
                        nargs=argparse.PARSER,
                        required=required,
                        choices=self._subquestion_cases,
                        help=help,
                        metavar=question.name)

    def add_parser(self, case, **kwargs):
        """Add a parser for a subquestion case

        Parameters
        ----------
        case: str
            name of the subquestion

        Returns
        -------
        parser
            corresponding parser object
        """
        # set prog from the existing prefix
        if kwargs.get('prog') is None:
            kwargs['prog'] = f"{self._prog_prefix} {case}"
        # create the parser and add it to the subquestion cases
        parser = self._parser_class(**kwargs)
        # register case
        self._subquestion_cases[case] = parser
        #
        return parser

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse the commandline and set the corresponding questions"""
        # values contains all commandline arguments starting from the one to be parsed
        # first one is the case
        case = values[0]
        # the others are the argument string for that case
        arg_strings_case = values[1:]

        if self.question.set(case) is True:
            parser = self._subquestion_cases.get(case, None)
        else:
            raise argparse.ArgumentError(self,
                                         f"{case} not in {', '.join(self._subquestion_cases)}")
        # parse the remaining arguments
        subnamespace, arg_strings_case = parser.parse_known_args(arg_strings_case, None)
        # set the values
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)
        # raise exception in case there are unparsed arguments left
        if arg_strings_case:
            raise argparse.ArgumentError(self,
                                         f"Unrecognized Arguments: {', '.join(arg_strings_case)}")


class _QuestionType:
    """Help class to simpulate type validation of the argparse"""

    def __init__(self, question):
        self.question = question
        self.msg = question.typ

    def __str__(self):
        return str(self.msg)

    def __repr__(self):
        return str(self.msg)

    def __call__(self, answer):
        try:
            return self.question.set_answer(answer)
        except ValidatorErrorNotInChoices:
            self.msg = self.question.choices
        raise ValueError(f"Could not set '{answer}'")
