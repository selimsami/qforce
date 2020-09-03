"""Provides the `Colt` class to use the questions functionality"""
from abc import ABCMeta
#
from .questions import QuestionASTGenerator
from .ask import AskQuestions
from .commandline import get_config_from_commandline


__all__ = ["Colt"]


def add_defaults_to_dict(clsdict, defaults):
    """ add defaults to dict """
    for key, default in defaults.items():
        if key not in clsdict:
            clsdict[key] = default


def delete_inherited_keys(keys, clsdict):
    """delete any key in clsdict if it's `inherited`"""
    for key in keys:
        if clsdict[key] == 'inherited':
            del clsdict[key]


def join_extend_questions(func1, func2):
    """combine extend question functions"""
    if isinstance(func1, classmethod):
        func1 = func1.__func__
    if isinstance(func2, classmethod):
        func2 = func2.__func__

    def _generate_subquestions(cls, questions):
        func1(questions)
        func2(cls, questions)

    return classmethod(_generate_subquestions)


def to_classmethod(clsdict, function_name):
    """convert the function in the clsdict to a classmethod"""
    if function_name in clsdict:
        func = clsdict[function_name]
        if not isinstance(func, classmethod):
            clsdict[function_name] = classmethod(func)


def colt_modify_class_dict(clsdict, bases):
    """setup the clsdict in colt to avoid inheritance problems

       it modifies both the clsdict and its annotations!
    """
    colt_defaults = {'_extend_questions': classmethod(lambda cls, questions: None),
                     '_questions': "",
                     }

    to_classmethod(clsdict, '_extend_questions')
    to_classmethod(clsdict, 'from_config')
    # rewrite that....it is horrible
    if clsdict.get('__annotations__', None) is not None:
        if clsdict['__annotations__'].get('extend_questions', None) == 'inherited':
            if '_extend_questions' in clsdict:
                if len(bases) > 0:
                    clsdict['_extend_questions'] = join_extend_questions(
                        getattr(bases[0], '_extend_questions'),
                        clsdict['_extend_questions'])
            else:
                clsdict['_extend_questions'] = getattr(bases[0], '_extend_questions')
            # delete task from annotations, and clean unnecessary annotations!
            del clsdict['__annotations__']['extend_questions']
            if clsdict['__annotations__'] == {}:
                del clsdict['__annotations__']
    #
    add_defaults_to_dict(clsdict, colt_defaults)
    delete_inherited_keys(["_questions"], clsdict)


class ColtMeta(ABCMeta):
    """Metaclass to handle hierarchical generation of questions"""

    def __new__(cls, name, bases, clsdict):
        """Modify clsdict before the new method of the metaclass is called"""
        colt_modify_class_dict(clsdict, bases)
        return ABCMeta.__new__(cls, name, bases, clsdict)

    @property
    def questions(cls):
        """return QuestionAST object"""
        return cls._generate_questions()

    def _generate_questions(cls):
        """gentarte QuestionAST object and extend it possibly"""
        questions = QuestionASTGenerator(cls._questions)
        cls._extend_questions(questions)
        return questions

    def _extend_questions(cls, questions):
        """In case additional questions should be added to the QuesionAST"""


class Colt(metaclass=ColtMeta):
    """Base Class for `Colt` classes"""

    @classmethod
    def generate_questions(cls, config=None, presets=None):
        """Generate an object to generate the question config
        either from commandline, or from an input file etc.

        Parameters
        ----------

        config: str, optional
            If not `None` name of a config file, the input should be read

        presets: str, optional
            presets for the questions

        Returns
        -------

        AskQuestions
            object to generate the questions config
        """
        return AskQuestions(cls.questions, config=config, presets=presets)

    @classmethod
    def from_questions(cls, *args, check_only=False, config=None, presets=None, **kwargs):
        """Initizialze the class using `Colt` question utilities

        Parameters
        ----------

        config: str, optional
            Name of a config file

        presets: str, optional
            presets for the questions

        check_only: bool, optional
            If True, check that the settings in the configfile are correct
            If False, ask missing values

        args, kwargs: optional
            arguments and keyword arguments passed to from_config aside from
            the questions config

        Returns
        -------
        PyObj
            anything that from_config returns. Intended to initalize the class
            so from_config should return an instance of the class.

        """
        questions = cls.generate_questions(config=config, presets=presets)
        #
        if check_only is True:
            answers = questions.check_only()
        else:
            if config is None:
                answers = questions.ask()
            else:
                answers = questions.generate_input(config)
        #
        return cls.from_config(answers, *args, **kwargs)

    @classmethod
    def from_config(cls, answer, *args, **kwargs):
        """Initizialze the class using questions config object

        Parameters
        ----------

        answer: obj
            Questions config object

        args, kwargs: optional
            arguments and keyword arguments passed to from_config aside from
            the questions config

        Returns
        -------
        Self
            Intended to initalize the class using the information provided by the config.
        """
        raise Exception("Cannot load from_config, as it is not implemented!, "
                        "also from_questions depend on that!")

    @classmethod
    def from_commandline(cls, *args, description=None, presets=None, **kwargs):
        """Initialize the class using input provided from the commandline

        Parameters
        ----------

        description: str, optional
            Description of the commandline interface, for better documentation,
            see `argparse.ArgumentParser(description)`

        presets: str, optional
            presets for the questions

        args, kwargs: optional
            arguments and keyword arguments passed to from_config aside from
            the questions config

        Returns
        -------
        PyObj
            anything that from_config returns. Intended to initalize the class
            so from_config should return an instance of the class.
        """
        answers = get_config_from_commandline(cls.questions, description=description,
                                              presets=presets)
        return cls.from_config(answers, *args, **kwargs)

    @classmethod
    def generate_input(cls, filename, config=None, presets=None):
        """Generate an inputfile that can later be used to initialze the class

        Parameters
        ----------

        filename: str
            Name of the inputfile

        presets: str, optional
            presets for the questions

        config: str, optional
            Name of a config file, data should be read from

        *args, **kwargs: optional
            arguments and keyword arguments passed to from_config aside from
            the questions config

        Returns
        -------
        AnswerBlock
            colt question obj
        """
        questions = cls.generate_questions(config=config, presets=presets)
        return questions.generate_input(filename)


class CommandlineInterface(Colt):
    """Basic Colt class to handle commandline parsing"""
    _description = None

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Return the config if from_commandline/from_questions is called"""
        return config

    def __init__(self, function):
        """Store the original function"""
        self.function = function
        self.__doc__ = self.function.__doc__

    def __call__(self, *args, **kwargs):
        """If the function is called with arguments: use it as is
        Else: get the arguments from the commandline"""
        # call with arguments
        if any(len(value) != 0 for value in (args, kwargs)):
            return self.function(*args, **kwargs)
        # call from commandline
        answers = self.from_commandline(description=self._description)
        return self.function(**answers)


def from_commandline(questions, *, description=None):
    """Decorate a function to call it using commandline arguments

    Parameters
    ----------
    questions: str
        Questions specifing the user input
    description: str, optional
        Description displayed in case -h is called

    Returns
    -------
    CommandlineInterface
        `Colt` class that acts as the replacement of the function.

        If the function is called without arguments, the argparse is used to get
        the arguments of the function
        Else the function is called normally

    Notes
    -----
    If you decorate a function that takes no arguments, always the commandline parser will
    be called.
    """
    #
    def _wrapper(func):
        """Wrapper function to decorate the function with the CommandlineInterface class"""
        cls = type('CommandlineInterface', (CommandlineInterface,),
                   {'_questions': questions, '_description': description})
        return cls(func)
    # return the new colt class
    return _wrapper
