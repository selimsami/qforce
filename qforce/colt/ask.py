"""Get User Input from commandline"""
import readline
from contextlib import contextmanager
#
from .qform import QuestionVisitor, QuestionForm


readline.parse_and_bind('tab:complete')


def select_completer(question):
    """generate a completer for a select choices case
    Parameters
    ----------
    question: ConcreteQuestion

    Returns
    -------
    function
        completer function for readline
    """
    def _completer(text, state):
        options = [choice for choice in question.choices if choice.startswith(text)]
        if state < len(options):
            return options[state]
        return None
    return _completer


@contextmanager
def completer(question):
    """Simple context manager for readline completion"""
    readline.set_completer(select_completer(question))
    yield
    readline.set_completer(None)


class CommandlineVisitor(QuestionVisitor):
    """QuestionVisitor to ask questions via the terminal"""

    _helpkeys = (":help", ":h")

    def __init__(self):
        self.ask_all = None

    def on_empty_entry(self, answer, question):
        print("Empty entry not possiple, please enter answer")

    def on_value_error(self, answer, question):
        print(f"Could not parse '{answer}', should be '{question.typ}', redo")

    def on_wrong_choice(self, answer, question):
        print(f"Answer '{answer}' not in {question.choices}!")

    def visit_qform(self, qform, description=None, ask_all=False):
        self.ask_all = ask_all
        if description is not None:
            print(description)
        qform.form.accept(self)

    def visit_question_block(self, block):
        if block.name != '':
            print(f"[{block.name}]")
        self._visit_block(block)

    def visit_concrete_question_select(self, question):
        if self._should_ask_question(question) is True:
            text = self._generate_select_question_text(question)
            if question.has_only_one_choice is True:
                # if only one choice, just print that one
                answer = question.choices[0]
                print(f"{text} {answer}")
                # set the answer
                self.set_answer(question, answer)
            else:
                # ask the question, using the completer
                with completer(question):
                    self._ask_question(text, question, question.comment)

    def visit_concrete_question_input(self, question):
        if self._should_ask_question(question) is True:
            text = self._generate_input_question_text(question)
            self._ask_question(text, question, question.comment)

    def visit_literal_block(self, block):
        pass

    def _generate_input_question_text(self, question):
        txt = self._basic_question_text(question)
        return txt + ": "

    @staticmethod
    def _basic_question_text(question):
        txt = question.label
        # cache it
        answer = question.answer
        if answer is None:
            txt += " [optional]"
        elif answer != "":
            txt += f" [{answer}]"
        return txt

    def _generate_select_question_text(self, question):
        txt = self._basic_question_text(question)
        txt += ", choices = (%s)" % (", ".join(str(opt) for opt in question.choices.as_list()))
        return txt + ": "

    def _should_ask_question(self, question):
        """check weather to ask the question or not"""
        if self.ask_all is True:
            return True
        return not question.is_set

    def _ask_question(self, text, question, comment):
        """Ask the question, and handle events like :h, :help"""
        answer = input(text).strip()  # strip is important!
        if any(answer == helper for helper in self._helpkeys):
            if comment is None:
                print("No help available")
            else:
                print(comment)
            return self._ask_question(text, question, comment)
        #
        if self.set_answer(question, answer) is False:
            return self._ask_question(text, question, comment)
        return True


class AskQuestions(QuestionForm):
    """Questionform to ask questions from the commandline"""

    visitor = CommandlineVisitor()

    def ask(self, description=None, config=None, ask_all=False,
            presets=None, raise_read_error=True):
        """Main routine to get settings from the user,
        if all answers are set, and ask_all is not True

        Parameters
        ----------

        config: str, optional
            name of an existing config file

        ask_all: bool, optional
            whether to ask all questions, or skip those already set

        presets: str, optional
            presets to be used

        Returns
        -------
        AnswerBlock
            user input
        """
        self.set_answers_and_presets(config, presets, raise_error=raise_read_error)
        if ask_all is True:
            return self._ask_impl(ask_all=ask_all, description=description)
        #
        if self.is_all_set:
            return self.get_answers()
        #
        return self._ask_impl(ask_all=ask_all, description=description)

    def check_only(self, config=None, presets=None):
        """Check that all answers set by config are correct and
        return the settings

        Parameters
        ---------
        config: str
            name of an existing config file

        presets: str
            presets to be used

        Returns
        -------
        AnswerBlock
            user input
        """
        self.set_answers_and_presets(config, presets)
        return self.get_answers(check=True)

    def generate_input(self, filename, config=None, presets=None, ask_all=False):
        """Generates an input file from user input
        Parameters
        ----------
        filename, str:
            name of the output file

        config, str:
            name of an existing config file

        ask_all, bool:
            whether to ask all questions, or skip those already set

        presets, str:
            presets to be used

        Returns
        -------
        AnswerBlock
            user input
        """
        #
        self.set_answers_and_presets(config, presets)
        #
        answer = self.ask(presets=presets, ask_all=ask_all)
        self.write_config(filename)
        return answer

    def _ask_impl(self, description=None, ask_all=False):
        """Actuall routine to get settings from the user

        Parameters
        ----------
        config, str:
            name of an existing config file

        ask_all, bool:
            whether to ask all questions, or skip those already set

        presets, str:
            presets to be used

        Returns
        -------
        AnswerBlock
            user input
        """
        self.visitor.visit(self, description=description, ask_all=ask_all)
        #
        return self.get_answers(check=False)
