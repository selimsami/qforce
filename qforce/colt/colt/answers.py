"""Storage for Answers in Colts Questions Module"""
from collections.abc import Mapping


class AnswersBlock(Mapping):
    """Simple Mapping used to store an answers block"""

    def __init__(self, dct=None):
        if dct is None:
            dct = {}
        self._data = dct

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def update(self, dct):
        """update mapping"""
        self._data.update(dct)


class SubquestionsAnswer(Mapping):
    """Storage elemement for the answers of a subquestion"""

    def __init__(self, name, main_answer, subquestion_answers):
        """A subquestion answer object
        Parameters
        ----------

        name: str
            Name of the subquestions (not really necessary)

        main_answer: str
            answer of the main questions, typ needs to be str!

        subquestion_answers: AnswersBlock
            answers of the corresponding `block`

        Note
        ----
        Comparisions are only done using the main_answer, to make the code easier
        """
        self.name = name
        self._main_answer = main_answer
        self._subquestion_answers = subquestion_answers
        if isinstance(self._subquestion_answers, SubquestionsAnswer):
            self.is_subquestion = True
        else:
            self.is_subquestion = False

    def __getitem__(self, key, default=None):
        if self.is_subquestion is True:
            if key == self._subquestion_answers.name:
                return self.subquestion_answers
        return self._subquestion_answers.get(key, default)

    def __iter__(self):
        return iter(self._subquestion_answers)

    def __len__(self):
        return len(self._subquestion_answers)

    def __eq__(self, other):
        """easier comparision

        Examples
        --------

        SubquestionsAnswer('case', 'case1', {}) == 'case1'
        >>> True

        SubquestionsAnswer('case', 'case1', {}) == 'case2'
        >>> False

        """
        if self._main_answer == other:
            return True
        return False

    def __ne__(self, other):
        """easier comparision

        Examples
        --------

        SubquestionsAnswer('case', 'case1', {}) != 'case1'
        >>> False

        SubquestionsAnswer('case', 'case1', {}) != 'case2'
        >>> True

        """
        if self._main_answer != other:
            return True
        return False

    @property
    def subquestion_answers(self):
        """Return answer of subquestions"""
        return self._subquestion_answers

    @property
    def value(self):
        """Return main answer"""
        return self._main_answer

    def __str__(self):
        return ('Subquestions('
                + str({f"{self.name} = {self._main_answer}": self._subquestion_answers}) + ')')

    def __repr__(self):
        return ('Subquestions('
                + str({f"{self.name} = {self._main_answer}": self._subquestion_answers}) + ')')
