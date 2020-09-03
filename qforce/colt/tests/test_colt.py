#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `colt` package."""

import pytest
import sys
from colt import Colt


@pytest.fixture
def base():
    """generate example for commandline parser"""
    class Base(Colt):
        _questions = """
        nstates = :: int
        natoms = :: int
        factor = 1.0 :: float
        screening = True :: bool
        """

        def __init__(self, nstates, natoms, factor, screening):
            self.nstates = nstates
            self.natoms = natoms
            self.factor = factor
            self.screening = screening

        @classmethod
        def from_config(cls, answers, *args, **kwargs):
            return cls(answers['nstates'], answers['natoms'],
                       answers['factor'], answers['screening'])

    return Base


def test_colt_from_commandline(base):
    """Test ask question"""
    # modify sys.argv
    # test_colt.py 10 231 -factor 2.8
    sys.argv = ['name', '10', '231', '-factor', '2.8']
    cls = base.from_commandline('some example')
    assert cls.nstates == 10
    assert cls.natoms == 231
    assert cls.factor == 2.8
    assert cls.screening is True


def test_colt_from_commandline_base(base):
    """Test ask question"""
    # modify sys.argv
    # test_colt.py 231 10
    sys.argv = ['name', '231', '10']
    cls = base.from_commandline('some example')
    assert cls.nstates == 231
    assert cls.natoms == 10
    assert cls.factor == 1.0
    assert cls.screening is True


def test_colt_question_inheritance(base):

    class Example(base):
        _questions = "inherited"

    sys.argv = ['name', '231', '10']
    cls = Example.from_commandline('some example')
    assert cls.nstates == 231
    assert cls.natoms == 10
    assert cls.factor == 1.0
    assert cls.screening is True
