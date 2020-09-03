# -*- coding: utf-8 -*-

"""Top-level package for Command Line Questions Tool."""

__author__ = """Maximilian Menger"""
__email__ = 'm.f.s.j.menger@rug.nl'
__version__ = '0.1.0'

__all__ = ["Colt", "Plugin", "PluginLoader", "from_commandline", "Validator", "NOT_DEFINED"]

# Helper class to handle easily questions with classes
from .colt import Colt
from .plugins import Plugin
from .pluginloader import PluginLoader
# Decorator to call functions with commandline arguments
from .colt import from_commandline
# Validator
from .validator import Validator, NOT_DEFINED
