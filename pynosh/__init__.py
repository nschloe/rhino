# -*- coding: utf-8 -*-
#
from . import modelevaluator_nls
from . import modelevaluator_bordering_constant
from . import numerical_methods
from . import preconditioners
from . import magnetic_vector_potentials
from . import yaml

__all__ = [
    "modelevaluator_nls",
    "modelevaluator_bordering_constant",
    "numerical_methods",
    "preconditioners",
    "magnetic_vector_potentials",
    "yaml",
]

__name__ = "pynosh"
__version__ = "0.2.2"
__author__ = "Nico Schlömer"
__author_email__ = "nico.schloemer@gmail.com"
