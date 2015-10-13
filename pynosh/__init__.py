# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyNosh.
#
#  PyNosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyNosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyNosh.  If not, see <http://www.gnu.org/licenses/>.
#
from . import modelevaluator_nls
from . import modelevaluator_bordering_constant
from . import numerical_methods
from . import preconditioners
from . import magnetic_vector_potentials
from . import yaml

__all__ = [
    'modelevaluator_nls',
    'modelevaluator_bordering_constant',
    'numerical_methods',
    'preconditioners',
    'magnetic_vector_potentials',
    'yaml'
    ]

__name__ = 'pynosh'
__version__ = '0.2.1'
__author__ = 'Nico Schlömer'
__author_email__ = 'nico.schloemer@gmail.com'
