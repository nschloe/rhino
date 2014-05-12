# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of pynosh.
#
#  pynosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  pynosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with pynosh.  If not, see <http://www.gnu.org/licenses/>.
#
import os
from distutils.core import setup
import codecs


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname),
                       encoding='utf-8'
                       ).read()

setup(name='pynosh',
      packages=['pynosh'],
      version='0.1.0',
      description='Nonlinear Schrödinger equations',
      long_description=read('README.md'),
      author='Nico Schlömer',
      author_email='nico.schloemer@gmail.com',
      url='https://github.com/nschloe/pynosh/',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )
