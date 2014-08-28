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
import numpy as np


def _main():
    a = np.ones((1, 3))
    b = np.ones((2, 1))
    print a - b  # ???
    # note by Andre: this is somewhat 'more' consistent than MATLAB!
    # In MATLAB you can do ones(1,3)+1.0 but not ones(1,3)+ones(2,1).

    a = np.ones((3, 1))
    b = np.ones((3,))
    print a * b
    print b * a
    print np.multiply(a, b)

    #a = np.array([[1,2], [3,4]])
    #x = np.array([[1],[1]])
    #y = np.array([[2],[2]])
    #print np.dot(a, x[:,0]) + np.dot(a, y[:,0])
    #print np.dot(a, x[:,[0]]) + np.dot(a, y[:,[0]])


if __name__ == '__main__':
    _main()
