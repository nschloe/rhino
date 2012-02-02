# -*- coding: utf-8 -*-
import numpy as np
# ==============================================================================
def _main():
    a = np.ones((1,3))
    b = np.ones((2,1))
    print a - b # ???
    # note by Andre: this is somewhat 'more' consistent than MATLAB!
    # In MATLAB you can do ones(1,3)+1.0 but not ones(1,3)+ones(2,1).

    #a = np.array([[1,2], [3,4]])
    #x = np.array([[1],[1]])
    #y = np.array([[2],[2]])
    #print np.dot(a, x[:,0]) + np.dot(a, y[:,0])
    #print np.dot(a, x[:,[0]]) + np.dot(a, y[:,[0]])
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
