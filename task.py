__author__ = 'user'

import numpy as np

class Task(object):

    def __init__(self, n):
        s1 = list(np.random.randint(5, size=n)-np.random.randint(7))
        s2 = list(np.random.randint(5, size=n)-np.random.randint(7))
        s1.append(0)
        s2.append(0)
        self.restrictions = [s1, s2]
        self.generation_elements = range(1, n+1)
        self.coefficients = list(np.random.randint(200, size=n))

if __name__ == '__main__':
    T = Task(15)
    print T.coefficients
    print T.generation_elements
    print T.restrictions
