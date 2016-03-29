from cycle_permutations import CyclePermutationSet
from branchandbound import BranchAndBound
from parallel_random import find_minimum_with_exhaustive_search, find_minimum, is_solution
import time
from task import *
import numpy as np

if __name__ == '__main__':

    T = Task(20)
    CyclePermutationSet = CyclePermutationSet(tuple(T.generation_elements))
    bb = BranchAndBound(T.generation_elements, T.coefficients)
    start3 = time.time()
    bbdaughter = bb.findcyclemin(T.generation_elements)
    finish3 = time.time()
    print "Point and min fuc value found using b&b without restriction: ", (bbdaughter.Adress[1:], bbdaughter.FuncValue)
    print "Time using b&b without restriction: ", (finish3 - start3)
    if is_solution(T.restrictions, bbdaughter.Adress[1:]):
        print "Restricrions are not working"
    else:
        print "Restricrions are working"
    start2 = time.time()
    print "Time started"
    point2, func_value2 = find_minimum(T.coefficients, T.restrictions, CyclePermutationSet, quiet=True)
    finish2 = time.time()
    print "Point and min fuc value found using random search: ", (point2, func_value2)
    print "Time using random search: ", (finish2 - start2)
