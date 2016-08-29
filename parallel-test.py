#Main window for solving

from permutations_by_transpositions import CyclePermutationByTransposition
from cycle_permutations import CyclePermutationSet
from branchandbound import BranchAndBound
from randomsearch import find_minimum_with_exhaustive_search, find_minimum, is_solution
import time
from task import *
import operator
import numpy as np


if __name__ == '__main__':

    task_dimension = 5
    error_counter = 0
    avg_time = 0
    avg_error1 = 0
    avg_error2 = 0
    print "Task dimention : ", task_dimension
    for i in xrange(10):
        print "Task in one dimension number : ", i
        T = Task(task_dimension)
        CyclePermTr = CyclePermutationByTransposition(T.generation_elements)
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
        point2, func_value2 = find_minimum(tuple(T.coefficients), T.restrictions, CyclePermTr, quiet=True)
        finish2 = time.time()
        print "Point and min fuc value found using random search + transpositions: ", (point2, func_value2)
        print "Time using random search: ", (finish2 - start2)
        avg_time += (finish2 - start2)
        if ((bbdaughter.FuncValue != None) and (func_value2 != None)):
            avg_error1 += abs(operator.truediv((func_value2 - bbdaughter.FuncValue), bbdaughter.FuncValue))
            avg_error2 += abs(operator.truediv((func_value2 - bbdaughter.FuncValue), func_value2))
            error_counter += 1
    print "Average time of solution in dimension: ", operator.truediv(avg_time, 10)
    print "Average error1 of solution in dimension: ", operator.truediv(avg_error1, error_counter)
    print "Average error2 of solution in dimension: ", operator.truediv(avg_error2, error_counter)
