from cycle_permutations import CyclePermutationSet
from branchandbound import BranchAndBound
#from parallel_random import find_minimum_with_exhaustive_search, find_minimum, is_solution
import time
import numpy as np

if __name__ == '__main__':

    print "Get started"
    n=10
    s1 = list(np.random.randint(5, size=n)-np.random.randint(8))
    s2 = list(np.random.randint(5, size=n)-np.random.randint(8))
    s1.append(0)
    s2.append(0)
    s = [s1, s2]
    print "S=", s
    co = list(np.random.randint(100, size=n))
    print "Coef=", co
    el=[]
    el.extend(range(1, n+1))
    print "Elem=", el
    func = tuple(co)
    print "Func=", func
    pset = CyclePermutationSet(tuple(el))
    print "PSet", pset.generation_elements
    bb = BranchAndBound(el, co)
    bb.
    #start3 = time.time()
    #bbdaughter = bb.findcyclemin(el)
    #finish3 = time.time()
    #print "Point and min fuc value found using b&b without restriction: ", (bbdaughter.Adress[1:], bbdaughter.FuncValue)
    #print "Time using b&b without restriction: ", (finish3 - start3)

    #if is_solution(s, bbdaughter.Adress[1:]):
        #print "Restricrions are not working"
    #else:
        #print "Restricrions are working"
    #start2 = time.time()
    #point2, func_value2 = find_minimum(func, s, pset, quiet=True)
    #finish2 = time.time()
    #print "Point and min fuc value found using random search: ", (point2, func_value2)
    #print "Time using random search: ", (finish2 - start2)
