__author__ = 'user'

from sympy.combinatorics import Permutation
from task import *
import numpy as np
import time
import operator
from parallel_random import find_minimum_with_exhaustive_search, find_minimum, is_solution

class MyPermutation(Permutation):
    """
    Describes the set of permutations
    """
    def __init__(self, point):
        super(MyPermutation, self).__init__(point)

    def is_solution(self, Task):
        """
        Checks whether the point is the solution for a given constraints system.
        """
        return is_solution(Task.restrictions, self.array_form)

    def all_compound_transpositions(self):
       """
       @return: matrix of compound transpositions for permutation self in lexicographic order
       """
       ct=[]
       for i in range(0, len(self.list())-1):
           for j in range(0, self.cycles):
               if(i in self.full_cyclic_form[j]):icycle=j
               if(i+1 in self.full_cyclic_form[j]):i1cycle=j
           if(icycle != i1cycle): ct.extend([[i, i+1]])
       return ct

    def number_of_all_compound_transpositions(self):
        return len(self.all_compound_transpositions())

    def compound_matrix(self, quiet=True):
        """
        Get the compound matrix of permutation
        rows are the number of cycle in permutation
        the columns - compound transpositions in lexicographic order:
        for permutation Cyclic form:
        [[0, 11, 2, 14], [1, 13, 8, 7, 10, 12, 3], [4], [5, 9, 6]]
        this matrix will be:

            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [8, 9], [9, 10], [10, 11], [11, 12], [13, 14]]
        0      1	    1	    1	    0	    0	    0	    0	    0	     1	        1	      1
        1      1	    1	    1	    1	    0	    1	    1	    1	     1	        1	      1
        2      0	    0	    0	    1	    1	    0	    0	    0	     0	        0	      0
        3      0	    0	    0	    0	    1	    1	    1	    1	     0	        0	      0
        """
        if not quiet:
            print "Number of compount transpositions: ", len(self.compound_transpositions())
            print "Number of cycles: ", self.cycles
        if len(self.all_compound_transpositions()):
            co_m = np.zeros(shape=((self.cycles),len(self.all_compound_transpositions()))).astype('int')
            for i in range(self.cycles):
                for j in range(len(self.all_compound_transpositions())):
                    if ((self.all_compound_transpositions()[j][0] in self.full_cyclic_form[i])or
                            (self.all_compound_transpositions()[j][1] in self.full_cyclic_form[i])):
                        co_m[i,j]=1
            if not quiet:
                print self.all_compound_transpositions()
                for row in co_m:
                    printArray([str(x) for x in row])
        return co_m

    def make_transposition(self, el_for_transpose):
        """
        @param el_for_transpose: list of elements needed to transpose
        For example [5,6]
        @return: new permutation wherein el_for_transpose changed their places
        """
        point = self.array_form
        index1=point.index(el_for_transpose[0])
        index2=point.index(el_for_transpose[1])
        point[index1]=el_for_transpose[1]
        point[index2]=el_for_transpose[0]
        temp_p = MyPermutation(point)
        return temp_p

    def simplest_cycle_permutation(self, quiet=True):
        """
        @param quiet: type False to print on ich step
        @return: cyclic permutation
        permutation is obtained by choosing the first compound transposition at each step
        """
        temp_p_1 = self
        while (temp_p_1.cycles>1):
            if not quiet:
                print "-----------------------------------"
                print "Number of cycles ", temp_p_1.cycles
                print "Cyclic structure: "
                print temp_p_1.cycle_structure
                print "Cyclic form: "
                print temp_p_1.full_cyclic_form
                print "Commit transposition", temp_p_1.compound_transpositions()[0]
                print "-----------------------------------"
            temp_p_1 = temp_p_1.make_transposition(temp_p_1.compound_transpositions()[0])
        return temp_p_1

    def best_cycle_permutation(self, Task):
        """
        @param Task: object class task from task.py
        @return: best cyclic permutation from self.
        permutation is obtained by sorting the compound transposition at each step and choosing the best
        """
        temp_p_2=self
        while (temp_p_2.cycles>1):
            print "-----------------------------------"
            print "Number of cycles ", temp_p_2.cycles
            print "Cyclic structure: "
            print temp_p_2.cycle_structure
            print "Cyclic form: "
            print temp_p_2.full_cyclic_form
            sorted_tr = sort_increas_func(p.compound_transpositions(), Task)
            print "Commit transposition", sorted_tr[0]
            p = temp_p_2.make_transposition(sorted_tr[0])
            print "Func coeff: ", Task.coefficients
            print "New permutation: ", (temp_p_2._array_form, temp_p_2.func_value(Task))
            print "Is solution? ", temp_p_2.is_solution(Task)
            print "-----------------------------------"
        return temp_p_2

    def func_value(self, Task):
        return sum(i * (j+1) for i, j in zip(Task.coefficients, self.array_form))

    def best_compound_transpositions(self, Task):
        """
        len(temp_matr) - number of rows
        len(temp_matr[0]) - number of columns
        temp_matr - temporate matrix of compound
        rows - for numbers of rows to be merged on each step
        all_tr - list of compound transpositions left on each step

        @return:list_tr - list of transpositions needed to get cyclic permutation with min increment of the objective function
        """
        list_tr = []
        temp_matr = self.compound_matrix(quiet=True)
        all_tr = self.all_compound_transpositions()
        while 0 in temp_matr:
            rows=[]
            sorted = self.sort_increas_func(all_tr, Task)
            list_tr.extend([sorted[0]])
            for x in range(len(temp_matr[:,all_tr.index(sorted[0])])):
                if temp_matr[:,all_tr.index(sorted[0])][x]:
                    rows.append(x)
            temp_matr[rows[1]] = temp_matr[rows[0],:]+temp_matr[rows[1],:]
            temp_matr = np.delete(temp_matr, rows[0], 0)
            while 2 in temp_matr:
                all_tr.remove(all_tr[index2d(temp_matr, 2)[1]])
                temp_matr = np.delete(temp_matr, index2d(temp_matr, 2)[1], 1)
        sorted = self.sort_increas_func(all_tr, Task)
        list_tr.extend([sorted[0]])
        return list_tr

    def evaluation(self, Task, transpositions):
        return sum(map(lambda x: (Task.coefficients[self.array_form.index(x[0])]-Task.coefficients[self.array_form.index(x[1])]), transpositions))

    def number_of_generating_permutations(self, Task, transpositions):
        counter = 1
        for i in range(len(transpositions)-1):
            if sorted(transpositions)[i][1] in sorted(transpositions)[i+1]:
                counter *= 2
        return counter

    def sort_increas_func(self, transpositions, Task):
            return sorted(transpositions, key=lambda x: Task.coefficients[self.array_form.index(x[0])]-Task.coefficients[self.array_form.index(x[1])])

"""

Functions

"""

def transpositions_sequences(all_transpositions, perm_numb):
    counter = 1
    all_transpositions = sorted(all_transpositions)
    permutations = [[] for _ in range(perm_numb)]
    for i in range(perm_numb): permutations[i].insert(0, all_transpositions[0])
    for i in range(len(all_transpositions)-1):
        if (all_transpositions[i][0]+1==all_transpositions[i+1][0]):
            for j in range (perm_numb):
                if (j%(2*counter)<counter):
                    index1 = permutations[j].index(all_transpositions[i])
                    permutations[j].pop(index1)
                    permutations[j].insert(index1, all_transpositions[i])
                    permutations[j].insert(index1+1, all_transpositions[i+1])
                else:
                    index1 = permutations[j].index(all_transpositions[i])
                    permutations[j].pop(index1)
                    permutations[j].insert(index1, all_transpositions[i+1])
                    permutations[j].insert(index1+1, all_transpositions[i])
            counter *= 2
        else:
            for j in range (perm_numb):
                permutations[j].insert(i+1, all_transpositions[i+1])
    return permutations

def find_min_of_linear_function(Task):
        """
        Gets the minimum of the linear function on the given set.

        Parameters:
         - coefs - the coefficients (c_i) of the linear function of type F(x) = sum(c_i*x_i)
        """

        # getting the func coefs with the initial indexes
        dict_coefs = dict(enumerate(Task.coefficients))

        # getting indexes. In this case we know which element of set correspond to the given element of the coefs
        keys = sorted(dict_coefs, key=dict_coefs.get, reverse=True)

        # copy generation elements
        res = list(T.generation_elements)

        # take each set elements according the keys.
        for i, j in enumerate(keys):
            res[j] = T.generation_elements[i]

        return res

def printArray(args):
    print "\t".join(args)

def index2d(list2d, value):
    return next((i, j) for i, lst in enumerate(list2d)
                for j, x in enumerate(lst) if x == value)

if __name__ == '__main__':
    avg_number_cycles = 0
    avg_number_compound_tr = 0
    avg_time_to_find_best_compount_tr = 0
    avg_number_cyclic_permutations = 0
    avg_time_solution = 0
    big_counter = 0
    while big_counter < 20:
        T = Task(40)
        new_p = MyPermutation(map(lambda x: x-1, find_min_of_linear_function(T)))
        #print "Permutation min of the linear function: "
        #print new_p.array_form
        #print "Function value: ", new_p.func_value(T)
        #print "Number of cycles ", new_p.cycles
        avg_number_cycles += new_p.cycles
        #print "Cyclic structure: "
        #print new_p.cycle_structure
        #print "Cyclic form: "
        #print new_p.full_cyclic_form
        #print "Is solution: ", new_p.is_solution(T)
        #print "Number of compound transpositions: ", new_p.number_of_all_compound_transpositions()
        avg_number_compound_tr += new_p.number_of_all_compound_transpositions()
        if new_p.cycles > 1:
            start = time.time()
            best_transpositions = new_p.best_compound_transpositions(T)
            finish1 = time.time()
            #print "Best compound transpositions finding time: ", (finish1 - start)
            avg_time_to_find_best_compount_tr += (finish1 - start)
            #print "Theoretical goal function delta: ", new_p.evaluation(T, best_transpositions)
            number_permutations = new_p.number_of_generating_permutations(T, best_transpositions)
            #print "Number of cyclic permutations, that can be generated: ", number_permutations
            avg_number_cyclic_permutations += number_permutations
            if number_permutations>0:
                sequences = transpositions_sequences(best_transpositions, number_permutations)
                #print "Sequences of transpositions to generate cyclic permutations: "
                cyclic_permutation = [[] for _ in range(number_permutations)]
                for i in range(len(sequences)):
                    #print sequences[i], "\t"
                    cyclic_permutation[i] = MyPermutation(new_p.array_form)
                i = 0
                for transpositions in sequences:
                    for elements in transpositions:
                        cyclic_permutation[i] = cyclic_permutation[i].make_transposition(elements)
                    #print "Cyclic permutation: ", cyclic_permutation[i].full_cyclic_form
                    #print "Function value on it: ", cyclic_permutation[i].func_value(T)
                    #print "Real goal function delta: ", cyclic_permutation[i].func_value(T) - new_p.func_value(T)
                    i+=1
            finish = time.time()
            #print "Time of solution: ", (finish - start)
            avg_time_solution += (finish - start)
        big_counter += 1
    print "Average number of cycles: ", operator.truediv(avg_number_cycles, big_counter)
    print "Average number of compound transpositions: ", operator.truediv(avg_number_compound_tr, big_counter)
    print "Average time to find best compound transposition: ", operator.truediv(avg_time_to_find_best_compount_tr, big_counter)
    print "Average number of cyclic permutations :", operator.truediv(avg_number_cyclic_permutations, big_counter)
    print "Average time of solution :", operator.truediv(avg_time_solution, big_counter)
    
