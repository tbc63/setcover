#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Carleton Coffrin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np
import time
import random

""" Determine if solution is viable.  accomplished
    by making sure there is at least 1 non-zero in 
    each column of the reduced adjacency (i.e., adjacency
    only considering rows corresponding to sets taken)"""
def is_viable(item_count, adj, ind_taken):
    for i in range(item_count):
        if len(np.where(adj[ind_taken, i] > 0)[0]) <= 0:
            return False
    return True

""" Determine if certain there is no possible solution
    (column in adjacency with all zeros) or if there
    is a set that must be included in any solution
    (non-sero row in column of adjacency with only 1 non-zero entry)"""
def must_take(cols):
    n = len(cols)
    take = []
    sol = 1
    for i in range(n):
        if len(cols[i]) <= 0:
            take = []
            sol = -i
            return sol, take
        elif len(cols[i]) == 1:
            j = list(cols[i])[0]
            take.append(j)
    return sol, take

""" This is a hybrid greedy algorithm. It utilizes concepts discussed in

     A Complete Solution to the Set Covering Problem
     Qi Yang, Adam Nofsinger, Jamie McPeek, Joel Phinney, Ryan Knuesel 

     specifically, looking to see if any sets used earlier in the algorithm
     have become redundant (i.e., other included sets cover all of the items
     that they cover). Redundant sets are removed from the best_taken list
     and have their costs removed from best_cost. The code also randomly selects
     from a list of sets which have the largest value in the sizes[] array.
     The code also updates the sizes[] array as items are covered so that
     it always reflects the number of uncovered items per unit cost covered
     by a particular row"""
def greedy_lar(item_count, costs, before, candidates, cols, cost, items, taken, init_covered, init_sizes):
    # initialize best_cost & best_taken (the eventual solutions), as well
    # as arrays/sets/lists used and modified by the algorithm. the function 
    # is called multiple times so one needs to make sure anything modified
    # by the function is a local copy.
    best_cost = costs
    best_taken = np.array(taken)
    sizes = np.array(init_sizes)
    covered = np.array(init_covered)
    after = set(before)
    clist = list(candidates)
    i = -1
    take = []
    # continue until all items are covered
    while len(np.where(covered == 0)[0] > 0):
        # sort available sets by decreasing value of their sizes[] array
        sort_ind = np.argsort(-sizes[clist])
        # determine list of sets that share the minimum value of sizes[] array
        k = clist[sort_ind[0]]
        ind = np.where(sizes[clist] == sizes[k])[0]
        # randomly choose a set from this list of candidate sets
        j = clist[random.choice(ind)]
        # remove the chosen set from the list of candidate sets
        clist.remove(j)
        # determine additional items covered by the set that have yet to be covered
        delta = items[j] - after 
        # update the set of items covered thus far
        after |= items[j]
        # if the current best choice increases coverage, add it to the take array,
        # update best_taken to 1 to denote it is taken, update best_cost, and
        # increment covered[] for all items covered by the set.
        if len(delta) > 0:
            i += 1
            take.append(j)
            best_cost += cost[j]
            best_taken[j] = 1
            covered[list(items[j])] += 1
            # for each new item covered by the set, decrement the sizes[] array
            # for non-redundant sets that have not yet been taken using the cols array
            for k in delta:
                for m in cols[k]:
                    if best_taken[m] == 0:
                        sizes[m] -= 1.0/cost[m]
            if i > 0:
                # check to see if any previous row included in the taken sets
                # contains items that all appear in at least 2 sets taken thus far
                # using the covered[] array. if such a previous set is found,
                # remove it from the take array remove its cost from best_cost
                # also set best_cost to -1 to denote it is redundant and decrement
                # covered for all items that the row covers
                rem = []
                for k in range(i):
                    m = take[k]
                    if len(np.where(covered[list(items[m])] <= 1)[0]) == 0:
                        best_taken[m] = -1
                        best_cost -= cost[m]
                        covered[list(items[m])] -= 1
                        rem.append(m)
                if len(rem) > 0:
                    for k in rem:
                        take.remove(k)
                        i -= 1

    # reset best_taken array for rendundant sets to be 0
    ind = np.where(best_taken < 0)[0]
    best_taken[ind] = 0
    # return the best_cost and best_taken array for the greedy_lar algorithm 
    print('Greedy Cost ', best_cost)
    return best_cost, best_taken


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    # read in the number of items that need to be covered
    # and the available sets to cover them
    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    # initialize lists and numpy arrays used in code
    sizes = np.zeros(set_count, dtype = float)
    covered = np.zeros(item_count, dtype = np.int8)
    taken = np.zeros(set_count, dtype = np.int8)
    adj = np.zeros((set_count, item_count), dtype = np.int8)
    cost = np.zeros(set_count, dtype = float)
    items = [set() for _ in range(set_count)]
    cols = [set() for _ in range(item_count)]
    #read in info for each of sets
    for i in range(1, set_count+1):
        parts = lines[i].split()
        # items covered by i-1 set
        ind = [int(i) for i in parts[1:]]
        # assign set cost
        cost[i-1] = float(parts[0])
        # update adjacency of row i-1 to be 1 for the columns of the items it covers
        adj[i-1, ind] = 1
        # items covered by set i-1
        items[i-1] = set(ind)
        # number of items covered by row i-1 per unit cost, look for max value in greedy_lar
        sizes[i-1] = float(len(ind))/cost[i-1]
        # add row i-1 to the cols set for each of the items it covers
        # this is used in greedy_lar to update sizes[] as items become covered
        for j in ind: 
            cols[j].add(i-1)
            
    # determine if certain sets are required to be taken in any solution
    # a row/set must be taken if it is the only row/set that contains
    # (or covers) an item (i.e., row i must be taken if the only non-zero
    # in the jth column of adj occurs in row i) 
    sol, take = must_take(cols)
    # if no possible solution, exit
    if sol < 0:
        print('No coverage for ' + str(-sol))
        return
    print('Set Count & Item Count ', set_count, item_count)
    print('Sets that must be taken ', take)
    # determine which items are covered by sets that must be taken
    # covered[i] = the number of sets included in the total cost that
    # contain or cover item i. for a solution to be valid, covered[i] >= 1
    # for all i in item_count. update initial cost and initial covered
    # set 'before'. also reduce sizes array for columns covered in taken sets
    before = set()
    init_cost = 0
    if len(take) > 0:
        for i in take:
            taken[i] = 1
            init_cost += cost[i]
            delta = items[i] - before
            before |= items[i]
            covered[list(items[i])] += 1
            # for each new item covered by the set, decrement the sizes[] array
            # for non-redundant sets that have not yet been taken using the cols array
            for k in delta:
                for m in cols[k]:
                    if taken[m] == 0:
                        sizes[m] -= 1.0/cost[m]

           
    # determine list of items not yet covered
    uncovered = np.where(covered == 0)[0]

#
#   THIS IS CODE TO COMPRESS ROWS THAT HAVE EXACTLY THE
#   SAME ADJACENCY. IT HASN'T BEEN DEBUGGED SINCE NO TEST
#   EXAMPLES SATISFIED THIS CONDITION. ONE COULD TRY TO
#   INCORPORATE INTO greedy_lar FUNCTION, BUT IT IS NOT
#   CLEAR IF IT WOULD IMPROVE PERFORMANCE
#
#    uniq, ind, inv, cnt = np.unique(adj, axis=0, 
#                                    return_index=True,
#                                    return_inverse=True,
#                                    return_counts=True)
#    print(uniq.shape[0],adj.shape[0])
#    if uniq.shape[0] < adj.shape[0]:
#        for i in range(len(ind)):
#            if cnt[i] > 1:
#                ind0 = np.where(inv == i)
#                j =ind0[np.amin(cost[ind0])]
#                taken[ind0] = -1
#                taken[j] = 0
            
    
    # determine set of candidate set items that cover these uncovered items
    # this can prune search space if len(take) > 0. these are rows/sets not
    # required to be taken initially. they are the starting uncovered set
    # for the greedy_lar function
    if (len(uncovered) == 0):
        print('Solution found with required stations')
        return
    else:
        candidates = []
        for i in uncovered:
            ind = set(np.nonzero(adj[:,i])[0])
            for j in ind:
                if taken[j] == 0:
                    if j not in candidates:
                        candidates.append(j)
    candidates = np.array(candidates)       
    
#
#   THIS CODE ONLY REQUIRED IF ONE USES THE np.unique CODE
#   COMMENTED OUT ABOVE
#
#    ind = np.where(taken < 0)[0]
#    taken[ind] = 0
    
        
    # initialize best solution and number of times greedy_lar will be 
    # executed. greedy_lar contains a random selection of possible
    # candidate sets, so multiple runs produce different answers. the
    # code reports the answer with the lowest cost.
    best_cost = 1.0e9
    best_taken = []
    if set_count < 100:
        num = set_count
    else:
        num = 100
    t1 = 1.0e6 * time.time()
    for i in range(num):
       tmp_cost, tmp_taken = greedy_lar(item_count, init_cost, before, candidates, cols, cost, items, taken, covered, sizes)
       if tmp_cost < best_cost:
           best_cost = tmp_cost
           best_taken = np.array(tmp_taken)
    t2 = 1.0e6 * time.time()
    print('Is Solution Viable = ', is_viable(item_count, adj, np.where(best_taken > 0)[0]))    
    print('Cost of Best Solution = ', best_cost)
    print('Total Time (sec) for ' + str(num) + ' Cycles of Modified Greedy LAR Algorithm', 1.0e-6 * (t2 - t1))
    
    # convert results into format needed for submission
    obj = best_cost
    solution = list(best_taken)
    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')

