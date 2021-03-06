# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:29:35 2021

@author: troyf
"""
from gurobipy import *


Taxis = [1,2]# Our taxis t
Passengers = ['A','B','C','D']# Our passengers p
RequestCombos = ['A','B','C','D','AD','BC','BD', 'Unallocated']# Our request combinations rc

T = range(len(Taxis))
P = range(len(Passengers))
RC = range(len(RequestCombos))
RC_real = range(len(RequestCombos)-1) # our ride combos minus the unallocated node
u_node = len(RequestCombos)-1 # index of the unallocated node

max_psg = 2 # maximum passengers per taxi
unallocated_cost = 5 # arbitrary cost of not allocating a passenger

# Our sharability graph costs showing total additional trip time if taxi t 
# is assigned ride combination rc
TripTimes = [
    [999, 2, 1, 999, 999, 4, 999, 0],
    [4, 2, 1, 2, 2, 6, 2, 0]
    ]

# Our sharability graph possible assignments of passenger p to  
# ride combination rc
Combos = [
    [1, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 1, 1]
    ]
m = Model('Taxi Allocations')

X = {(t,rc): m.addVar(vtype=GRB.BINARY) for t in T for rc in RC}
Y = {(p,rc): m.addVar(vtype=GRB.BINARY) for p in P for rc in RC}


m.setObjective(quicksum(X[t,rc]*TripTimes[t][rc] for t in T for rc in RC) +
               quicksum(Y[p,u_node]*unallocated_cost 
                               for p in P), GRB.MINIMIZE)

for t in T:
    # each taxi must take only one "real" request combo
    m.addConstr(quicksum(X[t,rc] for rc in RC_real) == 1)
    # taxis are never allocated to the "unallocated" node
    m.addConstr(X[t,u_node] == 0)

for rc in RC_real:
    # each "real" ride combo can't have more than one taxi allocated
    m.addConstr(quicksum(X[t,rc] for t in T) <= 1)
    # each "real" ride combo can only be used by passengers if  taxi is allocated
    m.addConstr(quicksum(Y[p,rc] for p in P) <= 
                quicksum(X[t,rc] for t in T)*max_psg)   
            

for p in P:
    # each passenger takes only one request combo
    m.addConstr(quicksum(Y[p,rc] for rc in RC) == 1)
    # each passenger can only go on feasible ride combos
    for rc in RC:
        m.addConstr(Y[p,rc] <= Combos[p][rc])

m.optimize()

print("Taxis")
for t in T:
    for rc in RC:
        if X[t,rc].x > 0:
            print(Taxis[t], RequestCombos[rc], [int(X[t,rc].x)])

print("Passengers")
for p in P:
    for rc in RC:
        if Y[p,rc].x > 0:
            print(Passengers[p], RequestCombos[rc], [int(Y[p,rc].x)])

print('Requests Used')
for rc in RC:
    if sum(Y[p,rc].x for p in P) > 0:
        print(RequestCombos[rc], [int(sum(Y[p,rc].x for p in P))])