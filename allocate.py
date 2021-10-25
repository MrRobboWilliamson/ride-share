# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:29:35 2021

@author: troyf
"""
from gurobipy import Model,GRB,quicksum
import numpy as np

def create_ILP_data_v2(rtv_graph):
    
    rtv_nodes = rtv_graph.nodes
    V = [n for n in rtv_nodes if isinstance(n,str)] # Our Vehicles v
    R = [n for n in rtv_nodes if isinstance(n,np.int64)] # Our Requests r
    T = [n for n in rtv_nodes if isinstance(n,tuple)]# Our Trips t
    
    VT = {} 
    # Our sharability graph of possible assignments of vehicle v to trip t 
    for v in V:
        trips = {}
        for t in rtv_graph.neighbors(v):
            cost =  rtv_graph.get_edge_data(v,t)['wait'] + \
                               rtv_graph.get_edge_data(v,t)['delay']
            try:
                greedy = rtv_graph.get_edge_data(v,t)['greedy']
            except KeyError:
                greedy = 0
            trips[t] = (cost, greedy)
        VT[v]=trips

    RT = {}
    # Our sharability graph of possible assignments of requests r to  
    # trips t
    for r in R:
        trips = []
        for t in rtv_graph.neighbors(r):
            trips.append(t)
        RT[r]=trips
        
    TV = {}
    # Our sharability graph of possible assignments of trips t to either 
    # vehicles v or requests r
    for t in T:
        vehicles = []
        for n in rtv_graph.neighbors(t):
            if type(n) == str:
                vehicles.append(n)

        TV[t]=vehicles 
    
    return V, R, T, VT, RT, TV


def greedy_assignment(rtv_graph, k):
    
    greedy_dict = {}  
    rtv_edges = rtv_graph.edges(data=True)
    rtv_nodes = rtv_graph.nodes
    V = [n for n in rtv_nodes if type(n) == str]# Our Vehicles v
    R = [n for n in rtv_nodes if type(n) != tuple and type(n) != str]# Our Requests r
    
    # create a dictionary of the edge data that can be used in dictionary
    # comprehension
    for u,v,d in rtv_edges:
        if d['edge_type']=='tv':
            greedy_dict[(u,v)] = [d['wait']+d['delay'], d['rnum'], 0]    
    
    # we work down from max capacity to single request trips
    while k > 0:
        # get all of the trips of the size of our current batch
        k_dict = {vt:d for (vt,d) in greedy_dict.items() if d[1]==k}
        
        # put that batch in a list and sort it by cost, cheapest first
        k_list = list(k_dict.items())
        k_list.sort(key=lambda vt:vt[1][0])
        
        # for vehicle-trip combo in this batch, cheapest to most expensive, 
        # check if we've used the vehicle and any of the requests yet and if 
        # all of those are present, we'll assign this combo as a starting 
        # assignment
        for vt in k_list:
            if type(vt[0][0]) == str:
                vpos = 0
                tpos = 1
            else:
                vpos = 1
                tpos = 0
            if vt[0][vpos] in V:
                rcheck = 0
                for r in vt[0][tpos]:
                    if r in R:
                        rcheck +=1
                if rcheck == k:
                    rtv_graph[vt[0][0]][vt[0][1]]['greedy'] = 1
                    V.remove(vt[0][vpos])
                    for r in vt[0][tpos]:
                        R.remove(r)   
        k -= 1
    
    return rtv_graph


def allocate_trips_v2(V, R, T, VT, RT, TV, suppress_output=False):    


    unallocated_cost = 99999 # arbitrary cost of not allocating a passenger
    
    m = Model('Allocations')
    
    if suppress_output:
        m.setParam('OutputFlag',0)
    
    # X is 1 if vehicle v is assigned to trip t, 0 otherwise
    X = {(v,t): m.addVar(vtype=GRB.BINARY) for v in V 
                                 for t in VT[v].keys()}
    
    # Z is 1 if request r is unallocated, 0 otherwise
    Z = {(r): m.addVar(vtype=GRB.BINARY) for r in R}
    
    # the objective function minimizes the wait and delay time of each 
    # vehicle-trip assignment plus the high cost of not servicing a request
    m.setObjective(quicksum(X[v,t]*VT[v][t][0] for v in V for t in VT[v].keys())
                           + quicksum(Z[r] * unallocated_cost for r in R)
                           ,GRB.MINIMIZE)
    
    for v in V:
        # each vehicle must take up to only one "real" trip
        m.addConstr(quicksum(X[v,t] for t in VT[v].keys()) <= 1)
        # assign each X variable a starting value of 0 or 1 based on our
        # greedy algorithm assignment
        for t in VT[v].keys():
            X[v,t].setAttr('Start', VT[v][t][1])
    
    for t in T:
        # each trip can't have more than one vehicle allocated
        m.addConstr(quicksum(X[v,t] for v in TV[t]) <= 1)
    
    for r in R:
        # each request must either be serviced by a trip or be unallocated
        m.addConstr(quicksum(X[v,t] for t in RT[r] for v in TV[t])
                                    + Z[r] == 1)
    
    m.optimize()
    
    Trips = dict()
    for v in V:
        for t in VT[v]:
            if X[v,t].x > 0.9:
                Trips[t] = v
    
    return Trips


def rebalance(V,unallocated,times,Taxis,suppress_output=False):    

    m = Model('Rebalance')
    
    R = unallocated.index
    
    # get all of our journey times for all of our vehicle-request combos
    jt = {}
    for v in V:
        jt[v] = {}
        for r in R:
            jt[v][r] = times[Taxis[v].loc,unallocated.loc[r,'from_node']]
    
    if suppress_output:
        m.setParam('OutputFlag',0)
    
    # Y is 1 if vehicle v is assigned to request r, 0 otherwise
    Y = {(v,r): m.addVar(vtype=GRB.BINARY) for v in V 
                                 for r in R}
    # create a variable for the minimum value of vehicles or requests
    if len(V) > len(R):
        minVR = {('a'): m.addVar(lb = len(R), ub = len(R))}
    else:
        minVR = {('a'): m.addVar(lb = len(V), ub = len(V))}
   
    # the objective function minimizes the travel time of idle vehicles to 
    # get to unallocated request origin nodes
    m.setObjective(quicksum(Y[v,r]*jt[v][r] for v in V for r in R)
                                                     , GRB.MINIMIZE)

    for v in V:
        # each vehicle must be allocated only up to one request
        m.addConstr(quicksum(Y[v,r] for r in R) <= 1)
    
    for r in R:
        # each request must allocated only up to one vehicle
        m.addConstr(quicksum(Y[v,r] for v in V) <= 1)
    
    # the number of rebalanced vehicle-request combos must be equal to the 
    # minimum of number of idle vehicles or number of unallocated requests
    m.addConstr(quicksum(Y[v,r] for v in V for r in R) == minVR['a'])
    
    m.optimize()
    
    Rebalanced = dict()
    for v in V:
        for r in R:
            if Y[v,r].x > 0.9:
                # allocate a path to the vehicle
                # a path is [(request_id,node,is_pickup),...]
                Rebalanced[v] = [(r,unallocated['from_node'].values[0],True),
                                 (r,unallocated['to_node'].values[0],False)]
    
    # return a dictionary of vehicles with their new destination node
    return Rebalanced