# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:29:35 2021

@author: troyf
"""
import networkx
from gurobipy import *



def create_ILP_data_v2(rtv_graph):
    
    rtv_nodes = rtv_graph.nodes
    V = [n for n in rtv_nodes if type(n) == str]# Our Vehicles v
    R = [n for n in rtv_nodes if type(n) != tuple and type(n) != str]# Our Requests r
    T = [n for n in rtv_nodes if type(n) == tuple]# Our Trips t
    
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
    TR = {}
    # Our sharability graph of possible assignments of trips t to either 
    # vehicles v or requests r
    for t in T:
        vehicles = []
        requests= []
        for n in rtv_graph.neighbors(t):
            if type(n) == str:
                vehicles.append(n)
            else:
                requests.append(n)
        TV[t]=vehicles
        TR[t]=requests
    
        
    
    return V, R, T, VT, RT, TV, TR


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
            greedy_dict[(u,v)] =[d['wait']+d['delay'], d['rnum'], 0]    
    
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


def allocate_trips_v2(V, R, T, VT, RT, TV, TR, suppress_output=False):    

    max_psg = 2 # maximum Requests per taxi
    unallocated_cost = 99999 # arbitrary cost of not allocating a passenger
    
    m = Model('Allocations')
    
    if suppress_output:
        m.setParam('OutputFlag',0)
    
    # X is 1 if vehicle v is assigned to trip t, 0 otherwise
    X = {(v,t): m.addVar(vtype=GRB.BINARY) for v in V 
                                 for t in VT[v].keys()}
     # Y is 1 if request r is served by trip t, 0 otherwise   
    Y = {(r,t): m.addVar(vtype=GRB.BINARY) for r in R 
                                 for t in RT[r]}
    # Z is 1 if request r is unallocated, 0 otherwise
    Z = {(r): m.addVar(vtype=GRB.BINARY) for r in R}
    
    # greedy allocation of starting position for the X variables
    
    
    # the objective function minimizes the wait and delay time of each 
    # vehicle-trip assignment plus the high cost of not servicing a request
    m.setObjective(quicksum(X[v,t]*VT[v][t][0] for v in V for t in VT[v].keys())
                           + quicksum(Z[r] * unallocated_cost for r in R)
                           , GRB.MINIMIZE)
    
    
    
    
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
        # each trip can only service a request if a vehicle has been assigned
        m.addConstr(quicksum(Y[r,t] for r in TR[t]) <= 
            quicksum(X[v,t] for v in TV[t])*max_psg)   
            
    for r in R:
        # each request must either be serviced by a trip or be unallocated
        m.addConstr(quicksum(Y[r,t] for t in RT[r]) + Z[r] == 1)

    m.optimize()

    # get all of the assigned vehicle-trips combinations    
    Vehicle_Trips = []
    for v in V:
        for t in VT[v].keys():
            if X[v,t].x > 0:
                
                # if len(t) > 1:
                #     print((v,t))
                
                Vehicle_Trips.append((v, t))
    
    # get all of the assigned request-trip combinations
    Request_Trips = []
    for r in R:
        if Z[r].x > 0:
            Request_Trips.append((r, 'unallocated'))
        else:
            for t in RT[r]:
                if Y[r,t].x > 0:
                    Request_Trips.append((r, t))
    
    return Vehicle_Trips, Request_Trips


'''
Original attempts at the allocation functions
'''
def create_ILP_data(rtv_graph):
    
    rtv_nodes = rtv_graph.nodes
    Vehicles = [n for n in rtv_nodes if type(n) == str]# Our Vehicles v
    Requests = [n for n in rtv_nodes if type(n) != tuple and type(n) != str]# Our Requests r
    Trips = [n for n in rtv_nodes if type(n) == tuple]# Our Trips t
    Trips.append('unallocated')
    
    rtv_edges = rtv_graph.edges(data=True)
    TripCosts = []  
    # Our sharability graph costs showing total additional trip time if taxi t 
    # is assigned ride combination rc
    for v in Vehicles:
        vcosts = []
        for t in Trips:
             try:
                 vcosts.append(rtv_graph.get_edge_data(v,t)['wait'] +
                               rtv_graph.get_edge_data(v,t)['delay'])
             except TypeError:
                vcosts.append(999999)
        TripCosts.append(vcosts)


    RequestTrips = []
    # Our sharability graph possible assignments of requests r to  
    # trips t
    for r in Requests:
        rcosts = []
        for t in Trips:
            if rtv_graph.has_edge(r,t) or t == 'unallocated':
                rcosts.append(1)
            else:
                rcosts.append(0)
        RequestTrips.append(rcosts)

    return Vehicles, Requests, Trips, TripCosts, RequestTrips


def allocate_trips(Vehicles, Requests, Trips, TripCosts, RequestTrips):
    
    V = range(len(Vehicles))
    R = range(len(Requests))
    T = range(len(Trips))
    T_real = range(len(Trips)-1) # our Trips minus the unallocated node
    u_node = len(Trips)-1 # index of the unallocated node
    
    max_psg = 2 # maximum Requests per taxi
    unallocated_cost = 99999 # arbitrary cost of not allocating a passenger
    
    m = Model('Allocations')
    
    X = {(v,t): m.addVar(vtype=GRB.BINARY) for v in V for t in T}
    Y = {(r,t): m.addVar(vtype=GRB.BINARY) for r in R for t in T}
    
    
    m.setObjective(quicksum(X[v,t]*TripCosts[v][t] for v in V for t in T) +
                   quicksum(Y[r,u_node]*unallocated_cost 
                                   for r in R), GRB.MINIMIZE)
    
    for v in V:
        # each vehicle must take up to only one "real" trip
        m.addConstr(quicksum(X[v,t] for t in T_real) <= 1)

    
    for t in T_real:
        # each "real" trip can't have more than one vehicle allocated
        m.addConstr(quicksum(X[v,t] for v in V) <= 1)
        # each "real" trip can only be used by Requests if a vehicle is allocated
        m.addConstr(quicksum(Y[r,t] for r in R) <= 
                    quicksum(X[v,t] for v in V)*max_psg)   
                
    
    for r in R:
        # each request must take exactly one trip even if it's the unallocated
        m.addConstr(quicksum(Y[r,t] for t in T) == 1)
        # each request can only go on a trip they belong to
        for t in T:
            m.addConstr(Y[r,t] <= RequestTrips[r][t])
    
    m.optimize()
    
    Trips = dict()
    for t in T:
        for v in V:
            if X[v,t].x > 0.9:
                Trips[t] = v
    
    return Trips
    
    # Vehicle_Trips = []
    # for v in V:
    #     for t in T:
    #         if X[v,t].x > 0:
    #             Vehicle_Trips.append((Vehicles[v], Trips[t]))
    
    # Request_Trips = []
    # for r in R:
    #     for t in T:
    #         if Y[r,t].x > 0:
    #             Request_Trips.append((Requests[r], Trips[t]))
    
    # return Vehicle_Trips, Request_Trips 