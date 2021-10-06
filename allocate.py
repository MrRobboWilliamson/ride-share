# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:29:35 2021

@author: troyf
"""
import networkx
from gurobipy import *

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
                vcosts.append(99999)
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
    
    Vehicle_Trips = []
    for v in V:
        for t in T:
            if X[v,t].x > 0:
                Vehicle_Trips.append((Vehicles[v], Trips[t]))
    
    Request_Trips = []
    for r in R:
        for t in T:
            if Y[r,t].x > 0:
                Request_Trips.append((Requests[r], Trips[t]))
    
    return Vehicle_Trips, Request_Trips