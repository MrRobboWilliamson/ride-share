# -*- coding: utf-8 -*-
"""
This is the model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import networkx as nx
from numbers import Number
import itertools

# our modules
from source_data import JourneyTimes, Requests
from taxis import Taxi, Passenger
from utils import assign_basejt, plot_requests, shortest_path, is_cost_error \
    , check_two_req, add_one_req, check_one_req_one_passenger

# cost error
class CostError(Exception):
    pass

np.random.seed(16)

# Shareability parameters
# vehicle capacity (k) and quality of service threshold (delta)
# M is the number of vehicles
k = 2
delta = 30
MaxWait = 120 # Two minutes
MaxQLoss = 2*MaxWait # Loss of quality of service (wait and in-vehicle delay)
M = 3000
Period = 7*24*60*60 # one week of operations
# nWindows = int(Period/delta)
nHours = int(Period/3600)
nDays = int(Period/(24*3600))
nIntersections = 4091
inters = np.arange(nIntersections)
day_trans = {0:0,1:1,2:1,3:1,4:1,5:1,6:2}

### Sets
# W = range(nWindows)
V = range(M)
H = range(nHours)
D = range(nDays)

### Our datasets will be the trip data
# I have downloaded the May 2013 (not 2015) data and 
# filtered for the week starting Sunday 5 May to end Saturday May 11
# filtered for requests within 100m of an intersection
# then did a nearest neighbour analysis to get the closest intersection to
# each request
# requests that start and end at the same node are also removed
rdf = Requests(r'datasets',delta,MaxWait).read_requests()

# get drop off locations from the "previous" saturday night
# to sample taxi locations at the drop off locations
sat_night = rdf[rdf['time']>Period-3600]
init_locs = sat_night['to_node'].values
np.random.shuffle(init_locs)
init_locs = init_locs[:M]

# print("Sample of initial location for the taxis to start")
# print(init_locs.shape)
# print(init_locs[:10])

### Journey time data
# this has been precomputed for every od combination of intersections 
# thanks Giovanni!!
# I have created a journey time class to retrieve the journey times
# for a given day and time (days are 0, 1, 2 for Sun, Weekday, Sat)
# these will help us calculate sharability
# as adding intermediate points will split the journey times
jt = JourneyTimes(r'datasets/journey_times')
    
# need to know what hour day we're in for a given
# seconds value

### Taxis
# we will also need to simulate taxis
# Michael used something called Hqueue or something to do this
# it should be in one of the python files on Blackboard
# I suggest we inject them at randomly sampled drop off points
Taxis = { f"v{v}": Taxi(f"v{v}",k,init_locs[v]) for v in V }
    
def build_shareable(t,requests,times,rv_graph,visualise=False):        
    """
    Searches each feasible pair of requests for the shortest path
    between them  
    
    TODO: future improvement - generate optional routes as weighted
    graphs and find shortest path using networkx feature. Instead of
    my hacky function.    
    """
    
    orequests = requests.copy()
    for r1,t1,o1,d1,ltp1,bjt1,qos1,_ in requests.to_records():         
        
        # pop this request off the list
        orequests = orequests.drop(r1)
        
        # only check feasible pairs
        jt12 = times[o1,:]
        jt21 = times[:,o1]
        to_check = orequests[
            (jt12[orequests['from_node']]+orequests['qos']<=MaxWait) |
            (jt21[orequests['from_node']]+qos1<=MaxWait)
            ]
                
        for r2,t2,o2,d2,ltp2,bjt2,qos2,_ in to_check.to_records():
            
            # get the shortest path to satisfy a pair of requests
            pair = dict([(r1,dict(route=(o1,d1),other=r2,wait=qos1,base=bjt1)),
                         (r2,dict(route=(o2,d2),other=r1,wait=qos2,base=bjt2))])
            
            # only assess once
            if not rv_graph.has_edge(r1,r2):
                
                cost,path = shortest_path(pair,times,MaxWait)
                                
                if path:             
                    # the key of the 'rr' graph is in the order of the 
                    # best to pickup first 
                    
                    #### check the cost and path ####
                    # check cost against pair
                    # if is_cost_error(cost,path,pair,times,show=True):
                    #     raise CostError("found one!")
                    
                    rv_graph.add_edge(r1,r2,cost=cost,path=path)
                    
                    # update requests df if there is a match
                    requests.loc[r1,'is_alone'] = False
                    requests.loc[r2,'is_alone'] = False
            else:
                # update requests df if there is a match
                requests.loc[r1,'is_alone'] = False
                requests.loc[r2,'is_alone'] = False
                
                # if we've already assessed this pair
                # we just need to update the costs with the 
                # additional waiting time
                rv_graph.edges[r1,r2]['cost'] += 2*delta
                
    return requests,rv_graph
            

def update_rv(t,requests,times,rv_graph,visualize=False):
    """
    Updates the RV graph
    
    Step 1:
        - update the pairwise shareability graph
        
    Step 2:
        - check which vehicles can be assigned to which requests
        - need to be able to assign requests to cabs with one availabe
          seat en route    
    """
        
    # this is the initial wait time
    requests['qos'] = t - requests['time'].values
    requests['is_alone'] = True
    
    # returns a mask of the lonely requests
    start = time.process_time()    
    requests,rv_graph = build_shareable(t,requests,times,rv_graph)
    end = time.process_time()
    print(f"  - build_shareable function {end-start:0.1f}s")
    
    if visualize:
        ### THIS WON'T WORK - NEEDS TO BE UPDATED ###
        # randomly get a request that is shareable
        r = list(np.random.choice(list(rv_graph.keys())))[0]
        
        # get all of the other requests
        to_plot = set()
        for pair in rv_graph['rr']:
            if r in pair:
                to_plot |= pair 
                
        print(r,to_plot)
        print(requests[requests.index.isin(to_plot)])
        plot_requests(t,requests[requests.index.isin(to_plot)],MaxWait,r)
    
    # Now we need to check which taxis can service which
    # requests
    for v in Taxis:
        
        if len(Taxis[v].passengers) < k:
            
            # get the requests that are serviceable
            cab = Taxis[v]
            jtfromme = times[cab.loc,:]
            
            # this cab can get to these requests within their max wait time 
            potentials = requests[requests['qos']+jtfromme[requests['from_node']]<=MaxWait]
            
            # if this cab can't service any requests, then skip to the next
            # cab - else update the rv_graph
            if potentials.empty:
                continue
            else:                                    
                # assign edges between cab and requests with the journey
                # time as the value - don't assign the initial qos twice - 
                # it is assigned in the shortest path function
                for i,delay in enumerate(jtfromme[potentials['from_node']]):
                    rv_graph.add_edge(potentials.index[i],v,cost=delay)
                    
    return rv_graph,requests

def create_rtv_graph(rv,requests):
    """
    Explore the cliques of the rv graph to find feasible trips
    """
    rtv_graph = nx.Graph()

        ### If you want to visualise the cliques ###
        # if len(clique) > 4:
            
        #     # color the cabs orange
        #     print(clique)
        #     subg = rv.subgraph(clique)
        #     fig,ax = plt.subplots(figsize=(10,6))
        #     colors = ['tab:orange' if type(n) == str else 'tab:blue' for n in list(subg.nodes)]    
        #     nx.draw(subg,node_size=30,with_labels=False,node_color=colors,
        #             alpha=0.5,width=1,ax=ax)
        
        #     break

### Evaluation
# create rv graph
# create rtv graph
# assign vehicles to trips
# requests that can't be satisfied stay in the request pool
# def shareable(requests):
#     """
#     Get shareable rides
#     """
    
#     # for each request, find other requests that are after it
#     for r in requests:
    
#     pass

# Cycle in hour chunks, so we don't have to check to load
# new journey times each iteration we only load them on the hour
rv_graph = nx.Graph()
active_requests = pd.DataFrame()
for d in D:
    
    # convert to sun, weekday, sat
    dt = day_trans[d]
    
    for h in H:
        
        # get the journey times matrix
        times = jt.get_hourofday(dt,h)
        
        # get all of the requests - we might want to preprocess these?
        req_slice = assign_basejt(rdf[(rdf['day']==d)&(rdf['hour']==h)],
                                  times)
        req_byw = req_slice.groupby('window')
        
        # get new requests in 30s windows and add them to the current set of 
        # requests
        for w,new_requests in req_byw:
            # this is the current time
            t = w*delta + delta
            active_requests = active_requests.append(
                new_requests.drop(['window','day','hour'],axis=1)
                )
                        
            # step 1: update the rv graph
            print(f"RV performance t={t}s, {active_requests.shape[0]} requests:")
            start = time.process_time()
            rv_graph,active_requests = update_rv(
                t,active_requests.copy(),
                times,rv_graph,visualize=False
                )
            end = time.process_time()
            
            # step 2: explore complete - subregions of the rv graph for
            # cliques
            print(f"  - number of edges: {len(list(rv_graph.edges))}")
            print(f"  - processing time: {end-start:0.1f}s\n")
            #rtv_graph = create_rtv_graph(rv_graph,active_requests)    
            
            
            ### testing rtv_graph
            rtv_graph = nx.Graph()
            for clique in nx.algorithms.clique.find_cliques(rv_graph):
        
            ### check if constraints are satisfied to then generate ###
            ### feasible trips and assign request and vehicles to trips ###
                
                # If the clique contains only Int64s and no Strings then it
                # is a number of requests that no vehicle can service, so we
                # assign them to the "unassigned" node. DO WE JUST DISCARD?
                # RW- assigning them to an "unassigned" trip sounds right
                
                # TF- Thinking about it, I reckon discarding is probably best 
                # as they are infeasible and won't make any difference to the
                # optimal solution in the ILP. We can just check after 
                # optimisation which requests have not been allocated.
                if all(isinstance(i, type(clique[0])) for i in clique[1:]):
                    pass
                    """
                    for r in clique:
                    # add 'rt' edge
                        rtv_graph.add_edge(r,"unassigned", edge_type = 'rt')
                    """
                else:
                    # Sort the clique in request order with vehicle at the end       
                    clique = sorted(clique, key=lambda x: (x is not None, '' 
                           if isinstance(x, Number) else type(x).__name__, x))
                                        
                    # Get the vehicle name, number of passengers in the vehicle,
                    # trip and the number of requests in the trip
                    vehicle = clique[-1]
                    # print(vehicle)
                    passengers = len(Taxis[vehicle].passengers)
                    
                    # RW: if we're going to use 'trip' as a node id, I think it
                    #     needs to sorted to avoid symetry issues - or use a frozenset
                    trip = tuple(clique[:-1])                    
                    reqs = len(trip)
                    
                    # Start with the "one request + empty vehicle" trips
                    # which will always be feasible because we checked in the rv
                    # graph creation.
                    if reqs == 1 and passengers == 0:
                        
                        rtv_graph = add_one_req(rtv_graph, rv_graph, trip, 
                                                vehicle, active_requests)
                                               
                    # then deal with the "one request + non-empty vehicle" trips.
                    # This trip will have a delay > 0 due to deviations
                    # for both passengers
                    elif reqs == 1 and passengers == 1:
                        
                        rtv_graph = check_one_req_one_passenger(Taxis,rv_graph
                                    ,rtv_graph,times,active_requests,vehicle,
                                    trip,MaxWait,MaxQLoss)
                                                   
                    # finally deal with the "more than one request" trips
                    else:

                        # if the vehicle is empty we add all the individual
                        # requests in the clique as seperate, one request trips 
                        # and check all the pairwise shared request trips
                        if passengers == 0:
                            for r in trip:
                                sub_trip = tuple([r])
                                rtv_graph = add_one_req(rtv_graph, rv_graph, 
                                            sub_trip, vehicle, active_requests)
                            
                            paired_trips = itertools.combinations(trip,2)
                            
                            for pair in paired_trips:
                                pair_trip = tuple(sorted(pair))
                                rtv_graph = check_two_req(rv_graph,rtv_graph,
                                        times,active_requests,vehicle,
                                        pair_trip,MaxWait)
                        
                        # if the vehicle has one passenger we check all the
                        # individual requests to see if they can fit with
                        # the current passenger
                        elif passengers == 1:
                            for r in trip:
                                sub_trip = tuple([r])
                                rtv_graph = check_one_req_one_passenger(Taxis,
                                        rv_graph,rtv_graph,times,
                                        active_requests,vehicle,sub_trip,
                                        MaxWait,MaxQLoss)
                        # otherwise we have a full vehicle and so none of
                        # these requests can be serviced by this vehicle
                        else:
                            pass
                        
            break
        
        break
    
    break