# -*- coding: utf-8 -*-
"""
This is the model
"""
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import time
import networkx as nx
from numbers import Number
import itertools

# our modules
from source_data import JourneyTimes, Requests
from taxis import Taxi
from utils import (
    Logger,assign_basejt,plot_requests,shortest_path, #is_cost_error,
    check_two_req, add_one_req, check_one_req_one_passenger,
    process_assignments,update_current_state,shortest_withpassenger
    )                   
from allocate import create_ILP_data_v2, allocate_trips_v2, greedy_assignment

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

### Dates and days for printing
DAY_NAMES = {i: daynm for i,daynm in enumerate(
    ["Sunday",
     "Monday",
     "Tuesday",
     "Wednesday",
     "Thursday",
     "Friday",
     "Saturday"])
    }
MY = "/5/2013"
DATES = {i: str(i+5)+MY for i in range(len(DAY_NAMES))}
# print(DAY_NAMES)
# print(DATES)

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
rdf = Requests(delta,MaxWait).read_requests()

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
jt = JourneyTimes() #r'datasets/journey_times')
    
# need to know what hour day we're in for a given
# seconds value

### Taxis
# we will also need to simulate taxis
# Michael used something called Hqueue or something to do this
# it should be in one of the python files on Blackboard
# I suggest we inject them at randomly sampled drop off points
event_logger = Logger(r'output_logs')
CabIds = { f"v{v}" for v in V }
Taxis = { f"v{v}": Taxi(f"v{v}",k,init_locs[v],event_logger,MaxWait,MaxWait) for v in V }
    
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
            

def create_rv_graph(t,requests,times): #,visualize=False):
    """
    Creates the RV graph
    
    Step 1:
        - update the pairwise shareability graph
        
    Step 2:
        - check which vehicles can be assigned to which requests
        - need to be able to assign requests to cabs with one availabe
          seat en route    
    """
        
    # rv graph
    rv_graph = nx.Graph()
    
    # this is the initial wait time
    requests['qos'] = t - requests['time'].values
    requests['is_alone'] = True
    
    # returns a mask of the lonely requests
    start = time.process_time()    
    requests,rv_graph = build_shareable(t,requests,times,rv_graph)
    end = time.process_time()
    print(f"  - build_shareable function {end-start:0.1f}s")
    
    # if visualize:
    #     ### THIS WON'T WORK - NEEDS TO BE UPDATED ###
    #     # randomly get a request that is shareable
    #     r = list(np.random.choice(list(rv_graph.keys())))[0]
        
    #     # get all of the other requests
    #     to_plot = set()
    #     for pair in rv_graph['rr']:
    #         if r in pair:
    #             to_plot |= pair 
                
    #     print(r,to_plot)
    #     print(requests[requests.index.isin(to_plot)])
    #     plot_requests(t,requests[requests.index.isin(to_plot)],MaxWait,r)
    
    # Now we need to check which taxis can service which
    # requests
    for v in Taxis:
                
        # get the cab
        cab = Taxis[v]
        
        # check where the cab will be next and the time to get there
        arrive_time,next_loc = cab.find_me(t)                
        time_to_next_loc = arrive_time - t
        
        # potential requests based on the time from the next location
        # to each request
        jt_from_me = times[next_loc,:]
        potentials = requests[
            requests['qos']+time_to_next_loc + 
            jt_from_me[requests['from_node']]<=MaxWait
            ]
        
        # if there's not potential pickups, no point checking anying
        # just skip this guy
        if potentials.empty:
            continue
        
        # check how many passengers will be in the cab at it's
        # next location
        passengers = cab.get_passengers(t)
        
        # if there are no passengers in the cab just create the 
        # rv edges
        if len(passengers) == 0:
                              
            # assign edges between cab and requests with the journey
            # time as the value - don't assign the initial qos twice - 
            # it is assigned in the shortest path function
            
            # is the cab is available, just assign edges
            for i,delay in enumerate(jt_from_me[potentials['from_node']]):
                rv_graph.add_edge(
                    potentials.index[i],
                    v,cost=delay+time_to_next_loc
                    )
                
        # if there are passengers on board, check where the cab will be
        # and check with every passenger if picking up the available
        # request will breach their constraints
        else:
            # for each passenger that will still be in the cab at the 
            # next location, check if they can afford to reroute to 
            # this request or be dropped off on the way
                        
            # if the cab is full at this point we can only service
            # a request after the next drop off
            #   - (not checking combinations, just assume next is first)
            if len(passengers) == k:
                
                # get the next drop off
                drop_time,drop_loc,drop_id = cab.get_next_dropoff()
                
                # which requests can we service after dropping off 
                time_to_drop = drop_time - t
                jt_from_drop = times[drop_loc,:]
                
                potentials = potentials[
                    potentials['qos'] + time_to_drop + 
                    jt_from_drop[potentials['from_node']]<=MaxWait
                    ]
                
                # if no potentials after the first drop, move on to the
                # next cab
                if potentials.empty:
                    continue
            
                # otherwise, remove this passengner from the list to check
                # the others
                passengers = [p for p in passengers if p.req_id != drop_id]
                
                # update next location to this dropoff point
                arrive_time = drop_time
                next_loc = drop_loc
            
            # here we need to check if the remaining passenger and each
            # request can be combined
            for p in passengers:               
                
                for r,req_t,o,d,ltp,bjt,qos,_ in potentials.to_records():
                                    
                    # check combinations
                    cost,path = shortest_withpassenger(
                        p,arrive_time,next_loc,
                        dict(req_id=r,route=(o,d),
                             req_t=req_t,base=bjt),
                        times,MaxWait
                        )
                    
                    # if there is a valid combination, add the edge with the 
                    # cost.
                    if path:
                        
                        # add the edge with the expected cost to the request
                        # it might be useful to have the calculated path too?
                        rv_graph.add_edge(r,v,cost=cost-qos, # remove the qos for consistency
                            path=path
                            )
                    
    return rv_graph,requests

def create_rtv_graph(t,rv,active_requests,times,MaxWait):
    """
    Explore the cliques of the rv graph to find feasible trips
    """
    rtv_graph = nx.Graph()
    
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
        # RW - not necessarily, we need these guys are used to 
        # rebalance the fleet
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
            num_passengers = len(Taxis[vehicle].get_passengers(t))
            
            # RW: if we're going to use 'trip' as a node id, I think it
            #     needs to sorted to avoid symetry issues - or use a frozenset
            trip = tuple(clique[:-1])                    
            reqs = len(trip)
            
            # Start with the "one request + empty vehicle" trips
            # which will always be feasible because we checked in the rv
            # graph creation.
            if reqs == 1 and num_passengers == 0:
                
                rtv_graph = add_one_req(t,rtv_graph,rv_graph,trip, 
                                        vehicle,active_requests)
                                       
            # then deal with the "one request + non-empty vehicle" trips.
            # This trip will have a delay > 0 due to deviations
            # for both passengers
            elif reqs == 1 and num_passengers == 1:
                
                rtv_graph = check_one_req_one_passenger(t,Taxis,rv_graph
                            ,rtv_graph,times,active_requests,vehicle,
                            trip,MaxWait) #,MaxQLoss)
                                           
            # finally deal with the "more than one request" trips
            else:
        
                # if the vehicle is empty we add all the individual
                # requests in the clique as seperate, one request trips 
                # and check all the pairwise shared request trips
                if num_passengers == 0:
                    for r in trip:
                        sub_trip = tuple([r])
                        rtv_graph = add_one_req(t,rtv_graph,rv_graph, 
                                    sub_trip,vehicle,active_requests)
                    
                    # RW: not sure if it's guaranteed, but I think combinations
                    # preserves order, so you might not need to sort again
                    # later on| TF: Sweet :)
                    paired_trips = itertools.combinations(trip,2)
                    
                    for pair in paired_trips:
                        # may not need to sort, can save a little bit of time
                        # pair_trip = tuple(sorted(pair))
                        rtv_graph = check_two_req(t,rv_graph,rtv_graph,
                                times,active_requests,vehicle,
                                pair,MaxWait)
                
                # if the vehicle has one passenger we check all the
                # individual requests to see if they can fit with
                # the current passenger
                elif num_passengers == 1:
                    for r in trip:
                        sub_trip = tuple([r])
                        rtv_graph = check_one_req_one_passenger(t,Taxis,
                                rv_graph,rtv_graph,times,
                                active_requests,vehicle,sub_trip,
                                MaxWait) #,MaxQLoss)
                # otherwise we have a full vehicle and so none of
                # these requests can be serviced by this vehicle
                else:
                    pass
                
    return rtv_graph    

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

# Cycle in hour chunks, so we don't have to check to load
# new journey times each iteration we only load them on the hour
active_requests = pd.DataFrame()
for d in D:
    
    # convert to sun, weekday, sat
    dt = day_trans[d]
    
    for h in H:
        
        # get the journey times matrix
        times,path_finder = jt.get_hourofday(dt,h)
        
        # get all of the requests - we might want to preprocess these?
        req_slice = assign_basejt(rdf[(rdf['day']==d)&(rdf['hour']==h)],
                                  times)
        req_byw = req_slice.groupby('window')
        
        # get new requests in 30s windows and add them to the current set of 
        # requests
        for w,new_requests in req_byw:
            # this is the current time
            t = w*delta + delta          
            
            print(f"\nStarting time window t={t}s\n")
            # step 0: based on current time, process in-flight trips to remove
            # passengers from cabs if they've finished their journey
            
            # and pickup passengers if they've been picked up
                        
            #### THIS IS WHEN WE PICKUP AND DROP OFF PASSENGERS ####
            print("Updating current state")
            start_update_state = time.process_time()
            active_requests = update_current_state(t,
                                                   active_requests,
                                                   Taxis)
            
            # add on the new requests
            active_requests = active_requests.append(
                new_requests.drop(['window','day','hour'],axis=1)
                )
            end_update_state = time.process_time()
            print(f"  - compute time = {end_update_state-start_update_state:0.1f}s\n")
                        
            # step 1: create the rv graph
            print(f"RV performance {active_requests.shape[0]} requests:")
            start = time.process_time()
            rv_graph,active_requests = create_rv_graph(
                t,active_requests.copy(),
                times
                )
            end = time.process_time()
            
            # step 2: explore complete - subregions of the rv graph for
            # cliques
            print(f"  - number of edges: {len(list(rv_graph.edges))}")
            print(f"  - processing time: {end-start:0.1f}s\n")
            
            start_rtv = time.process_time()
            rtv_graph = create_rtv_graph(t,rv_graph,active_requests,times,MaxWait)    
            end_rtv = time.process_time()
            print("RTV performance:")
            print(f"  - number of edges: {len(list(rtv_graph.edges))}")
            print(f"  - processing time: {end_rtv-start_rtv:0.1f}s\n")

            # step 3: optimization
            """
            V, R, T, VT, RT = create_ILP_data(rtv_graph)
            Vehicle_Trips1, Requests_Trips1 = allocate_trips(V, R, T, VT, RT)            
            """
            start_rtv = time.process_time()
            rt_graph = greedy_assignment(rtv_graph, k)
            end_rtv = time.process_time()
            print("Greedy Assignment performance:")
            print(f"  - number of edges: {len(list(rtv_graph.edges))}")
            print(f"  - processing time: {end_rtv-start_rtv:0.1f}s\n")

            start_rtv = time.process_time()
            V, R, T, VT, RT, TV = create_ILP_data_v2(rtv_graph)
            end_rtv = time.process_time()
            print("Create data performance:")
            print(f"  - number of edges: {len(list(rtv_graph.edges))}")
            print(f"  - processing time: {end_rtv-start_rtv:0.1f}s\n")

            
            # Get the Trip assignments            
            Trips = allocate_trips_v2(V, R, T, VT, RT, TV,
                                      suppress_output=True)
            
            # This function simply adds trips to Cabs as a "timetable"
            # we only create passengers and add them to cabs in subsequent
            # iterations
            start_process_assign = time.process_time()
            ignored,idle = process_assignments(
                t,Trips,Taxis,active_requests,path_finder,rtv_graph,CabIds
                )
            end_process_assign = time.process_time()
            
            print("Processing ILP assignments:")
            print(f"  - compute time {end_process_assign-start_process_assign:0.1f}s")
            print(f"  - {ignored.shape[0]}/{active_requests.shape[0]} ignored requests")
            print(f"  - {len(idle)}/{M} idle cabs")            
            
                        
            ##### STILL NEED TO DO REBALANCING #####
            
            if t == 600:
                break
        
        # dump the logs
        event_logger.dump_logs(d,h)
        
        # break    
    
    # break