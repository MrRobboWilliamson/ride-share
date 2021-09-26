# -*- coding: utf-8 -*-
"""
This is the model
"""
import numpy as np
import pandas as pd
from source_data import JourneyTimes, Requests
from taxis import Taxi, Passenger
np.random.seed(1234)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Shareability parameters
# vehicle capacity (k) and quality of service threshold (delta)
# M is the number of vehicles
k = 2
delta = 30
MaxWait = 120 # Two minuts
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
def assign_basejt(requests,times):
    """
    Assign base journey times to all of the requests
    """
    assigned = []
    for from_node,trips in requests.groupby('from_node'):
        
        # get the times
        jt_out = times[from_node,:]
        trips['base_jt'] = jt_out[trips['to_node'].values]
        assigned.append(trips)
        
    return pd.concat(assigned).sort_index()
    
# need to know what hour day we're in for a given
# seconds value

### Taxis
# we will also need to simulate taxis
# Michael used something called Hqueue or something to do this
# it should be in one of the python files on Blackboard
# I suggest we inject them at randomly sampled drop off points
Taxis = { v: Taxi(v,k,init_locs[v]) for v in V }

def show_viz(t,requests,r=None,times=None):
    
    wait_style = dict(
        alpha=0.5,label="wait"
        )
    journey_style = dict(
        label="journey"
        )
    delay_style = dict(
        alpha=0.5,label="delay"
        )
    
    # precalc times
    base_late = requests['latest_pickup'] + requests['base_jt']
    latest_dropoff = base_late + MaxWait    

    # create the plot    
    fig,ax = plt.subplots(figsize=(15,30))
    min_t = requests['time'].min() - 60
    max_t = latest_dropoff.max() + 60
    
    # plot the current time
    plt.vlines(t,-1,requests.shape[0]+1,ls=":",colors="black")
    
    # plot the wait times
    plt.hlines(requests.index,
               requests['time'],
               requests['latest_pickup'],
               **wait_style)
    
    # plot the journey times
    plt.hlines(requests.index,
               requests['latest_pickup'],
               base_late,
               **journey_style)
    
    # plot the delay times
    plt.hlines(requests.index,
               base_late,
               latest_dropoff,
               **delay_style)
    
    plt.xlim([min_t,max_t])
    plt.ylim([-1,requests.shape[0]+1])
    plt.legend()
    plt.show()
    
def shortest_path(leg,pair,path,times):
    """
    recursively find the shortest path between two requests by 
    minimizing waiting and in journey delays
    
    some helpful information:
        - only second pickup incurs an additional wait delay
        - when deciding who to drop first, 
            - if first drop, picked up second - only the last dropped gets jt delays
            - other wise they both do
    """    
    
    if leg > 1:        
        return 0,path
    
    best_path = 99999,False
    for r in pair:
        if leg == 0:            
            # init the path
            path = []
            
            # get the first node
            first = pair[r]['route'][leg]
            path.append((r,first))
            
            # get the second node in the journey
            other = pair[r]['other']
            second = pair[other]['route'][leg]
            path.append((other,second))
            
            # add the initial wait time and check against the MaxWait constraint
            cost = times[first,second] + pair[other]['wait']
            
            # assign a massive cost if we exceed the wait time
            # on the other request
            if cost > MaxWait:
                cost = float('inf')
                
            # now we can add in the first guy's initial wait cost
            ocost = pair[r]['wait']
                        
        else:
            # get the next node and the journey time
            next_ = pair[r]['route'][leg]
            path.append((r,next_))
            
            # get the other request and the last node - which
            # will be the other request's destination
            other = pair[r]['other']
            last = pair[other]['route'][1]
            
            # if this next node is owned by the same request as the last, there
            # will be no journey time delay
            if r == path[-1][0]:
                # so we can calculate the journey time delay on the other
                # request
                cost = times[path[0][1],path[1][1]] + \
                    times[path[1][1],next_] + \
                        times[next_,last] - \
                            pair[other]['base']
                            
                # update the path            
                path.append((other,last))
                ocost = 0                           
            else:
                # we have to calculate the journey time costs against both
                # routes
                jt0 = times[path[0][1],path[1][1]]
                jt1 = times[path[1][1],next_]
                jt2 = times[next_,last]
                
                # update the path and costs
                path.append((other,last))
                cost = jt0 + jt1 - pair[r]['base']
                ocost = jt1 + jt2 - pair[other]['base']
            
            # here we need to check the overall delay constraints
            if cost > MaxWait or ocost > MaxWait:
                cost = float('inf')
            
        path = list(path)
        jt = cost + ocost + shortest_path(leg+1,pair,path,times)[0]
        
        if jt < best_path[0]:
            best_path = jt,path
        
    return best_path


def build_shareable(t,requests,times,request_edges,to_check,visualise=False):        
    
    for r1,row in requests.iterrows(): 
        
        # pop this request off the list
        r1 = to_check.pop(to_check.index(r1))
        t1,o1,d1,ltp1,bjt1,qos1 = requests.loc[r1].values
        # p1 = Passenger(r1,o1,d1,t1,bjt1,qos1)
        
        for r2 in to_check:
            
            t2,o2,d2,ltp2,bjt2,qos2 = requests.loc[r2].values            
            # p2 = Passenger(r2,o2,d2,t2,bjt2,qos2)
            
            # get the shortest path to satisfy a pair of requests
            pair = dict([(r1,dict(route=(o1,d1),other=r2,wait=qos1,base=bjt1)),
                         (r2,dict(route=(o2,d2),other=r1,wait=qos2,base=bjt2))])
            best = shortest_path(0,pair,None,times)
            
            if best[1]:
                print(best)
                request_edges[(r1,r2)] = best    
                break           
            

def get_rv_graph(t,requests,times,visualize=False):
    """
    Generates a shareability graph
    
    Which foreach vehicle, generates a graph of pair-wise shareable requests
    - two requests are connected to an empty cab if the total travel
      delays experienced by either request are satisfied
    """
    
    # create shareable combinations
    request_edges = dict() # key frozenset(r1,r2) value sum of delays
    vehicle_edges = dict() # key (r,v) value 
    
    # this is the initial wait time
    requests['qos'] = t - requests['time'].values
        
    # get the request edges
    to_check = list(requests.index.values)
    request_edges = build_shareable(t,requests,times,request_edges,to_check)
    
    for v in Taxis:
        
        if Taxis[v].available:
            
            # get the requests that are servicable
            cab = Taxis[v]
            jtfromme = times[cab.loc,:]            
            potentials = requests.copy()[requests['qos']+jtfromme[requests['from_node']]<=MaxWait]
            
            if potentials.empty:
                continue
            
            if potentials.shape[0] == 1:
                # if there is only one combination posible, then record the 
                # cost and add it to the edges
                r = potentials.index.values[0]
                
                # calculate the delays
                total_delay = potentials['qos'].values[0] + \
                    jtfromme[potentials['from_node']][0]
                    
                # assign edges and record solo trib
                vehicle_edges[(r,v)] = total_delay
                request_edges[frozenset((r,r))] = total_delay
                
                # skip to the next cab
                continue
            
            
            break
         

### Evaluation
# get requests in 30s chunks
# generate shareability
# create rv, rtv graphs
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
        
        # these are the 30s windows
        for w,requests in req_byw:
            # this is the current time
            t = w*delta + delta
            
            # TODO: create shareability graph of requests
            # step 1: find vehicals that can satisfy these requests
            rv_graph = get_rv_graph(t,requests.iloc[:,[0,1,2,-2,-1]].copy(),
                                    times,visualize=False)
            
            break
        
        break
    
    break

# print(jt.get_hourofday(0,0)[:3,:3])

# print(reqs.head())










