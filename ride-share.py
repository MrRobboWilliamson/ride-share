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

def build_shareable(t,requests,times,request_edges,to_check,visualise=False):
    
    if not to_check:
        print('Done')
        return request_edges
    
    # pop a request off the list
    r = to_check.pop(0)
    
    # search for shareble combos
    this_request = requests.loc[r]
    # for other in requests.index:
        
        # total_delay = 0
        
        fig,ax = plt.subplots()
        min_t = requests['time'].min() - 60
        max_t = requests['time'].max() + 60
        
        
        # to_check
    
    
    
    # return build_shareble(requests,times,request_edges,checked)


def get_rv_graph(t,requests,times,visualize=False):
    """
    Generates a shareability graph
    
    Which foreach vehicle, generates a graph of pair-wise shareable requests
    - two requests are connected to an empty cab if the total travel
      delays experienced by either request are satisfied
    """
    
    # create shareable combinations
    request_edges = dict() # key (r1,r2) value sum of delays
    vehicle_edges = dict() # key (r,v) value 
    
    # this is the initial wait time
    requests['qos'] = t - requests['time'].values
        
    # get the request edges
    to_check = list(requests.index.values)
    request_edges = build_shareable(requests.copy(),times,request_edges,to_check)
    
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
                request_edges[(r,r)] = total_delay
                
                # skip to the next cab
                continue
            
            
            break
            
        
        # # fitler out request that start after the latest pickup
        # # time - and finish before my pickup time
        # shares = requests[requests['time']<=r['latest_pickup']].drop(i).copy()
        
        # print("feasible pick ups", shares.shape)
        
        # # get all the combinations where this request is picked up
        # # first
        # pre_node = r['from_node']
        # jt2nxt = times[pre_node,:] # times from this request
        # shares = requests.drop(i).copy()
        
        # # out of the potential shared rides, which ones will
        # # exceed the in journy qos?
        
        
        # # assign the origin node and calculate the time
        # # to get from the origin node to the other nodes
        # shares['enroute'] = jt2nxt[shares['from_node'].values]
        # shares['qos'] += jt2nxt[shares['from_node'].values]
        
        # print(shares.shape)
        
        # # remove any that violate the waiting contraint
        # can_wait = shares[shares['qos']<MaxWait]
        
        # # get the best qos
        # best_qos = can_wait['qos'].min()
        
        # print(can_wait.shape)
        
        # # now we need to check which available taxis that can satisfy the 
        # # waiting constraint for the orequest
        # # get all of the "in range" intersections
        # jt2me = times[:,pre_node] # times to this request
        # in_range = inters[jt2me+best_qos<=MaxWait]
        # candidates = [cab[2] for cab in CabRank if cab[2].loc in in_range]
        
        # # check which cabs can satisfy the request combinations
        # options = dict()
        # for j,share in can_wait.iterrows():
            
        #     # collect feasible cabs for this trip
        #     for cab in candidates:
                
        #         # get the results
        #         if share['qos'] + jt2me[cab.loc] <= MaxWait:
                    
        #             # these are the options for this request to pickup
        #             options[(i,j,cab.taxi_id)] = share['qos'] + jt2me[cab.loc]
                    
        # # now focussing on the 
             
        # break

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










