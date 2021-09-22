# -*- coding: utf-8 -*-
"""
This is the model
"""
import numpy as np
import heapq
from source_data import JourneyTimes, Requests
from taxis import Taxi, Passenger
np.random.seed(1234)

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
rdf = Requests(r'datasets',delta).read_requests()

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

# instantiate the taxis and add them to the priority queue
Taxis = [Taxi(v,k,init_locs[v]) for v in V]

# this is our priority queue for the taxis
CabRank = [[0,v,Taxis[v]] for v in V]
heapq.heapify(CabRank)

# function to get all of the available taxis
def get_taxis(t,qos,times):
    """
    Based on the node location and the position of the taxis
    in the cab rank, get the 
    
    TODO: consider taxis that will be available after finishing a job. These
    will be in range when they finish their jobs
    """
    # find intersections within range
    in_range = inters[times+qos<=MaxWait]
    print("\nget_taxis")
    print(" - in range intersections:",in_range.shape)
    
    # are there any taxis at these locations?
    # think about how this can be parallelized
    options = []
    out_of_range = []
    for spot in CabRank:
        if spot[2].loc in in_range:
            options.append(spot[2])
            
    print(" - ", len(options), "taxis in range")
    return options

def get_rv_graph(requests,times):
    """
    Generates a shareability graph
    
    Which requests and vehicles might be pair-wise shared.
    - two requests are connected to an empty cab
    """
    
    pass
    

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
day_trans = {
    0:0,1:1,2:1,3:1,4:1,5:1,6:2
    }
for d in D:
    
    # convert to sun, weekday, sat
    dt = day_trans[d]
    
    for h in H:
        
        # get the journey times matrix
        times = jt.get_hourofday(dt,h)
        
        # get all of the requests
        req_slice = rdf[(rdf['day']==d)&(rdf['hour']==h)]
        req_byw = req_slice.groupby('window')
        
        # these are the 30s windows
        for w,requests in req_byw:
            t = w*delta + delta
            # for each request get vehicles within the acceptable
            # wait time, limited by the quality of service (qos)
            
            # TODO: create shareability graph of requests
            # step 1: find vehicals that can satisfy these requests
            rv_graph = get_taxis(t,qos,jt2me)
            
            
            
            # for r,row in requests.iterrows():
            #     # calculate the initial qos - this is from the start
            #     # of the window to the request time
            #     qos = t - row['time']                
                
            #     # get all of the journey times to this request to calculate
            #     # the wait time for the taxis
            #     jt2me = times[:,row['from_node']]
                

                
                # break
            
            break
        
        break
    
    break
    



# print(jt.get_hourofday(0,0)[:3,:3])

# print(reqs.head())










