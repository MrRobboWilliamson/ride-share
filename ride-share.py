# -*- coding: utf-8 -*-
"""
This is the model
"""
import numpy as np
from source_data import JourneyTimes, Requests
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
nWindows = int(Period/delta)

### Sets
W = range(nWindows)
V = range(M)

### Our datasets will be the trip data
# I have downloaded the May 2013 (not 2015) data and 
# filtered for the week starting Sunday 5 May to end Saturday May 11
# filtered for requests within 100m of an intersection
# then did a nearest neighbour analysis to get the closest intersection to
# each request
# requests that start and end at the same node are also removed
rdf = Requests(r'datasets',delta).read_requests()
req_byw = rdf.groupby(['window'])

# get drop off locations from the "previous" saturday night
# to sample taxi locations at the drop off locations
sat_night = rdf[rdf['time']>Period-3600]
init_locs = sat_night['to_node'].values
np.random.shuffle(init_locs)
init_locs = init_locs[:M]

print("Sample of initial location for the taxis to start")
print(init_locs.shape)
print(init_locs[:10])

### Journey time data
# this has been precomputed for every od combination of intersections 
# thanks Giovanni!!
# I have created a journey time class to retrieve the journey times
# for a given day and time (days are 0, 1, 2 for Sun, Weekday, Sat)
# these will help us calculate sharability
# as adding intermediate points will split the journey times

# get the last da

### Taxis
# we will also need to simulate taxis
# Michael used something called Hqueue or something to do this
# it should be in one of the python files on Blackboard
# I suggest we inject them at randomly sampled drop off points


### Evaluation
# get requests in 30s chunks
# generate shareability
# create rv, rtv graphs
# assign vehicles to trips
# requests that can't be satisfied stay in the request pool
for w in W:
    # get the requests collected from the last 30s
    # note that requests at the start of the window
    # have already been waiting 30s
    requests = req_byw.get_group(w)
    
    # what other requests can be shared
    
    print(requests.shape)
    
    
    break




# print(jt.get_hourofday(0,0)[:3,:3])

# print(reqs.head())
print(len(req_byw))










