# -*- coding: utf-8 -*-
"""
This is the model
"""

from source_data import jt, reqs

# Shareability parameters
# vehicle capacity (k) and quality of service threshold (delta)
# M is the number of vehicles
k = 2
delta = 30
M = 3000

### Our datasets will be the trip data
# I have downloaded the May 2013 (not 2015) data and 
# filtered for the week starting Sunday 5 May to end Saturday May 11
# filtered for requests within 100m of an intersection
# then did a nearest neighbour analysis to get the closest intersection to
# each request
# requests that start and end at the same node are also removed

### Journey time data
# this has been precomputed for every od combination of intersections 
# thanks Giovanni!!
# I have created a journey time class to retrieve the journey times
# for a given day and time (days are 0, 1, 2 for Sun, Weekday, Sat)
# these will help us calculate sharability
# as adding intermediate points will split the journey times

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






print(jt.get_hourofday(0,0)[:3,:3])

print(reqs.head())











