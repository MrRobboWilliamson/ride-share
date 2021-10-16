#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 06:24:40 2021

@author: rob
"""
import os
import struct
import numpy as np
import pandas as pd
import networkx as nx
from utils import ConsoleBar

ROOT = 'datasets'
JT_FOLDER = r'journey_times'
LINK_FOLDER = r'link_times'
GRAPHTYPE = nx.DiGraph()

# get the ny city road network
ROADS = pd.read_csv(os.path.join(
    ROOT,'ny_roads.csv'),header=None,usecols=[0,1])

# update the indexing to match 0 indexing
ROADS -= 1

def create_weighted_graph(filenm):
        
    # get the link journey times and join to the
    # road links
    link_times = pd.read_csv(filenm,header=None)
    weighted_links = pd.concat([ROADS,link_times],axis=1)
    weighted_links.columns = ['o','d','jt']
            
    return nx.from_pandas_edgelist(
        weighted_links,source='o',target='d',
        edge_attr='jt',create_using=GRAPHTYPE
    )

# create an object to retrieve the shortest paths for 
class ShortestPathFinder:
    
    def __init__(self,network,times):
        
        self.G = network
        self.times = times
        self.memory_ = dict()
        
    def shortest_between_points(self,source,target):
        """
        Parameters
        ----------
        source : TYPE int, 
            source node
        target : TYPE int
            target node

        Returns
        -------
        shortest path as a list of nodes
        """
        
        # cache the paths, so we don't have to re-calculate
        if (source,target) not in self.memory_:
            _,path = nx.single_source_dijkstra(
                self.G,source,target=target,weight='jt')
            self.memory_[(source,target)] = path
        else:
            path = self.memory_[(source,target)]
            
        return path
    
    
    def assign_events(self,time,node,requests):
        """
        check if a request has an event at this node, and 
        if so, create an event for the timetable
        ensure no duplicate requests
        """
        
        # check for any pickups or dropoffs in the first node
        picks = requests[requests['from_node']==node].index.to_list()
        drops = requests[requests['to_node']==node].index.to_list()
                
        # init the timetable and add any initial pickups or dropoffs
        events = [[time,node,np.nan,np.nan],] # first event is the next loc
        for pick in picks:
            events.append([time,node,pick,np.nan])
            
        for drop in drops:
            events.append([time,node,np.nan,drop])
            
        return events
    
    
    def get_timetable(self,first_node,path,time,requests):
        """
        Parameters
        ----------
        path : list of requests and nodes
        requests : are
        
        Returns
        -------
        a dataframe with columns ['time','loc','pickup','dropoff']
        
        There's probably a more efficient way to do this, but
        my brain is shutting down
        """
        
        # check for any pickups or dropoffs in the first node
        # print(first_node)
        timetable = self.assign_events(time,first_node,requests)
        # print("\nGet timetable")
        # print(timetable)
                    
        # this is the allocated path from the rtv-graph - just need to 
        # insert the first node (cab's expected location)
        path = [first_node]+[step[1] for step in path]
        for source,target in zip(path[:-1],path[1:]):
            
            # get the detailed steps in the path
            route = self.shortest_between_points(source,target)
            # print()
            # print(source,target)
            # print(route)

            for from_,to_ in zip(route[:-1],route[1:]):
                # get the step time
                jt = self.times[from_][to_]
                
                # add the step to the timetable
                time += jt
                timetable += self.assign_events(time,to_,requests)
                  
        return pd.DataFrame(
            timetable,columns=['time','loc','pickup','dropoff']
            )
    

class JourneyTimes():
    
    """
    Assumes the journey times were extracted and lifted up a level
    to SatMat from Sat
    """
    
    def __init__(self):
        
        # create binary references
        self.root = ROOT
        self.jt_folder = os.path.join(ROOT,JT_FOLDER)
        self.link_folder = os.path.join(ROOT,LINK_FOLDER)
        sat = os.path.join(self.jt_folder,'SatMat')
        sun = os.path.join(self.jt_folder,'SunMat')
        wkd = os.path.join(self.jt_folder,'WeekMat')
        
        # rows are days: sunday, weekday and saturday
        # columns are hours
        self.jt_paths = [[os.path.join(day,f'Ho{h}.bin') for h in range(24)] \
                           for day in [sun,wkd,sat]]
        self.link_paths = [[os.path.join(self.link_folder,f'ny_{day}{h:02d}.csv')
                            for h in range(24)] \
                            for day in ['sun','wd','sat']]
            
    def get_shortest_path_finder(self,day,hour,times):
        """
        PARAMS:
            - day int
            - hour int
            
        RETURNS:
            - ShortestPathFinder object 
              to find the shortest path and times on a given day, hour
        """
        
        # create the road network for this time and day
        road_network = create_weighted_graph(self.link_paths[day][hour])
        return ShortestPathFinder(road_network,times)


    def get_journey_times(self,day,hour):
        """
        Reads the journey times from disk:
            - output size seems to be too big
            
        Returns shortest path journey times
        """
        
        with open(self.jt_paths[day][hour],'rb') as file:
            content = file.read()
                    
        # get the file content
        data = struct.unpack('H'*((len(content))//2),content)        
        times = np.reshape(data,(4092,4092))
        
        return times[1:,1:]        
        
        
    def get_hourofday(self,day,hour):
        
        """
        Reads the journey times and link times from disk:
                        
        Returns shortest path journey times and a weighted network to
        query the detailed shortest path and times.
        """
        
        times = self.get_journey_times(day,hour)
        path_finder = self.get_shortest_path_finder(day,hour,times)
        
        return times,path_finder       
                
                
class Requests():
    
    """
    Read in the request data
    window size is in seconds
    """
    
    def __init__(self,window_size=30,max_wait=120):
        self.root = ROOT
        self.delta = window_size
        self.max_wait = max_wait
    
    def read_requests(self,calc_jts=False):
        
        df = pd.read_csv(os.path.join(self.root,'ride_requests.csv'))
        df['window'] = np.floor(df['time']/self.delta).astype(int)
        
        # what hour and day are we in?
        df['day'] = np.floor(df['time']/(24*3600)).astype(int)
        df['hour'] = np.floor(df['time']/3600).astype(int) % 24
        df['latest_pickup'] = df['time'] + self.max_wait
        
        return df

if __name__ == "__main__":
    # reqs = Requests()
    # requests = reqs.read_requests(calc_jts=True)
    # requests.to_csv("datasets/requests_with_jts.csv")
    
    jt = JourneyTimes()
    times = jt.get_hourofday(0,7)   