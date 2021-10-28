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
    """
    Reads the road network street times and generates a
    corresponding directed graph
    """
    
    link_times = pd.read_csv(filenm,header=None)
    weighted_links = pd.concat([ROADS,link_times],axis=1)
    weighted_links.columns = ['o','d','jt']
            
    return nx.from_pandas_edgelist(
        weighted_links,source='o',target='d',
        edge_attr='jt',create_using=GRAPHTYPE
    )


class ShortestPathFinder:
    """    
    This is provided to taxis in a stateless fashion calculate intermediate
    nodes in the road network on a needs basis
    
    There should only be one instantiated path_finder per hour
    """
    
    def __init__(self,network,times):
        
        self.G = network
        self.times = times
        
        # attempt at memoization
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
        
    
    def get_timetable(self,first_node,path,time):
        """
        Parameters
        ----------
        first_node : loc_id of cab's current or next location
        path : list of requests, nodes and is_pickup flags
        time : reference time (s)
        
        Returns
        -------
        a dataframe with columns ['time','loc','pickup','dropoff']
        - pickup / dropoff will contain a request / passenger id
          where a pickup or drop off is scheduled at that location
        """
        
        # this is the allocated path from the rtv-graph - just need to 
        # insert the first node (cab's expected location)
        path = [(None,first_node,False),]+[step for step in path]
        events = [[time,first_node,None,None],]
        for source,target in zip(path[:-1],path[1:]):
            
            # get the detailed steps in the path
            route = self.shortest_between_points(source[1],target[1])
            
            # if we have co-located events, then just
            # process the target and there is no time adjustment
            if len(route) == 1:                
                if target[2]:
                    events.append(
                        [time,target[1],target[0],None]
                        )
                else:
                    events.append(
                        [time,target[1],None,target[0]]
                        )

            for from_,to_ in zip(route[:-1],route[1:]):
                # get the time-step and then step-in-time
                time += self.times[from_][to_]
                
                if to_ == target[1] and target[2]:
                    events.append(
                        [time,to_,target[0],None]
                        )
                elif to_ == target[1] and not target[2]:
                    events.append(
                        [time,to_,None,target[0]]
                        )
                else:
                    events.append(
                        [time,to_,None,None]
                        )                    
    
            
        return pd.DataFrame(
            events,columns=['time','loc','pickup','dropoff']
            )
    

class JourneyTimes():
    
    """
    Class for reading od journey time matricies
    - also gets a shortest path finder for a given day,hour
    """
    
    def __init__(self):
        
        # create file references
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
            - day int, 0-6
            - hour int, 0-23
            - times od journey times
            
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
            - seems to be too big to load them all at once
            
        Returns shortest path journey times
        """
        
        with open(self.jt_paths[day][hour],'rb') as file:
            content = file.read()
                    
        # get the file content
        data = struct.unpack('H'*((len(content))//2),content)        
        times = np.reshape(data,(4092,4092))
        
        # ignore the first row / column - all zeros
        return times[1:,1:]        
        
        
    def get_hourofday(self,day,hour):
        
        """
        Returns journey times and shortest path finder for a 
        given day, hour
        - day int, 0-6
        - hour int, 0-23
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
    pass