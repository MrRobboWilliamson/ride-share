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
from utils import ConsoleBar


class JourneyTimes():
    
    """
    Assumes the journey times were extracted and lifted up a level
    to SatMat from Sat
    """
    
    def __init__(self,root):
        
        # create binary references
        sat = os.path.join(root,'SatMat')
        sun = os.path.join(root,'SunMat')
        wkd = os.path.join(root,'WeekMat')
        
        # rows are days: sunday, weekday and saturday
        # columns are hours
        self.hour_paths = [[os.path.join(day,f'Ho{h}.bin') for h in range(24)] \
                           for day in [sun,wkd,sat]]
            
    def get_hourofday(self,day,hour):
        
        """
        Reads the journey times from disk:
            - output size seems to be too big
        """
        
        with open(self.hour_paths[day][hour],'rb') as file:
            content = file.read()
                    
        # get the file content
        data = struct.unpack('H'*((len(content))//2),content)        
        times = np.reshape(data,(4092,4092))
        
        return times[1:,1:]
        
        
    def show_paths(self):
        """
        Test that path names are correct
        """
        
        days = ['Sunday', 'Weekday', 'Saturday']
        
        for day in range(3):
            
            print(days[day])           
            
            for hour in range(24):
                
                print("  >",self.hour_paths[day][hour])
                
                
class Requests():
    
    """
    Read in the request data
    window size is in seconds
    """
    
    def __init__(self,root=r'datasets',window_size=30,max_wait=120):
        self.root = root
        self.delta = window_size
        self.max_wait = max_wait
    
    def read_requests(self,calc_jts=False):
        
        df = pd.read_csv(os.path.join(self.root,'ride_requests.csv'))
        df['window'] = np.floor(df['time']/self.delta).astype(int)
        
        # what hour and day are we in?
        df['day'] = np.floor(df['time']/(24*3600)).astype(int)
        df['hour'] = np.floor(df['time']/3600).astype(int) % 24
        df['latest_pickup'] = df['time'] + self.max_wait
        
        if calc_jts:
            # instantiate the journey time reader
            jt = JourneyTimes(r'datasets/journey_times')
            day_trans = {2:1,3:1,4:1,5:1,6:2}
            df['dtr'] = np.where(df['day'].isin(day_trans),
                                 df['day'].map(day_trans),
                                 df['day']).astype(int)
            
            # assign journey times ahead of time
            df_bytimeorigin = df.copy().groupby(['dtr','hour','from_node'])
            num_ticks = len(df_bytimeorigin)
            
            print(f"Allocating journey times to requests ({num_ticks} groups)")
            bar = ConsoleBar(num_ticks,length=80)
            for (day,hour,origin),requests in df_bytimeorigin:
                
                # get the journey times
                times = jt.get_hourofday(day,hour)
                
                # assign to the requests and collect the result
                df.loc[requests.index,'base_jt'] = times[origin,requests['to_node'].values]        
                
                bar.tick()
                
            df['latest_dropoff'] = df['latest_pickup'] + df['base_jt'] + self.max_wait
        
        return df

if __name__ == "__main__":
    reqs = Requests()
    requests = reqs.read_requests(calc_jts=True)
    requests.to_csv("datasets/requests_with_jts.csv")