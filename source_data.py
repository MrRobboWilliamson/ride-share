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
    
    def __init__(self,root,window_size):
        self.root = root
        self.delta = window_size
    
    def read_requests(self):
        
        df = pd.read_csv(os.path.join(self.root,'ride_requests.csv'))
        df['window'] = np.floor(df['time'] / self.delta).astype(int)        
        
        return df