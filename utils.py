#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 08:46:40 2021

@author: rob
"""
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

class ConsoleBar:
    def __init__(self,num_ticks,length=100):
        # check that end - start is greater than one and that they are both integers
        if type(num_ticks) is not int:
            raise TypeError('arg "num_ticks" must be type int')
        if num_ticks < 1:
            raise ValueError("num_ticks not > 0")

        #  get the absolute size and normalise
        self.num_ticks = num_ticks
        self.ticker = 0
        self.length = length

        # start the ticker
        # print('\r   {:>3.0f}%[{: <101}]'.format(0, '='*0+'>'), end='\r')

    def tick(self, step=1):
        print_end = '\r\r'
        self.ticker += step
        if self.ticker > self.num_ticks:
            self.ticker = self.num_ticks
            # print a warning
            print('Warning: The ticker is overreaching and has been capped at the end point', end='\r')
        elif self.ticker == self.num_ticks:
            print_end = '\n'

        progress = int((self.ticker/self.num_ticks)*100)
        bar_ticks = int((self.ticker/self.num_ticks)*self.length)
        bar_str = '   {:>3.0f}%[{: <' + str(self.length+1) + '}]'
        clear_output(wait=True)
        print(bar_str.format(progress, '.'*bar_ticks+'>'), end=print_end)
        
def plot_requests(t,requests,MaxWait,r=None):
    
    colors = ['tab:blue']*requests.shape[0]
    
    if r is not None:
        # update the color of the request in focus
        colors[list(requests.index).index(r)] = 'tab:orange'
        
    wait_style = dict(
        alpha=0.5,label="wait",colors=colors
        )
    journey_style = dict(
        label="journey",colors=colors
        )
    delay_style = dict(
        alpha=0.5,label="delay",colors=colors
        )
    
    print(colors)
    
    # precalc times
    base_late = requests['latest_pickup'] + requests['base_jt']
    latest_dropoff = base_late + MaxWait    

    # create the plot    
    fig,ax = plt.subplots(figsize=(15,max(2,int(requests.shape[0]/5))))
    min_t = requests['time'].min() - MaxWait
    max_t = latest_dropoff.max() + MaxWait
    
    # plot the current time
    plt.vlines(t,-1,requests.shape[0]+1,ls=":",colors="black")
    
    # plot the wait times
    y_positions = list(range(requests.shape[0]))
    plt.hlines(y_positions,
               requests['time'],
               requests['latest_pickup'],
               **wait_style)
    
    # plot the journey times
    plt.hlines(y_positions,
               requests['latest_pickup'],
               base_late,
               **journey_style)
    
    # plot the delay times
    plt.hlines(y_positions,
               base_late,
               latest_dropoff,
               **delay_style)
    
    plt.xlim([min_t,max_t])
    plt.ylim([-1,requests.shape[0]])
    
    # lable the requests
    ylabels = [None,] + list(requests.index) + [None,]
    yticks = [-1,] + y_positions + [y_positions[-1]+1,]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    plt.legend()
    plt.show()
    
    
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

__shortest_paths = dict()
def shortest_path(leg,pair,path,times,MaxWait):
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
        jt = cost + ocost + shortest_path(leg+1,pair,path,times,MaxWait)[0]
        
        if jt < best_path[0]:
            best_path = jt,path
        
    return best_path