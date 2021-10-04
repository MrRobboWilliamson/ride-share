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
        
def is_cost_error(cost,path,pair,times,show=False):
    """
    Check the costs
    """
    # get the request ids
    pick_1 = path[0][0]
    pick_2 = path[1][0]
    
    # get the leg times
    leg_1 = times[path[0][1],path[1][1]]
    leg_2 = times[path[1][1],path[2][1]]
    leg_3 = times[path[2][1],path[3][1]]    
    
    # get the waiting costs for the first and second pickup
    wait_1 = pair[pick_1]['wait']
    wait_2 = leg_1 + pair[pick_2]['wait']
    
    # first pickup gets the first and second leg journey time
    # both requests get the second leg (otherwise not shared)
    jt_1 = leg_1 + leg_2
    jt_2 = leg_2
    
    # get the drop off request id's
    drop_1 = path[2][0]
    drop_2 = path[3][0]
    
    # if the second pickup is dropped first, then first cops the last leg
    # else second cops the last leg
    if pick_2 == drop_1:        
        jt_1 += leg_3
    else:
        jt_2 += leg_3               
    
    # now we compare the journey times to the base journey times
    base_1 = pair[pick_1]['base']
    base_2 = pair[pick_2]['base']
    delay_1 = jt_1 - base_1
    delay_2 = jt_2 - base_2    
    
    if show:
        
        print("\nPath:",path)
        print(f"Costs: {cost} != {wait_1+wait_2+delay_1+delay_2}")
        
        # request 1 costs
        print(f"  R1: Wait: {wait_1}s")        
        if pick_2 == drop_1:
            print(f"      Delay: {delay_1}s = {jt_1} - {base_1} ({leg_1}+{leg_2}+{leg_3})")
        else:
            print(f"      Delay: {delay_1}s = {jt_1} - {base_1} ({leg_1}+{leg_2})")            
        
        # request 2 costs
        print(f"  R2: Wait: {wait_2}s = {pair[pick_2]['wait']} + {leg_1}")
        if pick_2 == drop_1:
            print(f"      Delay: {delay_2}s = {jt_2} - {base_2} (should be 0)")
        else:
            print(f"      Delay: {delay_2}s = {jt_2} - {base_2} ({leg_2}+{leg_3})")
            
    
    return (cost!=(wait_1+wait_2+delay_1+delay_2))
    
        
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

def shortest_path(pair,times,MaxWait):
    """
    TODO: update comments
    
    some helpful information:
        - only second pickup incurs an additional wait delay
        - when deciding who to drop first, 
            - if first drop, picked up second - only the last dropped gets jt delays
            - other wise they both do    
    """    
    
    # pickups and dropoffs
    pickups = list(pair.keys())
    dropoffs = list(pickups)
    
    best_path = 9999999,False   
    for first_pickup in pickups:
                                    
        # get this request pickup node
        first_node = pair[first_pickup]['route'][0]
            
        # get the second node in the journey
        second_pickup = pair[first_pickup]['other']
        second_node = pair[second_pickup]['route'][0]
            
        # add the initial wait time and check against the MaxWait constraint
        first_pickup_cost = pair[first_pickup]['wait']
        second_pickup_cost = times[first_node,second_node] + pair[second_pickup]['wait']
            
        # assign a massive cost if we exceed the wait time
        # on the other request
        if first_pickup_cost > MaxWait or second_pickup_cost > MaxWait:
            continue        
                                                   
        # now look at the dropoffs
        for first_dropoff in dropoffs:
            
            # get the last dropoff and the nodes
            second_dropoff = pair[first_dropoff]['other']            
            third_node = pair[first_dropoff]['route'][1]
            fourth_node = pair[second_dropoff]['route'][1]
                        
            # calculate the journey times
            jt_0 = times[first_node,second_node]
            jt_1 = times[second_node,third_node]
            jt_2 = times[third_node,fourth_node]
            
            # if the first drop is the same request as the second pickup, there
            # will be no journey time delay on the first drop
            # its all on the second drop
            if first_dropoff == second_pickup:                                
                first_dropoff_cost = 0 
                second_dropoff_cost = (jt_0+jt_1+jt_2) - pair[second_dropoff]['base']                         
            
            # otherwise it is split between the requests
            else:
                first_dropoff_cost = jt_0 + jt_1 - pair[first_dropoff]['base']
                second_dropoff_cost = jt_1 + jt_2 - pair[second_dropoff]['base']
            
            # here we need to check the overall delay constraints
            if first_dropoff_cost > MaxWait or second_dropoff_cost > MaxWait:
                continue
            
            # calculate the total cost
            total_cost = first_pickup_cost + second_pickup_cost + \
                first_dropoff_cost + second_dropoff_cost
                    
            if total_cost < best_path[0]:
                path = [(first_pickup,first_node),
                        (second_pickup,second_node),
                        (first_dropoff,third_node),
                        (second_dropoff,fourth_node)]                
                best_path = total_cost,path
        
    return best_path


# add a feasible single request trip to the rtv graph
# edge of the rv graph and no delay time for the trip. Add an
# 'tv' edge with those two values and a 'rt' edge between the
# request and the trip. NOTE: Trips are tuples so that multi
# request trips can be accomodated.
def add_one_req(rtv_graph, rv_graph, trip, vehicle, active_requests):
    
    # add 'rt' edge
    rtv_graph.add_edge(trip, vehicle, wait = 
            active_requests.loc[trip[0]]['qos'] + \
            rv_graph.get_edge_data(trip[0],vehicle)['cost'],
            delay = 0, edge_type = 'tv')
    # add 'rt' edge
    rtv_graph.add_edge(trip[0],trip, edge_type = 'rt')
    
    return rtv_graph


# check a one request trip with a vehicle that already has a passenger in it.
# If the added delay to the passenger in the vehicles is within bounds add 
# the trip to the rtv graph otherwise discard
def check_one_req_one_passenger(Taxis,rv_graph,rtv_graph,times,active_requests,
                  vehicle,trip,MaxWait,MaxQLoss):
    
    # get the passenger's details
    r1 = Taxis[vehicle].passengers[0].req_id
    o1 = Taxis[vehicle].loc # passenger is in the taxi
    d1 = Taxis[vehicle].passengers[0].drop_off_node
    qos1 = Taxis[vehicle].passengers[0].wait_time
    bjt1 = times[o1][d1] # whatever is left of the journey
    
    # get the new pick up request details
    r2 = trip[0]
    t2,o2,d2,ltp2,bjt2,qos2,_ = active_requests.loc[r2]
    qos2 += rv_graph.get_edge_data(r2,vehicle)['cost']
    
    # build the data pair for the shortest path algorithm
    pair = dict([(r1,dict(
                route=(o1,d1),other=r2,wait=qos1,base=bjt1)),
                (r2,dict(
                route=(o2,d2),other=r1,wait=qos2,base=bjt2))
                ])
    
    # get the cost and shortest path
    cost,path = shortest_path(pair,times,MaxWait)
    
    # if the shortest path cost exceeds the max delays,
    # discard the trip, otherwise add the 'rt' and 'tv' edges
    if cost > MaxQLoss:
        pass
    else:
        tot_wait = qos2 + qos1
        # add 'tv' edge
        rtv_graph.add_edge(trip, vehicle, wait = tot_wait, delay = cost - 
                           tot_wait, edge_type = 'tv', path = path)
        # add 'rt' edge
        rtv_graph.add_edge(r2,trip, edge_type = 'rt')
    
    return rtv_graph


# checks a trip that has two requests for feasibility and either discards it
# or adds it to the rtv_graph.  We know the trip delay and
# the wait time for the first pickup is within feasible 
# limits, just need to check the additional wait time for 
# the second pickup (time between first and second node)
def check_two_req(rv_graph,rtv_graph,times,active_requests,
                  vehicle,trip,MaxWait):

    trip_data = rv_graph.get_edge_data(trip[0],trip[1],vehicle)
    ad_wait = times[trip_data['path'][0][1]] \
                   [trip_data['path'][1][1]]
    r1 = trip_data['path'][0][0]
    r2 = trip_data['path'][1][0]  
  
    qos2 = active_requests.loc[r2]['qos']
    
    if (ad_wait + qos2) > MaxWait:
       pass
    else:
        qos1 = active_requests.loc[r1]['qos']
        tot_wait = \
              qos1 + ad_wait + qos2
        
        rtv_graph.add_edge(trip, vehicle, wait = 
             tot_wait, delay = trip_data['cost'] - tot_wait, 
             edge_type = 'tv', path = trip_data['path'])
        # add 'rt' edges
        rtv_graph.add_edge(r1,trip, edge_type = 'rt')
        rtv_graph.add_edge(r2,trip, edge_type = 'rt')
    
    return rtv_graph