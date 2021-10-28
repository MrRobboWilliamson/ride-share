#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 08:46:40 2021

@author: rob
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


class Logger:
    '''
    to pass to taxis to record events pickups and dropoffs
    '''
    def __init__(self,dump_path,stamp):
        self.all_records = list()
        self.dump_path = os.path.join(dump_path,stamp)
        
        # ensure the dump path exists
        Path(self.dump_path).mkdir(exist_ok=True)
        
    def make_log(self,time,passenger,cab,action,location=None):
        """
        Use this to record:
            - when a passenger is picked up or dropped off        
        """
        self.all_records.append(
            dict(time_stamp=time,
                 passenger_id=passenger,
                 cab_id=cab,
                 event=action,
                 location=location)
            )
        
    def dump_logs(self,day,hour):
        
        # put the data into a df, dump to a csv then clear the records to
        # save memory.
        pd.DataFrame(self.all_records).to_csv(
            os.path.join(self.dump_path,f"{day}_{hour}.csv"),index=False
            )
        self.all_records = []


def format_time(h,t):
    """
    formats the time in a more familiar way
    """
    
    # calculate the seconds and minutes from total seconds
    seconds = t%60
    minutes = (t//60)%60
    
    # if the minutes & seconds == 0, then increase
    # the hour by one
    h = h + 1 if seconds+minutes == 0 else h
    
    # return the string
    return f"{h}:{minutes:02d}:{seconds:02d}"
          
        
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
    """
    For plotting a bunch of requests as a Gantt style chart
    - not used anymore    
    """
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
    Checks for the lowest cost way to combine two requests in a shared
    trip
    - if any combination breaches the max wait or in journey delays
      then it is skipped
    - returns the corresponding path and cost
    - path is a list of tuples / False
        - each tuple is (request_id,location_id,is_pickup)
        - returns False if a feasible path can't be found 
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
                path = [(first_pickup,first_node,True),
                        (second_pickup,second_node,True),
                        (first_dropoff,third_node,False),
                        (second_dropoff,fourth_node,False)]                
                best_path = total_cost,path
        
    return best_path


def shortest_withpassenger(passenger,start_time,start_loc,
                           request_details,times,MaxWait):
    """
    Checks taxis with a passenger if it can be shared with a request
    - if any combination breaches max wait of the request or max in journey
      delays of the request / passenger, then skipped
    - returns lowest cost path if a feasible combination is found
    - returns False if no feasible combination is found
    """    

    # get the passenger current state
    pdrop_loc = passenger.drop_off_node
    
    # shortest time to drop the passenger
    # time_to_drop = times[start_loc,pdrop_loc]    
        
    # get the request details
    req_id = request_details['req_id']
    o,d = request_details['route']
    req_t = request_details['req_t']
    bjt = request_details['base']
    
    # generate options
    options = [
        [(passenger.req_id,pdrop_loc,False),(req_id,o,True),(req_id,d,False)], # drop passenger first
        [(req_id,o,True),(passenger.req_id,pdrop_loc,False),(req_id,d,False)], # pick req & drop pass
        [(req_id,o,True),(req_id,d,False),(passenger.req_id,pdrop_loc,False)]  # pick & drop req first
        ]
    
    best_path = 99999999,False    
    for combo in options:
    
        # go through the events and check the constraints
        # try to minimise the overall cost, but
        # only return the cost to the request (I think)
        current_loc = start_loc
        current_time = start_time
        cost = 0
        wait_cost = 0        
        r_pick_time = None
        for event in combo:
            # unpack the event
            r_id,loc,is_pickup = event
            
            # time to here and update location
            time_to_here = times[current_loc,loc]
            current_time += time_to_here
            current_loc = loc
            
            # check if this is a passenger drop off
            if r_id == passenger.req_id:
                
                # check the passenger delay constraint
                expected_jt = current_time - passenger.pickup_time
                if expected_jt - passenger.base_jt > MaxWait:
                    cost = float('inf')
                    break
                
                # add a cost
                cost += current_time - passenger.earliest_arrival
                
            elif is_pickup and r_id == req_id:
                
                # check the request wait constraint
                if current_time - req_t > MaxWait:
                    cost = float('inf')
                    break
                
                # set the request wait cost
                cost += current_time - req_t
                wait_cost = current_time - req_t
                r_pick_time = current_time
                
            # finally this is the request dropoff
            else:
                
                # check the journey time constraint on the request
                expected_jt = current_time - r_pick_time
                if expected_jt - bjt > MaxWait:
                    cost = float('inf')
                    break
                
                # add the delay
                cost += expected_jt - bjt
                
        # if the cost on this combo is better than the last, it's the best
        if cost < best_path[0]:
            best_path = cost,combo
            
    return wait_cost,best_path[1]
    


def add_one_req(t,rtv_graph,rv_graph,trip,vehicle,active_requests):
    """
    # add a feasible single request trip to the rtv graph
    # edge of the rv graph and no delay time for the trip. Add an
    # 'tv' edge with those two values and a 'rt' edge between the
    # request and the trip. NOTE: Trips are tuples so that multi
    # request trips can be accomodated.    
    """
    
    # get the pickup and drop off times to satisfy the trip with
    # this vehicle
    request = trip[0]
    details = active_requests.loc[request]
    wait = rv_graph.get_edge_data(request,vehicle)['cost']
        
    # the path here will just be to go from my current
    # location to pickup and drop off the request
    path = [(request,details['from_node'],True),
            (request,details['to_node'],False)]
    
    # add 'tv' edge
    rtv_graph.add_edge(trip,vehicle,wait=details['qos']+wait,
            delay=0,edge_type='tv',rnum=1,path=path)    
        
    # add 'rt' edge
    rtv_graph.add_edge(trip[0],trip,edge_type='rt')
    
    return rtv_graph


def check_one_req_one_passenger(t,Taxis,rv_graph,rtv_graph,times,active_requests,
                  vehicle,trip,MaxWait):#,MaxQLoss):
    """
    # check a one request trip with a vehicle that already has a passenger in it.
    # If the added delay to the passenger in the vehicles is within bounds add 
    # the trip to the rtv graph otherwise discard
    """    
    
    # get the passenger's details
    p1 = Taxis[vehicle].passengers[0]
    
    # get the taxi's next location and passenger's current delays
    t1,o1 = Taxis[vehicle].find_me(t)
    qos1 = p1.wait_time
    
    # get the new pick up request details
    r2 = trip[0]
    t2,o2,d2,ltp2,bjt2,qos2 = active_requests.loc[r2]
    qos2 += rv_graph.get_edge_data(r2,vehicle)['cost']
        
    # get the cost and shortest path of combining passenger and request
    cost,path = shortest_withpassenger(
                        p1,t1,o1,
                        dict(req_id=r2,route=(o2,d2),
                             req_t=t2,base=bjt2),
                        times,MaxWait
                        )  
    
    if path:
        # calculate the total cost                
        tot_wait = qos2 + qos1
        
        # add 'tv' edge
        rtv_graph.add_edge(trip,vehicle,wait=tot_wait,delay=cost,
                           edge_type='tv',rnum=1,path=path)
        
        # add 'rt' edge
        rtv_graph.add_edge(r2,trip,edge_type='rt')
    
    return rtv_graph


def check_two_req(t,rv_graph,rtv_graph,times,active_requests,
                  vehicle,trip,MaxWait):
    
    """
    # checks a trip that has two requests for feasibility and either discards it
    # or adds it to the rtv_graph.  We know the trip delay and
    # the wait time for the first pickup is within feasible 
    # limits, just need to check the additional wait time for 
    # the second pickup (time between first and second node)
    """

    # trip_data = rv_graph.get_edge_data(trip[0],trip[1],vehicle)
    rr_data = rv_graph.get_edge_data(trip[0],trip[1])
    r1 = rr_data['path'][0][0]
    r2 = rr_data['path'][1][0] 
    rv_data = rv_graph.get_edge_data(r1,vehicle)
    r1_wait_v = rv_data['cost']

    # recalc the waiting times for r1 and r2
    r2_wait_r1 = times[rr_data['path'][0][1]] \
                    [rr_data['path'][1][1]]
    qos2 = active_requests.loc[r2]['qos']
    qos1 = active_requests.loc[r1]['qos']
    
    if (r1_wait_v + qos1) > MaxWait or \
        (r2_wait_r1 + qos2 + r1_wait_v) > MaxWait:
        pass    
    else:
        total_wait = r1_wait_v + qos1 + r2_wait_r1 + qos2
        rr_wait = total_wait - r1_wait_v
        delay = rr_data['cost'] - rr_wait
        
        # r1_wait_v is the only value not in the request pair cost
        
        # all costs are stored on the T,v edge   
        rtv_graph.add_edge(trip,vehicle,
                           wait=total_wait,
                           delay=delay,
                           edge_type='tv', 
                           rnum = 2,
                           path=rr_data['path'])
        
        # add 'rt' edges
        rtv_graph.add_edge(r1,trip, edge_type = 'rt')
        rtv_graph.add_edge(r2,trip, edge_type = 'rt')
    
    return rtv_graph


def update_current_state(current_time,active_requests,Taxis,MaxWait,customers,
                         path_finder):
    """
    Purpose:
        - update cab state: pickup and dropoff passengers
        - remove picked up passengers from the active requests
        - remove passengers who's max wait is exceeded from active request
          pool
    """
    
    # loop over the cabs
    picked_up = []
    dropped_off = []
    for cab in Taxis.values():
                
        # skip if the cab is idle
        if cab.is_idle():
            continue        
        else:
            # update current state per cab and collect all the 
            # picked up and dropped off passengers
            picked,dropped = cab.update_current_state(current_time,
                                                      customers,
                                                      path_finder)
            picked_up += picked
            dropped_off += dropped
    
    # get the requests that were never assign a trip and who's
    # wait time has exceeded the MaxWait time
    ignored = active_requests[
        (current_time-active_requests['time']) > MaxWait
        ].index.values
    
    print(f"  - {len(picked_up)} passengers pickup up")
    print(f"  - {len(dropped_off)} passengers dropped off")
    print(f"  - {ignored.shape[0]} requests ignored")

    # combine the picked up and ignored passengers to remove them
    # from the active requests and the customer collections
    to_remove = set(picked_up) | set(ignored)
    for cust in to_remove:
        if cust in customers and not customers[cust]['is_rebalance']:
            customers.pop(cust) 
    return active_requests.drop(to_remove)
        

def book_trips(current_time,Trips,Taxis,active_requests,
               path_finder,rtv_graph,CabIds,customers):
    """
    This function sets trips for cabs and resets existing bookings
    - return the idle cabs and unallocated requests
    """    
    
    allocated_requests = set()
    allocated_taxis = set()
    for trip_requests,cab_id in Trips.items():
        
        # to add a trip to a cab, we need the corresponding path      
        path = rtv_graph.get_edge_data(trip_requests,cab_id)['path']
                
        # check if any requests have already been allocated
        # if so, remove the request from the old allocation
        for r in trip_requests:
            if r in customers:
                customers[r]['cab'].remove_booking(r,current_time,path_finder)
            customers[r] = dict(cab=Taxis[cab_id],is_rebalance=False)
        
        # set the trip in the new cab
        Taxis[cab_id].book_trip(
            path,
            current_time,
            active_requests.loc[list(trip_requests)],
            path_finder
            )
        
        # collect the allocated requests and cabs
        allocated_requests |= set(trip_requests)
        allocated_taxis.add(cab_id)
    
    # get the unallocated requests
    unallocated = active_requests.copy().drop(allocated_requests)
    
    # candidate idle cabs
    potentially_idle = CabIds - allocated_taxis
    
    # need to check if these cabs are actually idle
    idle = [v for v in potentially_idle if Taxis[v].is_idle()]
    
    return unallocated,idle


def balancing_act(current_time,allocations,Taxis,unallocated,path_finder,
                  idle,customers):
    """
    This function allocates unallocated requests to 
    idle vehicles
    """    
    
    allocated_requests = set()
    allocated_taxis = set()
    for cab_id,path in allocations.items():
        
        # check if any requests have already been allocated
        # if so, remove the request from the old allocation
        r = path[0][0]
        if r in customers:        
            customers[r]['cab'].remove_booking(r,current_time,path_finder)
            
        # setting is_rebalance=True stops the passenger from being picked
        # up - this is a design choice
        customers[r] = dict(cab=Taxis[cab_id],is_rebalance=True)    
        
        # set the trip in the new cab
        Taxis[cab_id].book_trip(
            path,
            current_time,
            unallocated.loc[[r],:],
            path_finder
            )
        
        # collect rebalanced requests and cabs
        allocated_requests.add(r)
        allocated_taxis.add(cab_id)
    
    # get the unallocated requests
    poor_sods = unallocated.drop(allocated_requests)
    
    # return the cabs that are still idle
    still_idle = set(idle) - allocated_taxis
    
    return poor_sods,still_idle
    
    