# -*- coding: utf-8 -*-
"""
This is the main model
"""
import numpy as np
import pandas as pd
import time
from datetime import datetime
import networkx as nx
from numbers import Number
import itertools

# our modules
from source_data import JourneyTimes, Requests
from taxis import Taxi
from utils import (
    Logger,assign_basejt,shortest_path,format_time,check_two_req,add_one_req,
    check_one_req_one_passenger,book_trips,update_current_state,
    shortest_withpassenger,balancing_act,is_cost_error
    )                   
from allocate import (
    create_ILP_data_v2,allocate_trips_v2,greedy_assignment,rebalance
    )

# cost error
class CostError(Exception):
    pass

np.random.seed(16)

### Shareability parameters ###
k = 2                                # vehicle capacity (passengers)
delta = 30                           # size of time windows (s)
MaxWait = 120                        # max wait and in-journey delay times (s)
M = 3000                             # number of vehicles (fleet size)
day_trans = {
    0:0,1:1,2:1,3:1,4:1,5:1,6:2      # to translate day of week to Sat,Sun,Wd
    }

### Target period ###
# days 0-6 for Sunday 5 May to Saturday 11 May
# hours 0-23
lower_bound = (5,19) # start Friday, 7pm
upper_bound = (5,21) #  stop Friday, 9pm

### Dates and days for printing ###
DAY_NAMES = {i: daynm for i,daynm in enumerate(
    ["Sunday",
     "Monday",
     "Tuesday",
     "Wednesday",
     "Thursday",
     "Friday",
     "Saturday"])
    }
MY = "/5/2013"
DATES = {i: str(i+5)+MY for i in range(len(DAY_NAMES))}

### Sets ###
V = range(M)
H = range(24)
D = range(7)

### Ride Request data ###
# - We have downloaded the May 2013 Yellow Taxi Cab data for NYC
# - Filtered for the week starting Sunday 5 May to end Saturday May 11
# - Filtered for requests within 100m of an intersection
# - Applied a nearest neighbour analysis to get the closest intersection to
#   each request
# - Requests that start and end at the same node are also removed
# - The Requests class returns the ride requests assigned in windows of size
#   delta
rdf = Requests(delta,MaxWait).read_requests()

### Journey time data ###
# - The JourneyTimes class is used to retrieve the od journey times matrix
#   for a given day and hour (days are 0, 1, 2 for Sun, Weekday, Sat)
# - It also contains a method to return a ShortestPathFinder class that
#   reads in individual street journey times to calculate the intermediate
#   nodes, this is required to "find" cabs at any given time interval when
#   generating the rv graph
jt = JourneyTimes()

### Data logging ###
# - Generate a time stamp to keep track of output_log versions
# - Logger is provided to the Taxi class to record pickups and dropoffs
stamp = datetime.now().strftime('%y%m%d%H%M%S')
event_logger = Logger(r'output_logs',stamp)

### Sample initial taxi locations ###
# - Drop off locations from 2 hours prior to start time
prestart_period = 2 # hours
prestart_lower = 24*3600*lower_bound[0]+(lower_bound[1]-prestart_period)*3600
prestart_upper = 24*3600*lower_bound[0]+lower_bound[1]*3600
pre_start = rdf[
    (rdf['time']>=prestart_lower) &
    (rdf['time']<=prestart_upper)
    ]
init_locs = pre_start['to_node'].values
np.random.shuffle(init_locs)
init_locs = init_locs[:M]

### Taxis ###
#  - instantiate cabs at the initial locations
CabIds = { f"v{v}" for v in V }
Taxis = { f"v{v}": Taxi(f"v{v}",k,init_locs[v],event_logger,MaxWait,MaxWait) 
         for v in V }
    
def build_shareable(t,requests,times,rv_graph):        
    """
    Searches each feasible pair of requests for the shortest path
    between them
     - t is the current time
     - requests is a dataframe of requests
     - times is the od journey time matrix
     - rv_graph is an NetworkX undirected graph class
     - we 
    """
    
    # 
    orequests = requests.copy()
    for r1,t1,o1,d1,ltp1,bjt1,qos1 in requests.to_records():         
        
        # pop this request off the list
        orequests = orequests.drop(r1)
        
        # only check feasible pairs
        jt12 = times[o1,:]
        jt21 = times[:,o1]
        to_check = orequests[
            (jt12[orequests['from_node']]+orequests['qos']<=MaxWait) |
            (jt21[orequests['from_node']]+qos1<=MaxWait)
            ]
                
        for r2,t2,o2,d2,ltp2,bjt2,qos2 in to_check.to_records():
            
            # get the shortest path to satisfy a pair of requests
            pair = dict([(r1,dict(route=(o1,d1),other=r2,wait=qos1,base=bjt1)),
                         (r2,dict(route=(o2,d2),other=r1,wait=qos2,base=bjt2))])
            
            # only assess once
            if not rv_graph.has_edge(r1,r2):
                
                cost,path = shortest_path(pair,times,MaxWait)
                                
                if path:             
                    # the key of the 'rr' graph is in the order of the 
                    # best to pickup first 
                    
                    ### comment this out to check costs ###
                    # check cost against pair
                    # if is_cost_error(cost,path,pair,times,show=True):
                    #     raise CostError("found one!")
                    
                    rv_graph.add_edge(r1,r2,cost=cost,path=path)
                    
            else:
                # if we've already assessed this pair
                # we just need to update the costs with the 
                # additional waiting time
                rv_graph.edges[r1,r2]['cost'] += 2*delta
                
    return rv_graph
            

def create_rv_graph(t,requests,times):
    """
    Creates the RV graph
    
    Step 1:
        - update the pairwise shareability graph
        
    Step 2:
        - check which vehicles can be assigned to which requests
        - need to be able to assign requests to cabs with one availabe
          seat en route    
    """
        
    # rv graph
    rv_graph = nx.Graph()
    
    # calculate the initial wait time for all requests
    requests['qos'] = t - requests['time'].values
    
    # returns the rv graph with all of the pairwise shareable
    # requests
    start = time.process_time()    
    rv_graph = build_shareable(t,requests,times,rv_graph)
    end = time.process_time()
    print(f"  - build_shareable function {end-start:0.1f}s")
        
    # Check which taxis can service which requests
    for v in Taxis:
                
        # Get the taxi object
        cab = Taxis[v]
        
        # check where the cab will be next and the time to get there
        arrive_time,next_loc = cab.find_me(t)                
        time_to_next_loc = arrive_time - t
        
        # potential requests based on the time from the cab's next location
        # to each request (me is the cab)
        jt_from_me = times[next_loc,:]
        potentials = requests[
            requests['qos']+time_to_next_loc + 
            jt_from_me[requests['from_node']]<=MaxWait
            ]
        
        # if there's not potential pickups, no point checking anying
        # just skip this taxi
        if potentials.empty:
            continue
        
        # check how many passengers will be in the cab at it's
        # next location
        passengers = cab.get_passengers(t)
        
        # if there are no passengers in the cab just create the 
        # rv edges
        if len(passengers) == 0:
                              
            # assign edges between cab and requests with the journey
            # time as the value - don't assign the initial qos - 
            # it is assigned in the shortest path function on the rr edge
            for i,delay in enumerate(jt_from_me[potentials['from_node']]):
                rv_graph.add_edge(
                    potentials.index[i],
                    v,cost=delay+time_to_next_loc
                    )
                
        # if there are passengers on board, check where the cab will be
        # and check with every passenger if picking up the available
        # request will breach their constraints
        else:
                        
            # if the cab is full at this point we can only service
            # a request after the next drop off
            #   - (not checking combinations, just assume next is first)
            if len(passengers) == k:
                
                # get the next drop off
                drop_time,drop_loc,drop_id = cab.get_next_dropoff()
                
                # which requests can we service after dropping off 
                time_to_drop = drop_time - t
                jt_from_drop = times[drop_loc,:]                
                potentials = potentials[
                    potentials['qos'] + time_to_drop + 
                    jt_from_drop[potentials['from_node']]<=MaxWait
                    ]
                
                # if no potentials after the first drop, move on to the
                # next cab
                if potentials.empty:
                    continue
            
                # otherwise, remove this passengner from the list to check
                # the others (for k > 2)
                passengers = [p for p in passengers if p.req_id != drop_id]
                
                # update next location to this dropoff point
                arrive_time = drop_time
                next_loc = drop_loc
            
            # check if the remaining passenger and each request can be combined
            for p in passengers:               
                
                for r,req_t,o,d,ltp,bjt,qos in potentials.to_records():
                                    
                    # check combinations
                    cost,path = shortest_withpassenger(
                        p,arrive_time,next_loc,
                        dict(req_id=r,route=(o,d),
                             req_t=req_t,base=bjt),
                        times,MaxWait
                        )
                    
                    # if there is a valid combination, add the edge with the 
                    # cost.
                    if path:
                        
                        # add the edge with the expected cost to the request
                        # and the path to satisfy them, remove qos as
                        # this live in the rr edge
                        rv_graph.add_edge(r,v,cost=cost-qos,
                            path=path
                            )
                    
    return rv_graph,requests

def create_rtv_graph(t,rv,active_requests,times,MaxWait):
    """
    Explore the cliques of the rv graph to find feasible trips
    """
    rtv_graph = nx.Graph()
    for clique in nx.algorithms.clique.find_cliques(rv_graph):
    
        ### Un-comment to print larger cliques ###    
        # if len(clique) > 4:    
            
        #     # get the sub graph
        #     subg = rv.subgraph(clique)
            
        #     # get the node positions
        #     pos = nx.networkx.kamada_kawai_layout(subg)
            
        #     # draw the graph
        #     fig,ax = plt.subplots(figsize=(10,6))
        #     colors = ['tab:orange' if type(n) == str else 'tab:blue' for n in list(subg.nodes)]  
        #     nx.draw(subg,node_size=3000,with_labels=True,node_color=colors,
        #             alpha=0.9,width=2,style=":",ax=ax,font_color='white')         
                        
        #     ax.set_xlim([1.2*x for x in ax.get_xlim()])
        #     ax.set_ylim([1.2*y for y in ax.get_ylim()])
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(r"results/clique.jpg")
        
        # If the clique contains only Int64s and no Strings then it
        # is a number of requests that no vehicle can service, so we
        # ignore them for now, to be picked up in rebalancing
        if all(isinstance(i, type(clique[0])) for i in clique[1:]):
            pass
        else:
            # Sort the clique in request order with vehicle at the end       
            clique = sorted(clique, key=lambda x: (x is not None, '' 
                   if isinstance(x, Number) else type(x).__name__, x))
                                
            # Get the vehicle name, number of passengers in the vehicle,
            # trip and the number of requests in the trip
            vehicle = clique[-1]
            num_passengers = len(Taxis[vehicle].get_passengers(t))
            trip = tuple(clique[:-1])                    
            reqs = len(trip)
                        
            # Start with the "one request + empty vehicle" trips
            # which will always be feasible because we checked in the rv
            # graph creation.
            if reqs == 1 and num_passengers == 0:
                
                rtv_graph = add_one_req(t,rtv_graph,rv_graph,trip, 
                                        vehicle,active_requests)
                                       
            # then deal with the "one request + non-empty vehicle" trips.
            # This trip will have a delay > 0 due to deviations
            # for both passengers
            elif reqs == 1 and num_passengers == 1:
                
                rtv_graph = check_one_req_one_passenger(t,Taxis,rv_graph
                            ,rtv_graph,times,active_requests,vehicle,
                            trip,MaxWait)
                                           
            # finally deal with the "more than one request" trips
            else:
                # if the vehicle is empty we add all the individual
                # requests in the clique as seperate, one request trips 
                # and check all the pairwise shared request trips
                if num_passengers == 0:
                    for r in trip:
                        sub_trip = tuple([r])
                        rtv_graph = add_one_req(t,rtv_graph,rv_graph, 
                                    sub_trip,vehicle,active_requests)
                    
                    paired_trips = itertools.combinations(trip,2)
                    
                    for pair in paired_trips:
                        rtv_graph = check_two_req(t,rv_graph,rtv_graph,
                                times,active_requests,vehicle,
                                pair,MaxWait)
                
                # if the vehicle has one passenger we check all the
                # individual requests to see if they can fit with
                # the current passenger
                elif num_passengers == 1:
                    for r in trip:
                        sub_trip = tuple([r])
                        rtv_graph = check_one_req_one_passenger(t,Taxis,
                                rv_graph,rtv_graph,times,
                                active_requests,vehicle,sub_trip,
                                MaxWait)
                        
                # otherwise we have a full vehicle and so none of
                # these requests can be serviced by this vehicle
                else:
                    pass
                
    return rtv_graph 


if __name__ == "__main__":
    
    # Dataframe for adding / removing requests in the request pool
    active_requests = pd.DataFrame()

    # This collection is used to shuffle customer bookings
    # between cabs
    customers = dict() 

    # This collection is used to capture computational statistics
    computation_log = []
    
    # Cycle in hour chunks, so we don't have to check to load
    # new journey times each iteration we only load them on the hour
    for d in D:
        
        # only asses from lower bound onwards
        if d < lower_bound[0]:
            continue

        # convert days to sun, weekday, sat
        dt = day_trans[d]

        for h in H:
            
            # only assess from lower bound onwards
            if d <= lower_bound[0] and h < lower_bound[1]:
                continue

            # get the journey times matrix for this day,hour
            times,path_finder = jt.get_hourofday(dt,h)

            # filter requests in this period and assign there base journey
            # times - precalc'd shortest path between origin - destination
            # group by time windows
            req_slice = assign_basejt(rdf[(rdf['day']==d)&(rdf['hour']==h)],
                                      times)
            req_byw = req_slice.groupby('window')

            # Cycle through 30s windows and step through the simulation loop
            for w,new_requests in req_byw:

                # generate a new computation record for this window
                comp_record = dict()

                # Calculate the current time from the window
                # value
                t = w*delta + delta
                
                comp_record['t'] = t
                comp_record['num_new_requests'] = new_requests.shape[0] 
                
                start_str = f" Current time t={t}s ({format_time(h,t)},"+\
                    f" {DAY_NAMES[d]} {DATES[d]}) "
                print(f"\n{start_str:-^80}\n")                
                
                # step 0: based on current time, process in-flight trips to remove
                # passengers from cabs if they've finished their journey
                # and pickup passengers if they've been picked up
                print("Updating current state")
                start_update_state = time.process_time()

                # add on the new requests to the request pool
                active_requests = active_requests.append(
                    new_requests.drop(['window','day','hour'],axis=1)
                    )
                
                comp_record['num_current_requests'] = active_requests.shape[0]

                # process pickups and drop offs - return remaining requests
                active_requests = update_current_state(t,
                                                       active_requests,
                                                       Taxis,
                                                       MaxWait,
                                                       customers,
                                                       path_finder)
                
                end_update_state = time.process_time()
                print(f"  - processing time = {end_update_state-start_update_state:0.1f}s\n")
                comp_record['state_update_time'] = end_update_state-start_update_state

                ### Generate RV & RTV graphs ###

                print(f"RV performance {active_requests.shape[0]} requests:")
                start_rv = time.process_time()
                
                # step 1: create the rv graph
                rv_graph,active_requests = create_rv_graph(
                    t,active_requests.copy(),
                    times
                    )
                
                end_rv = time.process_time()
                comp_record['rv_compute_time'] = end_rv-start_rv
                comp_record['num_rv_edges'] = len(list(rv_graph.edges))
                
                print(f"  - number of edges: {len(list(rv_graph.edges))}")
                print(f"  - processing time: {end_rv-start_rv:0.1f}s\n")
                print("RTV performance:")
                start_rtv = time.process_time()
                
                # step 2: explore complete - subregions of the rv graph for
                # cliques
                rtv_graph = create_rtv_graph(t,rv_graph,active_requests,times,MaxWait)    
                
                end_rtv = time.process_time()
                print(f"  - number of edges: {len(list(rtv_graph.edges))}")
                print(f"  - processing time: {end_rtv-start_rtv:0.1f}s\n")
                comp_record['rtv_compute_time'] = end_rtv-start_rtv
                comp_record['num_rtv_edges'] = len(list(rtv_graph.edges))

                ### Main optimization ###
                
                print("Greedy Assignment performance:")
                start_greedy = time.process_time()
                
                # greedy assignment for optimization
                rtv_graph = greedy_assignment(rtv_graph, k)
                
                end_greedy = time.process_time()
                print(f"  - processing time: {end_greedy-start_greedy:0.1f}s\n")
                comp_record['greedy_time'] = end_greedy-start_greedy

                print("Create data performance:")            
                start_data = time.process_time()
                
                # generate data for ILP
                V, R, T, VT, RT, TV = create_ILP_data_v2(rtv_graph)
                
                end_data = time.process_time()
                print(f"  - processing time: {end_data-start_data:0.1f}s\n")
                comp_record['data_gen_time'] = end_data-start_data

                print("Trip ILP assignment:")            
                num_main_ilp_variables = sum(len(t) for t in VT.values())
                num_main_ilp_variables += len(R)
                num_main_ilp_constraints = len(R) + len(V) + len(T)            
                print(f"  - {num_main_ilp_variables} ILP variables")    
                print(f"  - {num_main_ilp_constraints} ILP constraints")
                start_assign = time.process_time()
                
                # run ILP and get the trip assignments
                Trips = allocate_trips_v2(V, R, T, VT, RT, TV,
                                          suppress_output=True)
                
                end_assign = time.process_time()
                print(f"  - number of trips assigned: {len(Trips)}")
                print(f"  - processing time: {end_assign-start_assign:0.1f}s\n")
                comp_record['num_main_ilp_vars'] = num_main_ilp_variables
                comp_record['num_main_ilp_constr'] = num_main_ilp_constraints
                comp_record['main_ilp_time'] = end_assign-start_assign
              
                print("Booking assigned trips with cabs:")
                start_pro_ass = time.process_time()
                
                # add trips to cabs as "bookings" and removes old bookings 
                # return unallocated requests and idle cabs for rebalancing
                unallocated,idle = book_trips(
                    t,Trips,Taxis,active_requests,path_finder,rtv_graph,
                    CabIds,customers
                    )
                
                end_pro_ass = time.process_time()
                print(f"  - {unallocated.shape[0]}/{active_requests.shape[0]} unallocated requests")
                print(f"  - {len(idle)}/{M} idle cabs")
                print(f"  - processing time {end_pro_ass-start_pro_ass:0.1f}s")
                comp_record['num_trips'] = len(Trips)
                comp_record['booking_time'] = end_pro_ass-start_pro_ass

                ### Rebalancing optimization ###              
    
                print("\nRebalancing idle vehciles:")
                num_bal_ilp_vars = len(idle)*unallocated.shape[0]
                num_bal_ilp_constr = len(idle)+unallocated.shape[0]+1
                print(f"  - {num_bal_ilp_vars} ILP variables")
                print(f"  - {num_bal_ilp_constr} ILP constraints")
                start_rebalance = time.process_time()
                
                # allocate unallocated trips idle cabs
                rebalanced = rebalance(idle,unallocated,times,Taxis,
                                      suppress_output=True)
                               
                end_rebalance = time.process_time()
                print(f"  - {len(rebalanced)} redirected vehicles")
                print(f"  - processing time {end_rebalance-start_rebalance:0.1f}s")
                comp_record['num_bal_ilp_vars'] = num_bal_ilp_vars
                comp_record['num_bal_ilp_constr'] = num_bal_ilp_constr
                comp_record['bal_ilp_time'] = end_rebalance-start_rebalance
                comp_record['num_rebalanced'] = len(rebalanced)

                print("\nBooking unallocated requests with idle cabs")
                start_balancing_act = time.process_time()
                
                # apply allocations as "ghost bookings" - these will never be
                # picked up
                poor_sods,still_idle = balancing_act(t,
                                                     rebalanced,
                                                     Taxis,
                                                     unallocated,
                                                     path_finder,
                                                     idle,
                                                     customers)
                
                end_balancing_act = time.process_time()
                print(f"  - {len(still_idle)} vehicles could not be rebalanced")
                print(f"  - {poor_sods.shape[0]} requests could not be rebalanced")
                print(f"  - processing time {end_balancing_act-start_balancing_act:0.1f}s")
                comp_record['rebalance_booking_time'] = end_balancing_act-start_balancing_act

                # add the computation record
                computation_log.append(comp_record)            

            # dump the logs for the hour
            event_logger.dump_logs(d,h)
            
            # break after upper limit reached
            if d >= upper_bound[0] and h >= upper_bound[1]-1:
                break
            
        # break after break after upper limit reached
        if d >= 5 and h >= upper_bound[1]-1:
            break            

    # dump the computation logs
    computation_log = pd.DataFrame(computation_log)
    computation_log['MaxWait'] = MaxWait
    computation_log.to_csv(f'output_logs/{stamp}/computation_log.csv',index=False)