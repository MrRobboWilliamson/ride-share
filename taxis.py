# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:03:29 2021

@author: troyf
"""

import pandas as pd
import numpy as np
from colorama import Fore,Style
delta = 30

# Excption handling classes
class LatePickupError(Exception):
    pass
class LateDropoffError(Exception):
    pass
class MultiPickupError(Exception):
    pass
class MultiDropoffError(Exception):
    pass

# for printing pickup counts
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

class Taxi():
    
    """
    Creates a taxi for our simulation
    """
    
    def __init__(self,taxi_id,max_capacity,init_loc,logger,max_delay,max_wait):
        
        # create variables
        self.taxi_id = taxi_id
        self.max_capacity = max_capacity
        self.passengers = []
        self.loc = init_loc
        self.max_delay = max_delay
        self.max_wait = max_wait
                
        # parameters for tracking trips and availability
        self.current_timetable_ = pd.DataFrame()
        self.trip_data = dict()
        
        # for recording events - this is a pointer to a central log
        self.logger = logger
        
        # flag for error checking
        self.flag = False
        
    
    def update_current_state(self,current_time,
                             customers):
        """
        
        Parameters
        ----------
        time : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """

        # get the timetable up to now for realization
        # temp = self.current_timetable_.copy()
        to_realize = self.current_timetable_[
            self.current_timetable_['time']<=current_time
            ]
        
        # drop these from the timetable
        self.current_timetable_ = \
            self.current_timetable_.drop(to_realize.index)
            
        # if the timetable is empty, set the current location
        # of the cab to the last location in the timetable
        if self.current_timetable_.empty:
            self.loc = to_realize['loc'].values[-1]
                    
        # get the pickups and dropoffs
        pickups = to_realize[to_realize['pickup'].notnull()]
        dropoffs = to_realize[to_realize['dropoff'].notnull()]
        
        # put erronous requests here to track them
        errors = [2209,289]
                  
                  # ,2983,1246,293,716,2217,2371,
                  # 1328,666,2742,1231,482]
        if self.flag or any(self.current_timetable_['pickup'].isin(errors)) or\
            any(pickups['pickup'].isin(errors)):
            
            print()
            print(self)
            print(to_realize)
            print(self.current_timetable_)
            print(self.passengers)
            # print(self.trip_data)
            self.flag = True
            print()
            if any(dropoffs['dropoff'].isin(errors)):
                self.flag = False
        
        # go through and action pickups
        for idx,time,loc,r,_ in pickups.to_records():
            
            # pickup the passenger
            passenger = self.pickup_passenger(r,loc,time)
            customers[int(r)]['passenger'] = passenger
            
            # assign the passenger object to the trip
            # data for removal at dropoff
            self.trip_data[r]['passenger'] = passenger
                
        # go though the dropoffs
        for idx,time,loc,_,r in dropoffs.to_records():
            
            # pop the trip details
            dets = self.trip_data.pop(r)
            self.dropoff_passenger(dets['passenger'],loc,time)
            
            # finally remove the customer from the customers dict
            customers.pop(int(r))
        
        # return a list of the pickups to remove from active requests.
        return list(pickups['pickup'].values),list(dropoffs['dropoff'].values)
    
 
    def get_passengers(self,time):
        
        """
        Returns the taxi's current passengers (at the next
        location in the journey)
        """
        
        # check if any of the current passengers will dropped 
        # at the next location in the cab's journey
        if not self.current_timetable_.empty:
            tt = self.current_timetable_[self.current_timetable_['time']<=time]
            drops = tt['dropoff']
            return [p for p in self.passengers if p.req_id not in drops]
        else:
            return []
    

    def get_next_dropoff(self):
        """
        returns the next passenger to be dropped off 
        """
        
        # get the current scheduled dropoffs
        drops = self.current_timetable_[
            self.current_timetable_['dropoff'].notnull()
            ]
        
        # unpack and return the time location and req_id
        time,loc,pickup,dropoff = drops.iloc[0]
        return time,int(loc),dropoff
    

    def pickup_passenger(self,r,loc,time):
        
        """
        Picks up a passenger, records when the pickup was and how long the
        passenger waited
        """
        
        # Create a passenger
        passenger = Passenger(
            req_id=r,
            pickup_node=loc,
            drop_off_node=self.trip_data[r]["to_node"],
            req_time=self.trip_data[r]["time"],
            base_jt=self.trip_data[r]["base_jt"],
            max_wait=self.max_wait,
            max_delay=self.max_delay
            )
        
        # add the passenger to the cab and pick them up
        self.passengers.append(passenger)
        
        # check that our capacity isn't breached
        assert len(self.passengers) <= self.max_capacity
        
        try:        
            passenger.pick_me_up(time)
        except LatePickupError as e:
            print(f"\nLate pickup {e}\n")  
        except MultiPickupError as e:
            print(f"\nMulti pickup {e}\n")  
        
        # record the event
        self.logger.make_log(
            time,r,self.taxi_id,action='pickup',location=loc
            )
        
        # return the passenger to add a ref to the trip details
        return passenger
                
        
    def dropoff_passenger(self,passenger,loc,time):
        
        """
        Drops off a passenger, records when the drop off was and how long the
        travel time and entire request to drop off time was
        """
        self.passengers.remove(passenger)
        
        try:
            # try to drop off the passenger to     
            passenger.drop_me_off(time)            
        except LateDropoffError as e:      
            print(f"\nLate dropoff {e}\n")  
        except MultiDropoffError as e:            
            print(f"\nMulti dropoff {e}\n")         
        
        # record the event
        self.logger.make_log(
            time,passenger.req_id,self.taxi_id,action='dropoff',location=loc
            )
    
    
    def book_trip(self,path,current_time,requests,path_finder):
        """
        PARAMETERS
        ----------
        path : list of requests and pickup / dropoff nodes
        requests : is the slice of the active requests that
            contains the requests assigned to this trip.
        path_finder : is a utils.ShortestPathFinder object that is 
            used to assign intermediate nodes in the timetable.
        
        when we set a trip, we set the timetable, and the first
        item, will be the next node

        Returns
        -------
        None.
        """
        
        # add this new trip the current trip data
        self.trip_data = {**self.trip_data,**requests.to_dict('index')}
        
        # update the timetable. drop dups incase we have overlaps
        # test without if it's slow
        time,first_node = self.find_me(current_time)                
        self.current_timetable_ = path_finder.get_timetable(
            first_node,path,time
            )
        
        
    def reset_booking(self,jobs,current_time,path_finder):
        """
        Creates a new timetable for the remaining jobs
        """

        # turn the jobs into a path
        path = []
        for i,t,loc,pickup,dropoff in jobs.to_records():
            
            # check if the dropoff is nan
            if np.isnan(dropoff):
                path.append(
                    (pickup,loc,True)
                    )
            else:
                path.append(
                    (dropoff,loc,False)
                    )
        
        # print("\nPath from reset_booking:",self)
        # print(path)
        # print()
        
        time,first_node = self.find_me(current_time)
        self.current_timetable_ = path_finder.get_timetable(
            first_node,path,time
            )
        
    
    def remove_booking(self,r,current_time,path_finder):
        """
        Removes a request from the current timetable
        
        Reset booking to go from current (next) location to
        the next event in the timetable
        """
        
        # remove the bookings from the timetable and the trip from the 
        # trip data
        self.trip_data.pop(r)        
        to_remove = self.current_timetable_[
            (self.current_timetable_['pickup']==r) |
            (self.current_timetable_['dropoff']==r)].index
        tt = self.current_timetable_.drop(to_remove)
        
        # the jobs are any row of the tt with a pickup or dropoff entry
        jobs = tt.loc[:,['pickup','dropoff']].dropna(axis=0,how='all')
        
        # if no jobs clear the timetable and update the cab location.           
        if jobs.empty:
            # if any events remain, find the next event.  
            next_time,next_loc = self.find_me(current_time)
            self.current_timetable_ = pd.DataFrame()
            self.loc = next_loc
        
        # otherwise, use the jobs and current location to rebuild the
        # timetable.
        else:
            self.reset_booking(tt.loc[jobs.index],current_time,path_finder)                                                  
                        
        
    def find_me(self,current_time):
        """
        Returns time when arriving at the next node.
        
        If the cab is idle, return the current time and location
        """
        
        # if the cab is available just return the current time and location
        if self.is_idle():            
            return current_time,self.loc

        # otherwise check the timetable for the next node
        # return the arrival time and the node
        else:
            return self.current_timetable_[
                self.current_timetable_['time']>=current_time
                ].iloc[0,:2].astype(int)
    
    def is_idle(self):
        """
        Checks if the cab has no passengers or a trip allocated        
        """
        return len(self.passengers) == 0 and self.current_timetable_.empty
    
       
    def __repr__(self):
        
        return f"<Id: {self.taxi_id}, Loc: {self.loc}>"
    

class Passenger():
    
    """
    Creates a passenger for our simulation
    """
    
    def __init__(self,req_id,pickup_node,drop_off_node,req_time,base_jt,
                 max_wait,max_delay):
        
        # create variables
        self.req_id = req_id
        self.pickup_node = pickup_node
        self.drop_off_node = drop_off_node
        self.req_time = req_time
        self.base_jt = base_jt
        self.status = 0
        self.pickup_time = -1
        self.drop_off_time = -1
        self.wait_time = 0
        self.delay_time = 0
        self.travel_time = 0
        self.max_wait = max_wait
        self.max_delay = max_delay
        
        # we check against this time for passengers in cabs
        # if the cab can be diverted to pickup another passenger
        self.earliest_arrival = req_time + base_jt
        self.latest_arrival = self.earliest_arrival + max_delay + max_wait
        
        # for pickups and drop offs check how many times
        # a passenger is picked up or dropped off
        self.num_pickups = 0
        self.num_dropoffs = 0
                
        
    def pick_me_up(self, time):
        """
        Sets the passenger's pickup time
        """
        
        # check pickup
        self.pickup_time = time
        self.wait_time = time - self.req_time
        self.num_pickups += 1
        
        # check that the wait time is less than the constraint
        try:
            assert self.wait_time <= self.max_wait
        except AssertionError:
            # add num_str in case we never see the multi drop error
            num_str = ordinal(self.num_pickups)
            message = f"{Fore.RED}{self} ({num_str}): +{self.wait_time}s"+\
                f"{Style.RESET_ALL}"
            raise LatePickupError(message)
        
        # check that we only pickup passengers once
        try:
            assert self.num_pickups <= 1
        except AssertionError:
            message = f"{Fore.RED}{self}: {ordinal(self.num_pickups)}"+\
                f"{Style.RESET_ALL}"
            raise MultiPickupError(message)
        
        
    def drop_me_off(self, time):
        """
        This function:
            - sets the passenger's drop off time
            - counts the number of times dropped off
            - checks for errors on both
        """
        
        # set parameters
        self.drop_off_time = time
        self.travel_time = self.drop_off_time - self.pickup_time
        self.delay_time = self.travel_time - self.base_jt
        self.num_dropoffs += 1
        
        # check that the delay is within the constraint
        try:           
            assert self.delay_time <= self.max_delay
        except AssertionError:
            # add num_str in case we never see the multi drop error
            num_str = ordinal(self.num_dropoffs)
            message = \
                f"{Fore.RED}{self} ({num_str}): +{self.delay_time}s"+\
                    f"{Style.RESET_ALL}"
            raise LateDropoffError(message)
        
        # check the number for dropoffs is less than or equal to 1    
        try:
            assert self.num_dropoffs <= 1
        except AssertionError:
            message = f"{Fore.RED}{self}: {ordinal(self.num_dropoffs)}"+\
                f"{Style.RESET_ALL}"
            raise MultiDropoffError(message)
            
            
    def __repr__(self):
        
        return f"<req_id: {int(self.req_id)}>"