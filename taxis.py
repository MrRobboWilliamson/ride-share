# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:03:29 2021

@author: troyf
"""

import pandas as pd

delta = 30

class Taxi():
    
    """
    Creates a taxi for our simulation
    """
    
    def __init__(self,taxi_id,max_capacity,init_loc,logger,max_delay,max_wait):
        
        # create variables
        self.taxi_id = taxi_id
        self.max_capacity = max_capacity
        self.passengers = []
        self.current_route = []
        self.loc = init_loc
        self.max_delay = max_delay
        self.max_wait = max_wait
                
        # parameters for tracking trips and availability
        self.current_timetable_ = pd.DataFrame()
        self.trip_data = dict()
        
        # for recording events - this is a pointer to a central log
        self.logger = logger
        
           
    def get_id(self):
        
        """
        Returns the taxi id
        """
        return self.taxi_id
    
    def update_current_state(self,current_time):
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
        to_realize = self.current_timetable_[
            self.current_timetable_['time']<=current_time
            ]
        
        # drop these from the timetable
        self.current_timetable_ = \
            self.current_timetable_.drop(to_realize.index)
            
        # if the timetable is empty, set the current location
        # of the cab to the last location in the timetable
        if self.current_timetable_.empty:
            self.loc = to_realize['loc'][-1]
        
        # get the pickups and dropoffs
        pickups = to_realize[to_realize['pickup'].notnull()]
        dropoffs = to_realize[to_realize['dropoff'].notnull()]
        
        # go through the pickups
        for idx,time,loc,r,_ in pickups.to_records():
                        
            # pickup the passenger
            passenger = self.pickup_passenger(r,loc,time)
            self.trip_data[r]['passenger'] = passenger
                
        # go though the dropoffs
        for idx,time,loc,_,r in dropoffs.to_records():
            
            # pop the trip details
            #  - which includes a ref to the passenger
            dets = self.trip_data.pop(r)
            self.dropoff_passenger(dets['passenger'],loc,time)
        
        # return a list of the pickups to remove from active requests.
        return list(pickups['pickup'].values)
    
 
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
        
        # add the passenger to the cab
        self.passengers.append(passenger)
        passenger.set_status(2)
        passenger.set_pickup_time(time) # this will check the wait constraint
        
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
        passenger.set_drop_off_time(time)
        passenger.set_status(3)
        
        # record the event
        self.logger.make_log(
            time,passenger.req_id,self.taxi_id,action='dropoff',location=loc
            )


    def get_current_route(self):
        
        """
        Returns the taxi's current route
        """
        return self.current_route
    
    
    def set_trip(self,path,current_time,requests,path_finder):
        """
        PARAMETERS
        ----------
        path : list of requests and pickup / dropoff nodes
        
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
        self.current_timetable_ = self.current_timetable_.append(
            path_finder.get_timetable(
                first_node,path,time,requests
                )
            ).drop_duplicates().reset_index(drop=True)
                        
        
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
        
        return f"<Id: T{self.taxi_id}, Loc: {self.loc}>"
    

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
        
    def get_id(self):
        
        """
        Returns the passenger request id
        """
        return self.req_id
    
    
    def get_total_delay(self):
        return self.wait_time + self.get_travel_delay()
    
    
    def get_travel_delay(self):        
        return max(0,self.travel_time - self.base_jt)
    
 
    def get_pickup_node(self):
        
        """
        Returns the passenger's pickup node
        """
        return self.pickup_node
   
    
    def get_drop_off_node(self):
        
        """
        Returns the passenger's drop off node
        """
        return self.drop_off_node
    
    
    def get_req_day(self):
        
        """
        Returns the passenger's request day
        """
        return self.req_day
    
    
    def get_req_time(self):
        
        """
        Returns the passenger's time
        """
        return self.req_time

    
    def get_expected_duration(self):
        
        """
        Returns the passenger's expected ride time
        """
        return self.expected_duration
    
    
    def get_status(self):
        
        """
        Returns the passenger's status
            0 = Waiting for assignment
            1 = Assigned and waiting for pickup
            2 = Picked up and in Taxi
            3 = Dropped off
        """
        return self.status
    
    
    def set_status(self, new_status):
        
        """
        Sets the passenger's status
            0 = Waiting for assignment
            1 = Assigned and waiting for pickup
            2 = Picked up and in Taxi
            3 = Dropped off
        """
        self.status = new_status
        
        
    def get_pickup_time(self):
        """
        Gets the passenger's pickup time
        """
        return self.pickup_time
        
        
    def set_pickup_time(self, time):
        """
        Sets the passenger's pickup time
        """
        
        # check pickup
        assert time - self.req_time <= self.max_wait
        
        # assign pickup and wait times        
        self.pickup_time = time
        self.wait_time = time - self.req_time
        
        
    def get_drop_off_time(self):
        """
        Gets the passenger's drop off time
        """
        return self.drop_off_time
        
        
    def set_drop_off_time(self, time):
        """
        Sets the passenger's drop off time
        """
        
        # check dropoff
        # set the final travel time
        self.travel_time = time - self.pickup_time
        
        # check this against the base journey time
        assert self.travel_time - self.base_jt <= self.max_delay
        self.delay_time = self.travel_time - self.base_jt

        # finally set the drop off time        
        self.drop_off_time = time


    def get_wait_time(self):
        """
        Gets the passenger's wait time
        """
        return self.wait_time
        
    
    def get_travel_time(self):
        """
        Gets the passenger's travel time
        """
        return self.travel_time
        
        
# if __name__ == "__main__":
#     t1 = Taxi(1,2)
#     t2 = Taxi(2,2)
#     p1 = Passenger(1, 3, 9, 0, 650, 5)
#     p3 = Passenger(2, 6, 10, 0, 651, 10)
#     p4 = Passenger(3, 8, 3, 0, 652, 32)
