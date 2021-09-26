# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:03:29 2021

@author: troyf
"""
delta = 30

class Taxi():
    
    """
    Creates a taxi for our simulation
    """
    
    def __init__(self, taxi_id, max_capacity,init_loc):
        
        # create variables
        self.taxi_id = taxi_id
        self.max_capacity = max_capacity
        self.passengers = []
        self.current_route = []
        self.loc = init_loc
        
        # fields for checking for availability
        self.available = True
        self.next_available_time = None
        self.next_available_loc = None
           
    def get_id(self):
        
        """
        Returns the taxi id
        """
        return self.taxi_id
    
 
    def get_passengers(self):
        
        """
        Returns the taxi's current passengers
        """
        return self.passengers


    def pickup_passenger(self, passenger, current_time):
        
        """
        Picks up a passenger, records when the pickup was and how long the
        passenger waited
        """
        self.passengers.append(passenger)
        passenger.set_status(2)
        passenger.set_pickup_time(current_time)
        passenger.set_wait_time(current_time - passenger.get_req_time())
        
                
        
    def drop_off_passenger(self, passenger, current_time):
        
        """
        Drops off a passenger, records when the drop off was and how long the
        travel time and entire request to drop off time was
        """
        self.passengers.remove(passenger)
        passenger.set_drop_off_time(current_time)
        passenger.set_status(3)
        passenger.set_travel_time(current_time - passenger.get_pickup_time())


    def get_current_route(self):
        
        """
        Returns the taxi's current route
        """
        return self.current_route
    
    
    def set_current_route(self, route):
        
        """
        Sets the taxi's current route
        """
        self.current_route.append(route)
        
    def toggle_availability(self):
        """
        toggles availability status
        """
        self.available = not(self.available)
        
        
    def __repr__(self):
        
        return f"<Id: T{self.taxi_id}, Loc: {self.loc}>"
    

class Passenger():
    
    """
    Creates a passenger for our simulation
    """
    
    def __init__(self, req_id, pickup_node, drop_off_node, req_time, base_jt,
                 init_wait):
        
        # create variables
        self.req_id = req_id
        self.pickup_node = pickup_node
        self.drop_off_node = drop_off_node
        # self.req_day = req_day
        self.req_time = req_time
        self.base_jt = base_jt
        self.status = 0
        self.pickup_time = -1
        self.drop_off_time = -1
        self.wait_time = 0
        self.travel_time = 0
     
        
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
        self.pickup_time = time
        
        
    def get_drop_off_time(self):
        """
        Gets the passenger's drop off time
        """
        return self.drop_off_time
        
        
    def set_drop_off_time(self, time):
        """
        Sets the passenger's drop off time
        """
        self.drop_off_time = time


    def get_wait_time(self):
        """
        Gets the passenger's wait time
        """
        return self.wait_time
        
        
    def set_wait_time(self, time):
        """
        Sets the passenger's wait time
        """
        self.wait_time = time
        
    
    def get_travel_time(self):
        """
        Gets the passenger's travel time
        """
        return self.travel_time
        
        
    def set_travel_time(self, time):
        """
        Sets the passenger's travel time
        """
        self.travel_time = time
        
        
if __name__ == "__main__":
    t1 = Taxi(1,2)
    t2 = Taxi(2,2)
    p1 = Passenger(1, 3, 9, 0, 650, 5)
    p3 = Passenger(2, 6, 10, 0, 651, 10)
    p4 = Passenger(3, 8, 3, 0, 652, 32)
