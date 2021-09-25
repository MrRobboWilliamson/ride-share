#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 08:46:40 2021

@author: rob
"""
from IPython.display import clear_output, display

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
        
    