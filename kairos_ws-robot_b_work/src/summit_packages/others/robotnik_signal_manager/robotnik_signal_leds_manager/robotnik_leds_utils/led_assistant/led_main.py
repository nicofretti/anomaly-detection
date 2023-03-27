#!/usr/bin/env python3

# sudo apt-get install python3.8 python3-tk

from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import scrolledtext
import math

import led_calculator
import led_state_machine
import led_signals

window = Tk()

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
#tab3 = ttk.Frame(tab_control)

#tab_control.add(tab3, text='    State    ')
tab_control.add(tab1, text='    Signals    ')
tab_control.add(tab2, text='    Calc    ')
tab_control.pack(expand=1, fill='both')


w = 425 # width for the Tk root
h = 225 # height for the Tk root

# get screen width and height
ws = window.winfo_screenwidth() # width of the screen
hs = window.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = -100 + ((hs/2) - (h/2))

# set the dimensions of the screen and where it is placed
window.geometry('%dx%d+%d+%d' % (w, h, x, y))
window.title("Led Assistant")


t1 = led_state_machine.LedStateMachine(tab1)
t1.default_values()

t2 = led_calculator.LedCalculator(tab2)

#t3 = led_signals.LedSignals(tab3)
#t3.default_values()


window.mainloop()