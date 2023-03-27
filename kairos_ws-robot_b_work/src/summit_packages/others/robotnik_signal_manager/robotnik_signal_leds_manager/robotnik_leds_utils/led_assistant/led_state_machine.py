#!/usr/bin/env python3

# sudo apt-get install python3.8 python3-tk

from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import scrolledtext
import math
import os
import subprocess


class LedStateMachine:

    def __init__(self,tab):
        
        self.tab = tab

        self.labelState = Label(tab, text="State")
        self.labelState.place(x = 10, y = 13)

        self.comboState = StringVar()
        self.comboState.trace('w',self.comboStateUpdated)

        self.stateBox = ttk.Combobox(tab, textvariable = self.comboState)
        self.stateBox['values'] = ('Booting', 'Ready', 'Exit')
        self.stateBox.place(x = 60, y = 10, width=100, height=20)



        self.labelEffect = Label(tab, text="Effect")
        self.labelEffect.place(x = 10, y = 43)

        self.comboEffect = StringVar()
        self.comboEffect.trace('w',self.comboEffectUpdated)


        self.effectBox = ttk.Combobox(tab, textvariable = self.comboEffect)
        self.effectBox['values'] = ('Paint', 'Blink', 'Shift')
        self.effectBox.place(x = 60, y = 40, width=100, height=20)


        self.labelColor = Label(tab, text="Color")
        self.labelColor.place(x = 10, y = 73)

        self.entryRed = Entry(tab, width=4)
        self.entryRed.place(x = 60, y = 70)
        self.entryRed.insert(END, '255')

        self.entryGreen = Entry(tab, width=4)
        self.entryGreen.place(x = 110, y = 70)
        self.entryGreen.insert(END, '255')

        self.entryBlue = Entry(tab, width=4)
        self.entryBlue.place(x = 160, y = 70)
        self.entryBlue.insert(END, '255')



        self.labelLedInit = Label(tab, text="Start")
        self.labelLedInit.place(x = 10, y = 100)        
        
        self.entryLedInit = Entry(tab, width=4)
        self.entryLedInit.place(x = 60, y = 100)
        self.entryLedInit.insert(END, '1')

        self.labelLedEnd = Label(tab, text="End")
        self.labelLedEnd.place(x = 113, y = 100)        
        
        self.entryLedEnd = Entry(tab, width=4)
        self.entryLedEnd.place(x = 160, y = 100)
        self.entryLedEnd.insert(END, '140')




        self.labelMsOn = Label(tab, text="ms on")
        #self.labelMsOn.place(x = 10, y = 130)        
        
        self.entryMsOn = Entry(tab, width=6)
        #self.entryMsOn.place(x = 60, y = 130)
        self.entryMsOn.insert(END, '1000')


        self.labelMsOff = Label(tab, text="ms off")
        #self.labelMsOff.place(x = 10, y = 160)       
        
        self.entryMsOff = Entry(tab, width=6)
        #self.entryMsOff.place(x = 60, y = 160)
        self.entryMsOff.insert(END, '1000')



        self.labelSpeed = Label(tab, text="Speed")
        #self.labelSpeed.place(x = 10, y = 130)        
        
        self.entrySpeed = Entry(tab, width=6)
        #self.entrySpeed.place(x = 60, y = 130)
        self.entrySpeed.insert(END, '1000')


        self.labelDirection = Label(tab, text="Direction")
        #self.labelDirection.place(x = 10, y = 160)        
        
        self.comboDirection = StringVar()
        self.directionBox = ttk.Combobox(tab, textvariable = self.comboDirection, width=6, height=10)
        self.directionBox['values'] = ('right', 'left')
        #self.directionBox.place(x = 60, y = 160)




        # Output text
        self.text = scrolledtext.ScrolledText(tab ,width=20,height=10)
        self.text.place(x = 210, y = 10, width=200, height=120)


        # Set State button
        self.setStateButton = Button(tab, text="Set State", command=self.setStateButton)
        self.setStateButton.place(x = 210, y = 137, width=200, height=25)



        self.labelPassword = Label(tab, text="Password")
        self.labelPassword.place(x = 210, y = 173)      
        
        self.entryPassword = Entry(tab)
        self.entryPassword.place(x = 290, y = 170,  width=120, height = 20)
        self.entryPassword.insert(END, '')
        self.entryPassword.config(show="*");  


    def setStateButton(self):

        password = self.entryPassword.get()
        state = self.comboState.get().upper()
        mode = self.comboEffect.get().lower()
        color_R = self.entryRed.get() 
        color_G = self.entryGreen.get()
        color_B = self.entryBlue.get()
        color_W = 0
        start_led = self.entryLedInit.get()
        end_led = self.entryLedEnd.get()
        ms_on = self.entryMsOn.get()
        ms_off = self.entryMsOff.get()
        direction  = self.comboDirection.get()
        speed = self.entrySpeed.get()


        if ms_on == '':
            ms_on = '1000'
        if ms_off == '':
            ms_off = '1000'

        print("rosservice call /arduino_led_signaling/config/default_states" +
                  " \"{password: " "'"+ password + "'" ", state: " "'" + state + "'" ", mode: " "'" + mode + "'" "," + 
                  " color_R: " + str(color_R) + ", color_G: " + str(color_G) + ", color_B: " + str(color_B) + ", color_W: " + str(color_W) + 
                  ", start_led: " + str(start_led) + ", end_led: " + str(end_led) + ", ms_on: " + str(ms_on) + ", ms_off: " + str(ms_off) + 
                  ", direction: '" + direction + "', speed: " + str(speed) + "}\"")


        os.system("rosservice call /arduino_led_signaling/config/default_states" +
                  " \"{password: " "'"+ password + "'" ", state: " "'" + state + "'" ", mode: " "'" + mode + "'" "," + 
                  " color_R: " + str(color_R) + ", color_G: " + str(color_G) + ", color_B: " + str(color_B) + ", color_W: " + str(color_W) + 
                  ", start_led: " + str(start_led) + ", end_led: " + str(end_led) + ", ms_on: " + str(ms_on) + ", ms_off: " + str(ms_off) + 
                  ", direction: '" + direction + "', speed: " + str(speed) + "}\"")




    def comboEffectUpdated(self, index, value, op):
        
        if self.comboEffect.get() == "Paint":
            
            self.labelMsOn.place_forget()
            self.entryMsOn.place_forget()
            self.labelMsOff.place_forget()
            self.entryMsOff.place_forget()

            self.labelSpeed.place_forget()
            self.entrySpeed.place_forget()
            self.labelDirection.place_forget()
            self.directionBox.place_forget()
        

        if self.comboEffect.get() == "Blink":
            
            self.labelSpeed.place_forget()
            self.entrySpeed.place_forget()
            self.labelDirection.place_forget()
            self.directionBox.place_forget()

            self.labelMsOn.place(x = 10, y = 130)  
            self.entryMsOn.place(x = 60, y = 130)
            self.labelMsOff.place(x = 10, y = 160) 
            self.entryMsOff.place(x = 60, y = 160)
            

        if self.comboEffect.get() == "Shift":

            self.labelMsOn.place_forget()
            self.entryMsOn.place_forget()
            self.labelMsOff.place_forget()
            self.entryMsOff.place_forget()

            self.labelSpeed.place(x = 10, y = 130)  
            self.entrySpeed.place(x = 60, y = 130)
            self.labelDirection.place(x = 10, y = 160) 
            self.directionBox.place(x = 75, y = 160)
 
    
    def clearStates(self):

        self.entryRed.delete(0, END)
        self.entryGreen.delete(0, END)
        self.entryBlue.delete(0, END)
        self.entryLedInit.delete(0, END)
        self.entryLedEnd.delete(0, END)
        self.entryMsOn.delete(0, END)
        self.entryMsOff.delete(0, END)

    def comboStateUpdated(self, index, value, op):

        self.clearStates()

        if self.comboState.get() == "Booting":

            # Default values for Booting
            self.effectBox.current(1) 
            self.entryRed.insert(END, '100')
            self.entryGreen.insert(END, '100')
            self.entryBlue.insert(END, '100')
            self.entryLedInit.insert(END, '1')
            self.entryLedEnd.insert(END, '450')
            self.entryMsOn.insert(END, '500')
            self.entryMsOff.insert(END, '500')

        if self.comboState.get() == "Ready":
 
            # Default values for Ready
            self.effectBox.current(0)
            self.entryRed.insert(END, '0')
            self.entryGreen.insert(END, '255')
            self.entryBlue.insert(END, '0')
            self.entryLedInit.insert(END, '1')
            self.entryLedEnd.insert(END, '450')
        
        if self.comboState.get() == "Exit":

            # Default values for Exit
            self.effectBox.current(0)
            self.entryRed.insert(END, '0')
            self.entryGreen.insert(END, '0')
            self.entryBlue.insert(END, '255')
            self.entryLedInit.insert(END, '1')
            self.entryLedEnd.insert(END, '450')

    
    def default_values(self):

        self.stateBox.current(0)
        self.directionBox.current(0) 