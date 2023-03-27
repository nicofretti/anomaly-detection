#!/usr/bin/env python3

# sudo apt-get install python3.8 python3-tk

from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import scrolledtext
import math



class LedCalculator:

    def __init__(self,tab):

        # This value depends on refresh time used by ALS module. By default is 20 ms
        self.refresh_time = 20
        self.getSelected = IntVar() 

        # Init led
        self.labelInitLed = Label(tab, text="Init led")
        self.labelInitLed.grid(column=1, row=1, sticky=W, padx=(10, 20), pady = (5,0))

        self.entryInitLed = Entry(tab, width=10)
        self.entryInitLed.grid(column=1, row=2, padx=(10, 20))
        self.entryInitLed.insert(END, '1')

        # End led
        self.labelEndLed = Label(tab, text="End led")
        self.labelEndLed.grid(column=1, row=3, sticky=W, padx=(10, 20), pady = (5,0))

        self.entryEndLed = Entry(tab, width=10)
        self.entryEndLed.grid(column=1, row=4, padx=(10, 20))
        self.entryEndLed.insert(END, '140')

        # Desired time
        self.labelDesiredTime = Label(tab, text="Desired time")
        self.labelDesiredTime.grid(column=1, row=5, sticky=W, padx=(10, 20), pady = (5,0))

        self.entryDesiredTime = Entry(tab, width=10)
        self.entryDesiredTime.grid(column=1, row=6, padx=(10, 20))
        self.entryDesiredTime.insert(END, '1000')

        # Led increment
        self.labelLedIncrement = Label(tab, text="Led increment")
        self.labelLedIncrement.grid(column=1, row=7, sticky=W, padx=(10, 20), pady = (5,0))

        self.entryLedIncrement = Entry(tab, width=10)
        self.entryLedIncrement.grid(column=1, row=8, padx=(10, 20))
        self.entryLedIncrement.insert(END, '1')

        # Output text
        self.text = scrolledtext.ScrolledText(tab ,width=40,height=10)
        self.text.place(x = 110, y = 10, width=300, height=160)

        # Radiobuton Get Interval
        self.radInterval = Radiobutton(tab, text="Get interval", value=1, variable= self.getSelected)
        self.radInterval.place(x = 120, y = 175)
        self.getSelected.set(1)

        # Radiobuttton Get Real Speed
        self.realtime = Radiobutton(tab, text="Get real speed", value=2, variable= self.getSelected)
        self.realtime.place(x = 220, y = 175)

        # Calculate button
        self.calculateButton = Button(tab, text="Calculate", command=self.calculate)
        self.calculateButton.grid(column=1,row=9, padx=(10, 20), pady = (5,0))



    def findSpeed(self,start_led, end_led, led_increment, speed):

        refresh_time = self.refresh_time

        number_of_leds = end_led - start_led + 1

        number_of_leds = math.ceil(number_of_leds/led_increment)

        # When all leds turn off to repeat the secuence. This is a new extra led 
        number_of_leds_in_system = number_of_leds + 1

        speed_per_pixel = speed / number_of_leds_in_system   

        # Buscar el multiplo de 20 mas cercano a la baja
        remainder = speed_per_pixel % refresh_time

        # Si el resto es inferior al 60% de refresh_time, redondeo a la baja,  si no, redondeo a la alta
        if remainder < refresh_time*0.6:
            speed_per_pixel_rounded = speed_per_pixel - remainder
            
        else:
            inverse_remainder = refresh_time - remainder
            speed_per_pixel_rounded = speed_per_pixel + inverse_remainder


        # Si speed_per_pixe_rounded se ha redondeado a cero, automaticamente, pasa a valer el tiempo minimo
        # disponible, es decir, refresh_time

        if speed_per_pixel_rounded > 0:
            new_speed_per_pixel = speed_per_pixel_rounded
        else:
            new_speed_per_pixel = refresh_time

        new_speed = new_speed_per_pixel * number_of_leds_in_system

        return new_speed
        
    def calculateAvailableIntervals(self):

        start_led = int(self.entryInitLed.get())
        end_led = int(self.entryEndLed.get())
        led_increment = int(self.entryLedIncrement.get())
        desiredSpeed = int(self.entryDesiredTime.get())

        # Clear text output
        self.text.delete('1.0', END)
        

        self.text.insert(END, "You must use one of the following led " + '\n' +
                        "intervals if you want to achieve a real time of " 
                            + str(desiredSpeed) + " ms in shift effect:" +  '\n' + '\n')
            
        findAnyResult = False

        for led in range(start_led, end_led+1):

            result = self.findRealTime(start_led, led, led_increment, desiredSpeed)

            if result == desiredSpeed:
                findAnyResult = True
                self.text.insert(END, "[" + str(start_led) +  ", " + str(led) + "]" + '\n')

        if not findAnyResult:

            self.text.insert(END, "No combination reaches the desired time" + '\n' + 
            "Consider modifying the conditions" + '\n')

    def roundToRefreshTime(self,speed_decimal):

        refresh_time = self.refresh_time
        speed_rounded = 0
        speed = int(speed_decimal)

        remainder = speed % refresh_time

        # Round up
        if(remainder < refresh_time*0.6):

            speed_rounded = speed - remainder
        

        # Else, rounds down
        else:

            inverse_remainder = refresh_time - remainder
            speed_rounded = int(speed + inverse_remainder)
        


        #If speed_rounded has been rounded to zero, then it is assigned the minimum possible value, 
        # that is, refresh_time

        if(speed_rounded <= 0):

            speed_rounded = refresh_time
        

        return speed_rounded
        
        

    def findRealTime(self,start_led, end_led, led_increment, speed):

        refresh_time = self.refresh_time 
    
        number_of_leds = end_led - start_led +1
    
        number_of_leds_in_system = math.ceil(number_of_leds/led_increment)
        
        speed_per_led_raw = speed / number_of_leds_in_system

        
        real_speed_per_led = self.roundToRefreshTime(speed_per_led_raw)

        real_speed = real_speed_per_led * number_of_leds_in_system

        return real_speed




    def calculateRealTime(self):

        start_led = int(self.entryInitLed.get())
        end_led = int(self.entryEndLed.get())
        led_increment = int(self.entryLedIncrement.get())
        speed = int(self.entryDesiredTime.get())

        # Clear text output
        self.text.delete('1.0', END)

        real_speed = self.findRealTime(start_led, end_led, led_increment, speed)


        self.text.insert(END, "If you want to use the LED interval:" + '\n' + '\n' +
                        "[" + str(start_led) + ", " + str(end_led) + "] " + '\n' + '\n' +
                        "The real assigned speed in shift effect will be:"  +  '\n' +  '\n' +
                        str(real_speed) + " ms " +  '\n' + '\n')

        

        return real_speed


    def calculate(self):

        if self.getSelected.get() == 1:

            self.calculateAvailableIntervals()

        if self.getSelected.get() == 2:

            self.calculateRealTime()






