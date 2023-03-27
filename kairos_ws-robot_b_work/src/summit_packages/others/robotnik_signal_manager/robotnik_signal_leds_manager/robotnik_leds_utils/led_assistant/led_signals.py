#!/usr/bin/env python3

# sudo apt-get install python3.8 python3-tk

from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import scrolledtext
import math
import os
from tkinter.filedialog import *
import getpass
import pickle

class LedSignals:

    def __init__(self,tab):

        self.filename = "Untitled.yaml"
        self.directoryPath = "/"
        
        self.labelWorkspace = Label(tab, text="Robot config")
        self.labelWorkspace.place(x = 10, y = 13)

        self.entryWorkspace = Entry(tab, width = 30)
        self.entryWorkspace.place(x = 100, y = 13)

        self.openDirectoryButton = Button(tab, text="...", command=self.openDirectory)
        self.openDirectoryButton.place(x = 350, y = 10, width=30, height=25)


        self.rosparamButton = Button(tab, text="Update rosparams", command=self.rosparamUpdate)
        self.rosparamButton.place(x = 450, y = 10, width=200, height=25)


        self.leftFrame = Frame(tab)
        self.leftFrame.grid(row=1, column=1)
        self.leftFrame.place(y = 50, relwidth=0.5, relheight=1.0)

        self.text = Text(self.leftFrame)
        self.text.pack(expand=True, fill='both')
        

        self.rightFrame = Frame(tab)
        self.rightFrame.grid(row=1, column=1)
        self.rightFrame.place(y = 50, relx = 0.5, relwidth=0.5, relheight=1.0)

        self.text_2 = Text(self.rightFrame)
        self.text_2.pack(expand=True, fill='both')


        self.initFlag = True

        self.loadDirectory()

        # # Set State button
        # self.newFileButton = Button(tab, text="New File", command=self.newFile)
        # self.newFileButton.place(x = 210, y = 137, width=200, height=25)

        # self.openFileButton = Button(tab, text="Open File", command=self.openFile)
        # self.openFileButton.place(x = 210, y = 107, width=200, height=25)

        # self.saveFileButton = Button(tab, text="Save File", command=self.saveFile)
        # self.saveFileButton.place(x = 210, y = 77, width=200, height=25)

        # self.saveAsFileButton = Button(tab, text="Save As", command=self.saveAs)
        # self.saveAsFileButton.place(x = 210, y = 47, width=200, height=25)

    def rosparamUpdate(self):

        print("aa")
        print("rosparam load " + self.directoryPath + "/signals.yaml /leds_driver")
        os.system("rosparam load " + self.directoryPath + "/signals.yaml /leds_driver")

    def newFile(self):

        self.filename = "Untitled.yaml"
        self.text.delete('1.0', END)

    def saveFile(self):

        if not self.initFlag:

            t = self.text.get('1.0', END)
            f = open(self.filename, 'w')
            f.write(t)
            f.close()
        else:
            self.saveAs()

    def saveAs(self):

        f = asksaveasfile(mode='w', defaultextension = '.yaml')
        t = self.text.get('1.0',END)

        try:
            self.filename = f.name
            self.initFlag = False

            try:
                f.write(t.rstrip())
            except:
                showerror(title="Ooops!", message = "Unable to save file...")

        except:
            self.initFlag = True


    def openFile(self):

        f = askopenfile(mode ='r')
        t = f.read()
        self.text.delete('1.0',END)
        self.text.insert(END,t)
    
    def saveDirectory(self, last_directory):

        saveVector = [last_directory]
        pickle.dump(saveVector, open("last_path", "wb"))

    def loadDirectory(self):

        saveVector = pickle.load(open("last_path", "rb"))
        
        return saveVector


    def openDirectory(self):

        initialDirectory = '/'

        if  self.loadDirectory() is None:

            username = getpass.getuser()
            initialDirectory = '/home/'+username

        else:
            initialDirectory = self.loadDirectory()

            print(initialDirectory)

        self.directoryPath = askdirectory(initialdir=initialDirectory, title='Select robot config')

        self.saveDirectory(self.directoryPath)
       
        self.entryWorkspace.delete(0,END)
        self.entryWorkspace.insert(END, self.directoryPath)
        self.entryWorkspace.xview(END) 


        f1 = open(self.directoryPath + '/signals.yaml')
        t1 = f1.read()
        self.text.delete('1.0',END)
        self.text.insert(END,t1)

        f2 = open(self.directoryPath + '/leds_config.yaml')
        t2 = f2.read()
        self.text_2.delete('1.0',END)
        self.text_2.insert(END,t2)
        

    def default_values(self):

        self.entryWorkspace.delete(0,END)
        self.entryWorkspace.insert(END, self.loadDirectory())
        self.entryWorkspace.xview(END) 	
        