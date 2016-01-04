#!/usr/bin/env python

from Tkinter import *
#from Tkinter.Message import showinfo

#def reply():
#    showinfo(title='MIOSOTYS', message='Button pressed!')

class mioGUI(Frame):
    def __init__(self, parent=None):
        Frame.__init__(self, parent)
        button = Button(self, text='Press')
        button.pack()
        
if __name__ == '__main__':
    window = mioGUI()
    window.pack()
    window.mainloop()