import tkinter,tkinter.constants
from tkinter import filedialog
from tkinter import * 
from tkinter import Frame
import os
root = Tk() 
frame=Frame(root, width=300, height=100)
frame.pack()
root.title('Auto_label')
button_opt = {'fill': tkinter.constants.BOTH, 'padx': 5, 'pady': 5}
img_dirname=''
def openDirectory():
	img_dirname = filedialog.askdirectory(parent=root, initialdir=os.getcwd(), title='Select your image folder')
	
Button(root, text = 'Select your image folder', fg = 'black', command= openDirectory).pack()
model_dirname = ''
def openFile():
	model_dirname = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title='Select your model folder')
	
Button(root, text = 'Select your model folder', fg = 'black', command= openFile).pack()
def code():
	os.system('python get_inference.py '+img_dirname+' '+model_dirname)
	Label(root, text = "done").pack()

Button(root, text = 'submit for annotation', fg = 'black', command= code).pack()
root.mainloop()
