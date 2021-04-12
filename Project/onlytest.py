import tkinter as tk
import os
import openpyxl
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
from itertools import cycle
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# start tkinter app code
root = tk.Tk()
root.title("Discrimination of Reflected Sound Signal(MLP Classifier)")

# setting canvas size and grid
canvas = tk.Canvas(root, width=1000, height=1000)
canvas.grid(rowspan=14, columnspan=15)

#Heading
tk.Label(root, text = "Testing from Loaded Model", font = 'customFont1', fg = "black", bg = "sky blue", width = 40).grid(columnspan = 1, row = 0, column = 0)

start_column = tk.IntVar()
tk.Label(root, text="Signal Starts at Column : ", font='customFont1', fg="black", width=20).grid(columnspan=1, row=2,
                                                                                                 column=0)
signal_start = tk.Entry(root, width=5, textvariable=start_column).grid(row=2, column=1, sticky="W")

end_column = tk.IntVar()
tk.Label(root, text="Signal Ends at Column : ", font='customFont1', fg="black", width=20).grid(columnspan=1, row=2,
                                                                                                 column=2)
signal_end = tk.Entry(root, width=5, textvariable=end_column).grid(row=2, column=3, sticky="W")


# For Testing
tk.Label(root, text="Testing the Model", font='customFont1', fg="black", bg="sky blue", width=15).grid(columnspan=1,
                                                                                                       row=3, column=0)
def browse_file():
    global file_path
    filename = filedialog.askopenfilename()
    file_path.set(filename)


file_path = tk.StringVar()
tk.Label(root, text="Path for Testing the Model : ", font='customFont1', fg="black", width=25).grid(columnspan=1, row=4,
                                                                                                    column=0)
training_path = tk.Entry(root, width=60, textvariable=file_path).grid(row=4, column=1)

browseFileBtn = tk.Button(root, text="Browse File", command=lambda: browse_file(), font='customFont1', bg="azure",
                          fg="black", height=2, width=15)
browseFileBtn.grid(row=4, column=2)

#To Load ML model
def trained_file():
    global pkl_filepath
    filename = filedialog.askopenfilename()
    pkl_filepath.set(filename)

pkl_filepath= tk.StringVar()
tk.Label(root, text="Path for Loading the Model File: ", font='customFont1', fg="black", width=30).grid(columnspan=1, row=5,
                                                                                                    column=0)
training_path = tk.Entry(root, width=60, textvariable=pkl_filepath).grid(row=5, column=1)

browseLoadFileBtn = tk.Button(root, text="Browse Model", command=lambda: trained_file(), font='customFont1', bg="azure",
                          fg="black", height=2, width=20)
browseLoadFileBtn.grid(row=5, column=2)

def browse_outputFile():
    global file_output_path
    outputfilename = filedialog.askdirectory()
    filename = "/Output.xlsx"
    file_output_path.set(outputfilename + filename)

file_output_path = tk.StringVar()
tk.Label(root, text="Path to save data file after testing : ", font='customFont1', fg="black", width=40).grid(
    columnspan=1, row=7, column=0)
training_path = tk.Entry(root, width=60, textvariable=file_output_path).grid(row=7, column=1)

browseOFileBtn = tk.Button(root, text="Browse Output Folder", command=lambda: browse_outputFile(), font='customFont1',
                           bg="azure", fg="black", height=2, width=20)
browseOFileBtn.grid(row=7, column=2)

def testing():
    global file_path, file_output_path,start_column, end_column,pkl_filepath
    try:
        start_column = start_column.get()
    except:
        tk.messagebox.showwarning("Starting Column empty", "Please specify starting column!")

    try:
        end_column = end_column.get()
    except:
        tk.messagebox.showwarning("Ending Column empty", "Please specify ending column!")
    try:
        file_path = file_path.get()
    except:
        tk.messagebox.showwarning("Testing file path empty", "Please specify file path for Testing!")

    try:
        file_output_path = file_output_path.get()
    except:
        tk.messagebox.showwarning("Output file path empty", "Please specify file path for the Output file!")

    try:
        pkl_filepath = pkl_filepath.get()
    except:
        tk.messagebox.showwarning("Output file path empty", "Please specify file path for the Output file!")

    signal_width = range(start_column, end_column)

    test_data = pd.read_excel(file_path, header=None, usecols=signal_width)

    #pkl_filename = "pickle_model.pkl"
    with open(pkl_filepath, 'rb') as file:
        pickle_model = pickle.load(file)

    y_test = pickle_model.predict(test_data.values)

    workbook = openpyxl.load_workbook(file_path)
    worksheet = workbook.worksheets[0]
    worksheet.insert_cols(3407)
    y = 1
    for x in range(len(y_test)):
        cell_to_write = worksheet.cell(row=y, column=3407)
        cell_to_write.value = y_test[x]
        y += 1

    workbook.save(file_output_path)
    tk.messagebox.showinfo("Completed", "Testing is completed and Output file is generated")


testBtn = tk.Button(root, text="Test", command=lambda: testing(), font='customFont1', bg="azure", fg="black", height=3,
                    width=15)
testBtn.grid(columnspan=6, column=0, row=8)


def Close():
    root.destroy()


# Button for closing
exit_button = tk.Button(root, text="Exit", command=Close, height=3, width=15).grid(columnspan=6, column=3, row=8)

root.mainloop()
