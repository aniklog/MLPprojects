# importing modules
import tkinter as tk
from tkinter.font import Font
from tkinter.filedialog import askopenfile
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# start tkinter app code
root = tk.Tk()

# setting canvas size and grid
canvas = tk.Canvas(root, width=1000, height=1000)
canvas.grid(columnspan=5, rowspan=10)

# GUI instruction
customFont1 = Font(family="Helvetica", size=20, weight="bold", slant="roman")
customFont2 = Font(family="Helvetica", size=20, weight="bold", slant="italic")
customFont3 =  Font(family="Helvetica", size=20, weight="bold", slant="roman")
instructions = tk.Label(root, text="Select the excel file for testing", font='customFont1',fg="black", bg="orange",width=40)
instructions.grid(columnspan=6, column=0,row=0)

# initializing variable for input
df_input = None
    
#Browse button for opening file
browseBtnTxt = tk.StringVar()
button_explore = tk.Button(root,textvariable=browseBtnTxt,command = lambda:open_file(),width = 20, height = 4,
                            fg = "blue",bg="white")
browseBtnTxt.set("Browse File")
button_exit = tk.Button(root,text = "Exit",command = exit,width = 15, height = 4,fg = "blue",bg="white")
button_explore.grid(column = 2, row = 1)
button_exit.grid(column = 0,row = 6)

# initializing signal detail input fields
tk.Label(root, text="Start column:", font='customFont2').grid(columnspan=1, column=0,row=2)
signalStartColumn = tk.Entry(root)
signalStartColumn.grid(row=2, column=1)

tk.Label(root, text="End column:", font='customFont2').grid(columnspan=1, column=2,row=2)
signalEndColumn = tk.Entry(root)
signalEndColumn.grid(row=2, column=3)

tk.Label(root, text="Signal Row:", font='customFont2').grid(columnspan=1, column=4,row=2)
signalRow = tk.Entry(root)
signalRow.grid(row=2, column=5)

# Predict Button
predictBtnTxt = tk.StringVar()
predictBtn = tk.Button(root, textvariable=predictBtnTxt, command=lambda:predict() ,font='Railway', bg="azure", fg="black",height=3, width=15)
predictBtnTxt.set("Predict")
predictBtn.grid(columnspan=6, column=0, row= 3)

# initializing text label for showing predicted output 
model_result = tk.Label(root, text="", font='Railway')
model_result.grid(columnspan=6, column=0,row=7)

# initializing plot
#fig = Figure(figsize = (4, 3), dpi = 100)
#fig.add_subplot(111)
#canvas_fig = FigureCanvasTkAgg(fig, master = root)
#canvas_fig.get_tk_widget().grid(columnspan=6, column=0, rowspan=3,row=4)

# function for uploading file and saving dataframe to variable 'df_input'
def open_file():
    signalStartColumn.delete(0, 'end')
    signalEndColumn.delete(0, 'end')
    signalRow.delete(0, 'end')
    print(".")
    browseBtnTxt.set("Loading")
    model_result.config(text = "")
    file = askopenfile(parent=root, mode="rb", title="Choose a file", filetype=[("Excel file","*.xlsx")])
    global plot1, df_input
    
    # check if there is file uploaded
    if file:
        print(file)
        print('File is loaded.')
        file_name = file.name
        file_name = str(file_name)
        file_path=file_name.replace("/", "\\")
        print(file.name)
        df_input = pd.read_excel(r''+file_path, header=None)  
        print('File is read.')
        
        browseBtnTxt.set("Browse")

def predict():
    df = df_input
    print(df)
    # check if value is null then assigning default value
    if signalStartColumn.get():
        signalStartColumnInt = int(signalStartColumn.get())
        print("signal Start Column: ", signalStartColumnInt)
    else:
        signalStartColumnInt = 6

    if signalEndColumn.get():
        signalEndColumnInt = int(signalEndColumn.get())
        print("signal End Column: ", signalEndColumnInt)
    else:
        signalEndColumnInt = 3406

    if signalRow.get():
        signalRowInt = int(signalRow.get())
        print("signal Row: ", signalRowInt)
    else:
        signalRowInt = 1

    signalLength = (signalEndColumnInt - signalStartColumnInt)
    df = df.iloc[signalRowInt - 1: signalRowInt, signalStartColumnInt: signalEndColumnInt]
    print("signalLength: ", signalLength)
    print("df: ")
    print(df)

#Prediction and validation
    predictBtnTxt.set("Predicting")
    filename = 'trained_model.sav'
    model = pickle.load(open(filename, 'rb'))
    print('Model loaded.')
    result = model.predict(df)
    print(result)


# end tkinter app code
root.mainloop()

