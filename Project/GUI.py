import tkinter as tk
import os
import openpyxl
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from tkinter import messagebox
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# start tkinter app code
root = tk.Tk()
root.title("Discrimination of Reflected Sound Signal(MLP Classifier)")

# setting canvas size and grid
canvas = tk.Canvas(root, width = 1000, height = 1000)
canvas.grid(columnspan = 6, rowspan = 12)

#For Training
tk.Label(root, text = "Training the Model", font = 'customFont1', fg = "black", bg = "sky blue", width = 15).grid(columnspan = 1, row = 0, column = 0)

folder_path = tk.StringVar()    
tk.Label(root, text = "Path for Training the Model : ", font = 'customFont1', fg = "black", width = 20).grid(columnspan = 1, row = 1, column = 0)
training_path = tk.Entry(root, width = 60, textvariable = folder_path).grid(row = 1, column = 1)

start_column = tk.IntVar()
tk.Label(root, text = "Signal Starts at Column : ", font = 'customFont1', fg = "black", width = 20).grid(columnspan = 1, row = 1, column = 2)
signal_start = tk.Entry(root, width = 5, textvariable = start_column).grid(row = 1, column = 3)  

end_column = tk.IntVar()
tk.Label(root, text = "Signal Ends at Column : ", font = 'customFont1', fg = "black", width = 20).grid(columnspan = 1, row = 1, column = 4)
signal_start = tk.Entry(root, width = 5, textvariable = end_column).grid(row = 1, column = 5)

tk.Label(root, text = "Accuracy obtained after validating the model with 20% of training data : ", font = 'customFont1', fg = "black", width = 40, wraplength = 250).grid(columnspan = 1, row = 3, column = 0)
string_variable = tk.StringVar()
tk.Label(root, textvariable = string_variable).grid(columnspan = 1, row = 3, column = 1)

def main():
    global training_set1, training_set2, training_set3, validation_set1, validation_set2, validation_set3
    global folder_path, start_column, end_column, classifier, string_variable
    
    content = folder_path.get()
    if content == '':
        tk.messagebox.showwarning("Folder Path empty","Please specify folder path!")
    
    try :
        start_column = start_column.get()
    except :
        tk.messagebox.showwarning("Starting Column empty", "Please specify starting column!")
        
    try :
        end_column = end_column.get()
    except :
        tk.messagebox.showwarning("Ending Column empty", "Please specify ending column!")
    
    path = content
    signal_width = range(start_column, end_column)

    object1_data = pd.DataFrame()
    object2_data = pd.DataFrame()
    object3_data = pd.DataFrame()  
    training_set = pd.DataFrame()
    training_set1 = pd.DataFrame() 
    training_set2 = pd.DataFrame() 
    training_set3 = pd.DataFrame()   
    validation_set = pd.DataFrame()
    validation_set1 = pd.DataFrame() 
    validation_set2 = pd.DataFrame() 
    validation_set3 = pd.DataFrame()   
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(subdir, dir)
            for subdir1, dirs1, files1 in os.walk(subdir_path):
                if dir == "Object 1":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = signal_width)
                            df.insert(0, "Object Type", "Object 1")
                            object1_data = object1_data.append(df,ignore_index=True)
                    training_set1, validation_set1 = train_test_split(object1_data, test_size = 0.2, random_state = 21)
                    
                elif dir == "Object 2":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = signal_width)
                            df.insert(0, "Object Type", "Object 2")
                            object2_data = object2_data.append(df,ignore_index=True)
                    training_set2, validation_set2 = train_test_split(object2_data, test_size = 0.2, random_state = 21)
                
                elif dir == "Object 3":
                    for f in files1:
                            df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = signal_width)
                            df.insert(0, "Object Type", "Object 3")
                            object3_data = object3_data.append(df,ignore_index=True)
                    training_set3, validation_set3 = train_test_split(object3_data, test_size = 0.2, random_state = 21)

    #Appending training data from all the object types        
    training_set = training_set.append(training_set1,ignore_index = True)
    training_set = training_set.append(training_set2,ignore_index = True)
    training_set = training_set.append(training_set3,ignore_index = True)

    #Appending validation data from all the object types  
    validation_set = validation_set.append(validation_set1,ignore_index = True)
    validation_set = validation_set.append(validation_set2,ignore_index = True)
    validation_set = validation_set.append(validation_set3,ignore_index = True)

    X_train = training_set.iloc[:, 1:].values
    Y_train = training_set.iloc[:, 0].values
    X_val = validation_set.iloc[:, 1:].values
    y_val = validation_set.iloc[:, 0].values

    #Initializing the MLPClassifier
    classifier = MLPClassifier(hidden_layer_sizes=(100,50,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

    #Fitting the training data to the network
    classifier.fit(X_train, Y_train)

    #Predicting y for X_val
    y_pred = classifier.predict(X_val)

    #Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(y_pred, y_val)
    print(cm, "\n")

    #Ploting confusion matrix
    disp = plot_confusion_matrix(classifier, X_val, y_val,
                                    cmap=plt.cm.Blues)
    plt.show()

    #Printing the accuracy
    accuracy = accuracy_score(y_pred, y_val)
    print("Accuracy of MLPClassifier : ", accuracy, "\n")
    accuracy_percent = accuracy * 100
    string_variable = string_variable.set("{0:.2f}".format(accuracy_percent))

    #Printing F1
    print(classification_report(y_pred,y_val), "\n")

    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    print("TruePostive(TP) : ", TP, "\n")
    print("TrueNegative(TN) : ", TN, "\n")
    print("FalsePostive(FP) : ", FP, "\n")
    print("False Negative(FN) : ", FN, "\n")

    # False discovery rate
    FDR = FP/(TP+FP)
    print("False Discovery Rate(FDR) : ", FDR, "\n")

    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative Preductive Value(NPV) : ", NPV, "\n")

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("True Positive Rate(TPR) : ", TPR, "\n")

    # Specificity, selectivity or true negative rate
    TNR = TN/(TN+FP) 
    print("True Negative Rate(TNR) : ", TNR, "\n")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_val))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='green', linestyle='dotted', linewidth=4)

    colors = cycle(['purple', 'sienna', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of object {0} (area = {1:0.2f})'
                ''.format(i + 1 , roc_auc[i]))
        print("roc_auc_score of object " , i+1, ": ", roc_auc[i])

    print("\n")
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    tk.messagebox.showinfo("Completed", "Training and Validating model is completed successfully")
    

trainBtnTxt = tk.StringVar()
trainBtn = tk.Button(root, textvariable = trainBtnTxt, command = lambda:main() ,font = 'customFont1', bg = "azure", fg = "black",height = 3, width = 15)
trainBtnTxt.set("Train")
trainBtn.grid(columnspan = 6, column = 0, row = 2)

#For Testing
tk.Label(root, text = "Testing the Model", font = 'customFont1', fg = "black", bg = "sky blue", width = 15).grid(columnspan = 1, row = 4, column = 0)

file_path = tk.StringVar()  
tk.Label(root, text = "Path for Testing the Model : ", font = 'customFont1', fg = "black", width = 20).grid(columnspan = 1, row = 5, column = 0)
training_path = tk.Entry(root, width = 60, textvariable = file_path).grid(row = 5, column = 1)

file_output_path = tk.StringVar()
tk.Label(root, text = "Path to save data file after testing : ", font = 'customFont1', fg = "black", width = 40).grid(columnspan = 1, row = 6, column = 0)
training_path = tk.Entry(root, width = 60, textvariable = file_output_path).grid(row = 6, column = 1)

def testing():
    global file_path, file_output_path
    
    filepath = file_path.get()
    if filepath == '':
        tk.messagebox.showwarning("Testing file path empty", "Please specify file path for Testing!")
        
    fileoutput_path = file_output_path.get()
    if fileoutput_path == '':
        tk.messagebox.showwarning("Output file path empty", "Please specify file path for the Output file!")
        
    signal_width = range(start_column, end_column)
    test_path = filepath
    output_path = fileoutput_path

    test_data = pd.read_excel(test_path, header = None, usecols = signal_width)

    y_test = classifier.predict(test_data.values)

    workbook = openpyxl.load_workbook(test_path)
    worksheet = workbook.worksheets[0]
    worksheet.insert_cols(3407)
    y = 1
    for x in range(len(y_test)):
        cell_to_write = worksheet.cell(row = y, column = 3407)
        cell_to_write.value = y_test[x]
        y += 1

    workbook.save(output_path)
    tk.messagebox.showinfo("Completed", "Testing is completed and Output file is generated")
    
testBtnTxt = tk.StringVar()
testBtn = tk.Button(root, textvariable = testBtnTxt, command = lambda:testing() ,font = 'customFont1', bg = "azure", fg = "black",height = 3, width = 15)
testBtnTxt.set("Test")
testBtn.grid(columnspan = 6, column = 0, row = 7)

def Close():
    root.destroy()
  
# Button for closing
exit_button = tk.Button(root, text = "Exit", command = Close, height = 3, width = 15).grid(columnspan = 6, column = 3, row = 7)

root.mainloop()
