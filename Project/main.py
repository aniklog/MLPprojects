# import os
# import pandas as pd
# from os import walk

# path = r'/Users/mouni.kolisetty/Documents/Sem 3/machine learning/Files for task 5'
# for f in os.listdir(path):
#    if f.startswith("Object"):
#    print (os.path.join(path , f))
# # df = pd.read_excel (r'/Users/mouni.kolisetty/Documents/Sem 3/machine learning/Files for task 5/Object 1/record_data_2020-10-29_10-41-1_H.xlsx')
# # print (df)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
path = r'/Users/mouni.kolisetty/Documents/Sem 3/machine learning/Files for task 5'

# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith(".xlsx"):
#             df = pd.read_excel (os.path.join(subdir, file))
#             print(df)
#             #print(os.path.join(subdir, file))
object1_data = pd.DataFrame()
object2_data = pd.DataFrame()
object3_data = pd.DataFrame()  
training_set = pd.DataFrame()  
validation_set = pd.DataFrame()  
for subdir, dirs, files in os.walk(path):
    for dir in dirs:
        subdir_path = os.path.join(subdir, dir)
        for subdir1, dirs1, files1 in os.walk(subdir_path):
            if dir == "Object 1":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 1")
                        object1_data = object1_data.append(df,ignore_index=True)
                training_set1, validation_set1 = train_test_split(object1_data, test_size = 0.2, random_state = 21)
                print(training_set1)
                print(validation_set1)
                print(object1_data.isnull().sum().sum()) #To check is there is any null(NaN) in the data
                
            elif dir == "Object 2":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 2")
                        object2_data = object2_data.append(df,ignore_index=True)
                training_set2, validation_set2 = train_test_split(object2_data, test_size = 0.2, random_state = 21)
                print(training_set2)
                print(validation_set2)
                print(object2_data.isnull().sum().sum())
            elif dir == "Object 3":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 3")
                        object3_data = object3_data.append(df,ignore_index=True)
                training_set3, validation_set3 = train_test_split(object3_data, test_size = 0.2, random_state = 21)
                print(training_set3)
                print(validation_set3)
                print(object3_data.isnull().sum().sum())
training_set = training_set.append(training_set1,ignore_index = True)
training_set = training_set.append(training_set2,ignore_index = True)
training_set = training_set.append(training_set3,ignore_index = True)
print(training_set)

validation_set = validation_set.append(validation_set1,ignore_index = True)
validation_set = validation_set.append(validation_set2,ignore_index = True)
validation_set = validation_set.append(validation_set3,ignore_index = True)
print(validation_set)

X_train = training_set.iloc[:, 1:-1].values
Y_train = training_set.iloc[:, 0].values
X_val = validation_set.iloc[:, 1:-1].values
y_val = validation_set.iloc[:, 0].values

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

#Fitting the training data to the network
classifier.fit(X_train, Y_train)

#Predicting y for X_val
y_pred = classifier.predict(X_val)

#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_val)

#Printing the accuracy
print(cm)
print("Accuracy of MLPClassifier : ", accuracy(cm))
print("Completed")
            
            
            
          