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
test_data = pd.DataFrame()    
for subdir, dirs, files in os.walk(path):
    for dir in dirs:
        subdir_path = os.path.join(subdir, dir)
        for subdir1, dirs1, files1 in os.walk(subdir_path):
            if dir == "Object 1":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 1")
                        object1_data = object1_data.append(df,ignore_index=True)
                print("object1_data is uploaded")
                print(object1_data)
                print(object1_data.isnull().sum().sum()) #To check is there is any null(NaN) in the data
                
            elif dir == "Object 2":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 2")
                        object2_data = object2_data.append(df,ignore_index=True)
                print("object2_data is uploaded")
                print(object2_data)
                print(object2_data.isnull().sum().sum())
            elif dir == "Object 3":
                for f in files1:
                        df = pd.read_excel(os.path.join(subdir1,f), header = None, usecols = "G:DZZ")
                        df.insert(0, "Object Type", "Object 3")
                        object3_data = object3_data.append(df,ignore_index=True)
                print("object3_data is uploaded")
                print(object3_data)
                print(object3_data.isnull().sum().sum())
test_data = test_data.append(object1_data,ignore_index = True)
test_data = test_data.append(object2_data,ignore_index = True)
test_data = test_data.append(object3_data,ignore_index = True)
print(test_data)
print("Completed")
            
            
            
          