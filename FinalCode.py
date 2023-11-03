'''Read me
Good Day

This code goes thorugh the following stages

Stage 1: Importing Libraries
Stage 2: Functions
Stage 3: Cleaning Data
Stage 4: Making Windows
Stage 5: Pre-Processing for ANN model
Stage 6: ANN Model
Stage 7: Saving Results to CSV file
'''

###Stage 1: Importing Libraries

#!/usr/bin/env python3
import os
#sys.stdout.flush() 
import numpy as np
import pandas as pd
import datetime, copy, imp
import time
import os
import re
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import csv  
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

##Imports for ANN:   
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Activation, Flatten

#%%

###Stage 2: Functions


#Aggregate Data for an individual in a window from startDate to endDate.    
def timeline_summary(tbl,startDate='NoDate',endDate='NoDate'):
    if startDate != 'NoDate' and endDate != 'NoDate':
        tbl = tbl.loc[ (tbl.Date >= startDate) & (tbl.Date <= endDate) ]
        
    return pd.Series({
        'NumGoodTestResult': (tbl.Event == 'GoodTestResult').sum(),
        'NumStay': (tbl.Event == 'Stay').sum(),
        'NumBadTestResult': (tbl.Event == 'BadTestResult').sum(),
        'NumVitalsCrash': (tbl.Event == 'VitalsCrash').sum(),
        'Tenure': (tbl.Date.max()-tbl.Date.min()).days
    })

#Flags all the patients with Vitals Crash
def find_vitals_crash(tbl,startDate,endDate):
    advOutIdx = (tbl.Event == 'VitalsCrash') & (tbl.Date >= startDate) & (tbl.Date <= endDate)
    if advOutIdx.sum() > 0:
        return pd.Series({
            'Flag': 'Crash',
            'Time': (tbl.loc[advOutIdx].Date.min() - startDate).days
        })
    else:
        return pd.Series({
            'Flag': 'NoCrash',
            'Time': max(0,(endDate - startDate).days)
        })
    
#Generates the Matrix for the Neural Network    
def generate_sequences(x, seq_len, shuffle=True):# seq len is changed by number of functs
    result = []
    index = 0
    for R in range(round((x.shape[0] - seq_len)/seq_len)):
        result.append(x[index: index + seq_len])
        index = index + seq_len
    result = np.array(result)
    return result


#%%
###Stage 3: Cleaning Data
# In this stage we are filtering Out Players with less than 15 events in the Observation Window

#The data actually goes from 1993 to 2015

##Observation Window:
Start_Time = obsStart = pd.to_datetime('1999-01-01')
End_Time = obsEnd = pd.to_datetime('2013-12-31')

##FollowUp Window:
followStart = pd.to_datetime('2014-01-01')
followEnd = pd.to_datetime('2014-12-31')


##Data Import
dataFileStr = '/home/musa.taib/MLBHospitalData.hd5'
dat = pd.read_hdf(dataFileStr,key='Data')

##Data Preprocessing
Unique_Events = dat.Event.unique()

tte = dat.groupby(level=0).apply(find_vitals_crash,startDate=followStart,endDate=followEnd)
ftr = dat.groupby(level=0).apply(timeline_summary)

#Dataset_X_Values
ftrObs = dat.groupby(level=0).apply(timeline_summary,startDate=obsStart,endDate=obsEnd)

#We are replacing the Flag in observed period with the flags in the followup period
ftrObs['Flag'] = tte.Flag

#We will set the limit to only those people who have tenured for more then 15 days in the last 2 years
Temp_TransferVariable= ftrObs.reset_index()
 
PatientName = Temp_TransferVariable.Player
data = []
y = 0

for L1 in PatientName:
    if ftrObs.loc[L1].Tenure >= 15:
        data.append(Temp_TransferVariable.loc[y])
    y = y + 1

Data_Base = pd.DataFrame(data, columns=['Player','NumGoodTestResult','NumStay','NumBadTestResult','NumVitalsCrash','Tenure','Flag'])
Data_Base = Data_Base.set_index('Player')
Y_data_series = Data_Base["Flag"]
Data_Base = Data_Base.drop("Flag",1)
Y_data = pd.DataFrame(Y_data_series)

#Making a matrix of all zeroes for every patient that we will use in Stage 4 to zero pad the data
x = 0
for x in Data_Base.index:
    Data_Base.loc[x,"NumBadTestResult"] = 0
    Data_Base.loc[x,"NumGoodTestResult"] = 0
    Data_Base.loc[x,"NumStay"] = 0
    Data_Base.loc[x,"NumVitalsCrash"] = 0
    Data_Base.loc[x,"Tenure"] = 0
    

#%%

###Stage 4: Making Windows
  
Number_of_Windows = os.getenv('SLURM_ARRAY_TASK_ID') 
Number_of_Windows = int(Number_of_Windows)
No_of_patients = len(Data_Base.index)

#Making Intervals
Delta = End_Time - Start_Time
No_days = Delta.days
End_year = End_Time.year
End_month = End_Time.month
No_days_perwindow = round(No_days/Number_of_Windows)
Next_Year = Start_Time + datetime.timedelta(days= No_days_perwindow)

Period_Start = Start_Time

appended_data = []
for x in range (Number_of_Windows):
    Period_End = Period_Start + datetime.timedelta(days= No_days_perwindow)
    if (Period_End>= End_Time): #To make sure we dont go above the date
        Period_End = End_Time
    
    obsStart = Period_Start
    obsEnd = Period_End
    
    ##Data Preprocessing
    Unique_Events = dat.Event.unique()
     
    ftr = dat.groupby(level=0).apply(timeline_summary)
    
    # X_Values (for the Window)
    ftrObs = dat.groupby(level=0).apply(timeline_summary,startDate=obsStart,endDate=obsEnd)


    #We are replacing the Flag in observed period with the flags in the followup period
    ftrObs['Flag'] = tte.Flag
    
    ObsftrEdt = pd.DataFrame(data, columns=['Player','NumGoodTestResult','NumStay','NumBadTestResult','NumVitalsCrash','Tenure','Flag'])
    ObsftrEdt = ObsftrEdt.set_index('Player')
    ObsftrEdt = ObsftrEdt.drop("Flag",1)
    Period_Start = Period_End
    
    
    #Loop to make datasets
    loop_x1 = 0
    X_Fill = Data_Base.copy(deep=True)
    for loop_x1 in Data_Base.index:
            try:
                X_Fill.loc[loop_x1] = ObsftrEdt.loc[loop_x1] #Filling Values in the Zero Padded Matrix
            except:
                continue
    
    loop_x2 = 0
    loop_x1 = 0
    appended_data.append(X_Fill)
      
appended_data = pd.concat(appended_data)
appended_data = appended_data.sort_index()


#%%
###Stage 5: Pre-Processing for ANN model

data_df = appended_data
x = list(data_df.columns) 
seq_len = Number_of_Windows
num_features = 5
z_dimension = No_of_patients
X_data = generate_sequences(data_df, seq_len)

#Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_data)
Y_data2 = encoder.transform(Y_data)
Y_data2 = generate_sequences(Y_data2, 1)


skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in skf.split(X_data, Y_data2):
     X_train, X_test = X_data[train_index], X_data[test_index]
     Y_train, Y_test = Y_data2[train_index], Y_data2[test_index]

#%%
###Stage 6: ANN Model

model = Sequential()
model.add(Flatten(input_shape=(Number_of_Windows, 5)))
model.add(Dense(10, activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
# For a binary classification problem as I want to detect if there are vital signs or not
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=("accuracy"))
model.summary()


#Fiting
model.fit(x=X_train, y=Y_train,epochs=250, validation_data=(X_test, Y_test), verbose=1)


#Model Evaluation:
predictions = model.predict_classes(X_test)

#%%
###Stage 7: Saving Results to CSV file

Output = classification_report(Y_test,predictions,output_dict=True)
Output_df = pd.DataFrame(Output).transpose()
Accuracy = Output_df["precision"].loc["accuracy"]
Macro_Average =  Output_df["precision"].loc["macro avg"]
F1_Score_0 = Output_df["precision"].loc["0"]
F1_Score_1 = Output_df["precision"].loc["1"]
Windows = Number_of_Windows

data = [Windows,Accuracy,Macro_Average,F1_Score_0,F1_Score_1]


with open('Outputs.csv', 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(data) #writeData

