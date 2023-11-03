
'''Cleaning up the output file
Data = pd.read_csv('/home/musajtaib/Desktop/Outputs.csv')  
Data = Data.sort_values("WindowSize")
Data = Data.set_index("WindowSize")
Data.to_csv('/home/musajtaib/Desktop/ENSF_Outputs.csv')
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})


Graph_Data = pd.read_csv('/home/musajtaib/Desktop/ENSF_Outputs.csv')
Graph_Data = Graph_Data.set_index("WindowSize")
#ax = Graph_Data.plot.bar(rot=0)


#matplotlib inline

Accuracy = Graph_Data["Accuracy"]
F1_Score_0 =  Graph_Data["F1_Score_0"]
F1_Score_1 =  Graph_Data["F1_Score_1"]
Macro_Average =  (F1_Score_0 + F1_Score_1)/2
WindowSize =  Graph_Data.index

#%% 
#Bar Charts


fig, axs = plt.subplots(2)
fig.suptitle('Accuracy & F1 Score 1 [BarCharts]')
axs[0].bar(WindowSize,Accuracy,color = "black")  
axs[0].set(ylabel = 'Accuracy')

axs[1].bar(WindowSize,F1_Score_1,color = "black")  
axs[1].set(ylabel='F1_Score_1')
axs[1].set(xlabel='Number of windows')
plt.grid()
fig.savefig("F1&AccuracyBar.eps", format="eps", dpi=600)
plt.show()


fig, axs = plt.subplots(2)
fig.suptitle('Macro Average & F1 Score 0 [BarCharts]')
axs[0].bar(WindowSize,Macro_Average,color = "black")  
axs[0].set(ylabel = 'Macro Average')

axs[1].bar(WindowSize,F1_Score_0,color = "black")  
axs[1].set(ylabel='F1_Score_0')
axs[1].set(xlabel='Number of windows')
plt.grid()
fig.savefig("F1&MacroAverageBar.eps", format="eps", dpi=600)
plt.show()

'''

#%%
#Scatter Plots

plt.scatter(WindowSize, Accuracy,color = "black")
plt.ylabel('Accuracy')
plt.xlabel('No of Windows')
plt.grid()
plt.show()

plt.scatter(WindowSize, Macro_Average,color = "black")
plt.ylabel('Macro_Average')
plt.xlabel('No of Windows')
plt.grid()
plt.show()

plt.scatter(WindowSize, F1_Score_0,color = "black")
plt.ylabel('F1_Score_0')
plt.xlabel('No of Windows');
plt.grid()
plt.show()

plt.scatter(WindowSize, F1_Score_1,color = "black")
plt.ylabel('F1_Score_1')
plt.xlabel('No of Windows')
plt.grid()
plt.show()

#%%

Accuracy_10 = Accuracy.sort_values(ascending=False)
Macro_Average_10 = Macro_Average.sort_values(ascending=False)
F1_Score_1_10 = F1_Score_1.sort_values(ascending=False)
F1_Score_0_10 = F1_Score_0.sort_values(ascending=False)

#Final_Value_35 = Graph_Data.iloc[[34]]
#Final_Value_206 = Graph_Data.iloc[[205]]
'''
#%%

