import pandas as pd

"""###Reading the Boxplot DataFrames"""

df1=pd.read_csv("path to df_boxplotBERT.csv")
df2=pd.read_csv("path to df_boxplotBiLSTM.csv")
df3=pd.read_csv("path to df_boxplotKeyword.csv")

df1=df1.drop(columns=['Unnamed: 0'])
df2=df2.drop(columns=['Unnamed: 0'])
df3=df3.drop(columns=['Unnamed: 0'])

"""###Selecting Rows for Precision, Recall, and F-score from DataFrame"""

BERTP=df1.head(20)
BERTR=df1[20:].head(20)
BERTF=df1[40:].head(20)

LSTMP=df2.head(20)
LSTMR=df2[20:].head(20)
LSTMF=df2[40:].head(20)

key1P=df3.head(20)
key1R=df3[20:].head(20)
key1F=df3[40:].head(20)

column_names = df1.columns.tolist()
column_names

BERTP_Measurement = BERTP['Measurement'].tolist()
BERTP_Temperature = BERTP['Temperature'].tolist()
BERTP_Mass = BERTP['Mass'].tolist()
BERTP_Size = BERTP['Size'].tolist()
BERTP_Constraint = BERTP['Time Constraint'].tolist()
BERTP_NonlabelData = BERTP['Non-labelData'].tolist()
BERTP_LabelData = BERTP['LabelData'].tolist()
BERTP_Data = BERTP['Data'].tolist()
BERTP_Overall = BERTP['Overall'].tolist()

"""###Extracting Lists from DataFrame"""

LSTMP_Measurement = LSTMP['Measurement'].tolist()
LSTMP_Temperature = LSTMP['Temperature'].tolist()
LSTMP_Mass = LSTMP['Mass'].tolist()
LSTMP_Size = LSTMP['Size'].tolist()
LSTMP_Constraint = LSTMP['Time Constraint'].tolist()
LSTMP_NonlabelData = LSTMP['Non-labelData'].tolist()
LSTMP_LabelData = LSTMP['LabelData'].tolist()
LSTMP_Data = LSTMP['Data'].tolist()
LSTMP_Overall = LSTMP['Overall'].tolist()

key1P_Measurement = key1P['Measurement'].tolist()
key1P_Temperature = key1P['Temperature'].tolist()
key1P_Mass = key1P['Mass'].tolist()
key1P_Size = key1P['Size'].tolist()
key1P_Constraint = key1P['Time Constraint'].tolist()
key1P_NonlabelData = key1P['Non-labelData'].tolist()
key1P_LabelData = key1P['LabelData'].tolist()
key1P_Data = key1P['Data'].tolist()
key1P_Overall= key1P['Overall'].tolist()

BERTR_Measurement = BERTR['Measurement'].tolist()
BERTR_Temperature = BERTR['Temperature'].tolist()
BERTR_Mass = BERTR['Mass'].tolist()
BERTR_Size = BERTR['Size'].tolist()
BERTR_Constraint = BERTR['Time Constraint'].tolist()
BERTR_NonlabelData = BERTR['Non-labelData'].tolist()
BERTR_LabelData = BERTR['LabelData'].tolist()
BERTR_Data = BERTR['Data'].tolist()
BERTR_Overall = BERTR['Overall'].tolist()

LSTMR_Measurement = LSTMR['Measurement'].tolist()
LSTMR_Temperature = LSTMR['Temperature'].tolist()
LSTMR_Mass = LSTMR['Mass'].tolist()
LSTMR_Size = LSTMR['Size'].tolist()
LSTMR_Constraint = LSTMR['Time Constraint'].tolist()
LSTMR_NonlabelData = LSTMR['Non-labelData'].tolist()
LSTMR_LabelData = LSTMR['LabelData'].tolist()
LSTMR_Data = LSTMR['Data'].tolist()
LSTMR_Overall = LSTMR['Overall'].tolist()

key1R_Measurement = key1R['Measurement'].tolist()
key1R_Temperature = key1R['Temperature'].tolist()
key1R_Mass = key1R['Mass'].tolist()
key1R_Size = key1R['Size'].tolist()
key1R_Constraint = key1R['Time Constraint'].tolist()
key1R_NonlabelData = key1R['Non-labelData'].tolist()
key1R_LabelData = key1R['LabelData'].tolist()
key1R_Data = key1R['Data'].tolist()
key1R_Overall = key1R['Overall'].tolist()

BERTF_Measurement = BERTF['Measurement'].tolist()
BERTF_Temperature = BERTF['Temperature'].tolist()
BERTF_Mass = BERTF['Mass'].tolist()
BERTF_Size = BERTF['Size'].tolist()
BERTF_Constraint = BERTF['Time Constraint'].tolist()
BERTF_NonlabelData = BERTF['Non-labelData'].tolist()
BERTF_LabelData = BERTF['LabelData'].tolist()
BERTF_Data = BERTF['Data'].tolist()
BERTF_Overall = BERTF['Overall'].tolist()

LSTMF_Measurement = LSTMF['Measurement'].tolist()
LSTMF_Temperature = LSTMF['Temperature'].tolist()
LSTMF_Mass = LSTMF['Mass'].tolist()
LSTMF_Size = LSTMF['Size'].tolist()
LSTMF_Constraint = LSTMF['Time Constraint'].tolist()
LSTMF_NonlabelData = LSTMF['Non-labelData'].tolist()
LSTMF_LabelData = LSTMF['LabelData'].tolist()
LSTMF_Data = LSTMF['Data'].tolist()
LSTMF_Overall = LSTMF['Overall'].tolist()

key1F_Measurement = key1F['Measurement'].tolist()
key1F_Temperature = key1F['Temperature'].tolist()
key1F_Mass = key1F['Mass'].tolist()
key1F_Size = key1F['Size'].tolist()
key1F_Constraint = key1F['Time Constraint'].tolist()
key1F_NonlabelData = key1F['Non-labelData'].tolist()
key1F_LabelData = key1F['LabelData'].tolist()
key1F_Data = key1F['Data'].tolist()
key1F_Overall = key1F['Overall'].tolist()

"""####Creating a List of Lists"""

BERTPList=[BERTP_Measurement,BERTP_Temperature,BERTP_Mass,BERTP_Size,BERTP_Constraint,BERTP_NonlabelData,BERTP_LabelData,BERTP_Data,BERTP_Overall ]

LSTMPList=[LSTMP_Measurement,LSTMP_Temperature,LSTMP_Mass,LSTMP_Size,LSTMP_Constraint,LSTMP_NonlabelData,LSTMP_LabelData,LSTMP_Data,LSTMP_Overall ]

key1PList=[key1P_Measurement,key1P_Temperature,key1P_Mass,key1P_Size,key1P_Constraint,key1P_NonlabelData,key1P_LabelData,key1P_Data,key1P_Overall ]

BERTRList=[BERTR_Measurement,BERTR_Temperature,BERTR_Mass,BERTR_Size,BERTR_Constraint,BERTR_NonlabelData,BERTR_LabelData,BERTR_Data,BERTR_Overall ]

LSTMRList=[LSTMR_Measurement,LSTMR_Temperature,LSTMR_Mass,LSTMR_Size,LSTMR_Constraint,LSTMR_NonlabelData,LSTMR_LabelData,LSTMR_Data,LSTMR_Overall]

key1RList=[key1R_Measurement,key1R_Temperature,key1R_Mass,key1R_Size,key1R_Constraint,key1R_NonlabelData,key1R_LabelData,key1R_Data,key1R_Overall ]

BERTFList=[BERTF_Measurement,BERTF_Temperature,BERTF_Mass,BERTF_Size,BERTF_Constraint,BERTF_NonlabelData,BERTF_LabelData,BERTF_Data,BERTF_Overall ]

LSTMFList=[LSTMF_Measurement,LSTMF_Temperature,LSTMF_Mass,LSTMF_Size,LSTMF_Constraint,LSTMF_NonlabelData,LSTMF_LabelData,LSTMF_Data,LSTMF_Overall]

key1FList=[key1F_Measurement,key1F_Temperature,key1F_Mass,key1F_Size,key1F_Constraint,key1F_NonlabelData,key1F_LabelData,key1F_Data,key1F_Overall ]

def Average(lst):
    return sum(lst) / len(lst)

"""###plot"""

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = [BERTP_Overall,BERTP_Measurement,BERTP_Constraint,BERTP_Data,
        BERTP_Temperature,BERTP_Mass,BERTP_Size,
        BERTP_NonlabelData,BERTP_LabelData]

ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(BERTP_Overall)
avg2 = Average(BERTP_Measurement)
avg3= Average(BERTP_Constraint)
avg4 = Average(BERTP_Data)
avg5 = Average(BERTP_Temperature)
avg6 = Average(BERTP_Mass)
avg7 = Average(BERTP_Size)
avg8 = Average(BERTP_NonlabelData)
avg9 = Average(BERTP_LabelData)

means = [avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9]

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))


def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title
plt.title("BERT Precision",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BERT Precision.pdf',bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = [BERTR_Overall,BERTR_Measurement,BERTR_Constraint,BERTR_Data,
        BERTR_Temperature,BERTR_Mass,BERTR_Size,
        BERTR_NonlabelData,BERTR_LabelData]

ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']

# AVG
avg1 = Average(BERTR_Overall)
avg2 = Average(BERTR_Measurement)
avg3= Average(BERTR_Constraint)
avg4 = Average(BERTR_Data)
avg5 = Average(BERTR_Temperature)
avg6 = Average(BERTR_Mass)
avg7 = Average(BERTR_Size)
avg8 = Average(BERTR_NonlabelData)
avg9 = Average(BERTR_LabelData)

means = [avg1, avg2, avg3, avg4, avg5, avg6, avg7, avg8,avg9]

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth=2)
        
    plt.plot([], c=color_code)

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize=15, fontweight='bold')


plt.title("BERT Recall", fontsize=15, fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15, fontweight='bold')
plt.xticks(fontsize=15, fontweight='bold')
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BERT Recall.pdf', bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = [BERTF_Overall,BERTF_Measurement,BERTF_Constraint,BERTF_Data,
        BERTF_Temperature,BERTF_Mass,BERTF_Size,
        BERTF_NonlabelData,BERTF_LabelData]

ticks = ['Overall','Measurment', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(BERTF_Overall)
avg2 = Average(BERTF_Measurement)
avg3= Average(BERTF_Constraint)
avg4 = Average(BERTF_Data)
avg5 = Average(BERTF_Temperature)
avg6 = Average(BERTF_Mass)
avg7 = Average(BERTF_Size)
avg8 = Average(BERTF_NonlabelData)
avg9 = Average(BERTF_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("BERT F-score",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BERT F-score.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=[LSTMP_Overall,LSTMP_Measurement,LSTMP_Constraint,LSTMP_Data,
      LSTMP_Temperature,LSTMP_Mass,LSTMP_Size,
      LSTMP_NonlabelData,LSTMP_LabelData]
ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(LSTMP_Overall)
avg2 = Average(LSTMP_Measurement)
avg3 = Average(LSTMP_Constraint)
avg4 = Average(LSTMP_Data)
avg5 = Average(LSTMP_Temperature)
avg6= Average(LSTMP_Mass)
avg7 = Average(LSTMP_Size)
avg8 = Average(LSTMP_NonlabelData)
avg9 = Average(LSTMP_LabelData)

means = [avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("BiLSTM Precision",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BiLSTM Precision.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data=[LSTMR_Overall,LSTMR_Measurement,LSTMR_Constraint,LSTMR_Data,
      LSTMR_Temperature,LSTMR_Mass,LSTMR_Size,
      LSTMR_NonlabelData,LSTMR_LabelData]
ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(LSTMR_Overall)
avg2 = Average(LSTMR_Measurement)
avg3 = Average(LSTMR_Constraint)
avg4 = Average(LSTMR_Data)
avg5 = Average(LSTMR_Temperature)
avg6= Average(LSTMR_Mass)
avg7 = Average(LSTMR_Size)
avg8 = Average(LSTMR_NonlabelData)
avg9 = Average(LSTMR_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("BiLSTM Recall",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BiLSTM Recall.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data=[LSTMF_Overall,LSTMF_Measurement,LSTMF_Constraint,LSTMF_Data,
      LSTMF_Temperature,LSTMF_Mass,LSTMF_Size,
      LSTMF_NonlabelData,LSTMF_LabelData]
ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(LSTMF_Overall)
avg2 = Average(LSTMF_Measurement)
avg3 = Average(LSTMF_Constraint)
avg4 = Average(LSTMF_Data)
avg5 = Average(LSTMF_Temperature)
avg6= Average(LSTMF_Mass)
avg7 = Average(LSTMF_Size)
avg8 = Average(LSTMF_NonlabelData)
avg9 = Average(LSTMF_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("BiLSTM F-score",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('BiLSTM F-score.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data =[key1P_Overall,key1P_Measurement,key1P_Constraint,key1P_Data,
       key1P_Temperature,key1P_Mass,key1P_Size,
       key1P_NonlabelData,key1P_LabelData]
ticks = ['Overall','Measurment', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(key1P_Overall)
avg2 = Average(key1P_Measurement)
avg3 = Average(key1P_Constraint)
avg4 = Average(key1P_Data)
avg5= Average(key1P_Temperature)
avg6 = Average(key1P_Mass)
avg7= Average(key1P_Size)
avg8 = Average(key1P_NonlabelData)
avg9 = Average(key1P_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("Keyword Search Precision",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('Keyword Search Precision.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data =[key1R_Overall,key1R_Measurement,key1R_Constraint,key1R_Data,
       key1R_Temperature,key1R_Mass,key1R_Size,
       key1R_NonlabelData,key1R_LabelData]
ticks = ['Overall','Measurement', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(key1R_Overall)
avg2 = Average(key1R_Measurement)
avg3 = Average(key1R_Constraint)
avg4 = Average(key1R_Data)
avg5= Average(key1R_Temperature)
avg6 = Average(key1R_Mass)
avg7= Average(key1R_Size)
avg8 = Average(key1R_NonlabelData)
avg9 = Average(key1R_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("Keyword Search Recall",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('Keyword Search Recall.pdf',bbox_inches='tight')
plt.show()

#This code is final.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data =[key1F_Overall,key1F_Measurement,key1F_Constraint,key1F_Data,
       key1F_Temperature,key1F_Mass,key1F_Size,
       key1F_NonlabelData,key1F_LabelData]
ticks = ['Overall','Measurment', 'Time\nConstraint','Data',
         'Temperature', 'Mass', 'Size', 
         'Non-label\nData', 'Label\nData']
  
# AVG
avg1 = Average(key1F_Overall)
avg2 = Average(key1F_Measurement)
avg3 = Average(key1F_Constraint)
avg4 = Average(key1F_Data)
avg5= Average(key1F_Temperature)
avg6 = Average(key1F_Mass)
avg7= Average(key1F_Size)
avg8 = Average(key1F_NonlabelData)
avg9 = Average(key1F_LabelData)

means = [avg1,avg2, avg3,avg4,avg5,avg6,avg7,avg8,avg9]
#print("means",means)

positions = np.array(np.arange(0, 12, 1))
widths = 0.4

fig, ax = plt.subplots(figsize=(7, 7))

def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()

for i, d in enumerate(data):
    bp = ax.boxplot(d, positions=[i], widths=widths, showmeans=True, vert=False, sym='r+')
    define_box_properties(bp, 'blue')
    
    for line in bp['means']:
        x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
        text = '{:.2f}'.format(means[i])
        plt.annotate(text, xy=(x - 0.01, y+ 0.21), fontsize = 15,fontweight='bold')

# set the title

plt.title("Keyword Search F-score",fontsize = 15,fontweight='bold')
plt.yticks(np.array(np.arange(0, 9)), ticks, fontsize=15,fontweight='bold')
plt.xticks(fontsize = 15,fontweight='bold')
#plt.xlabel("value", fontsize = 13)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.savefig('Keyword Search F-score.pdf',bbox_inches='tight')
plt.show()