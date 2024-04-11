import pandas as pd
from sklearn.metrics import classification_report

"""###Automate procedure of reading classification reports and creating dataframes for boxplots
1) Automate copy pasting classification reorts in a dataframe where each cell has an index of a seed number and column of a label.
2) Read the datframe output as a csv file and calculate Metrics
3) Statistical tests
"""

# create a dictionary with empty lists for each column
data_dict = {'Measurement': [], 'Temperature': [], 'Mass': [], 'Size': [], 'Time Constraint': [],
             'Non-labelData': [], 'LabelData': [], 'Data': [], 'Overall': [],'Convention': []}

# populate each list with 20 empty values
for i in range(20):
    for col in data_dict.keys():
        data_dict[col].append('')

# create the dataframe from the dictionary
df = pd.DataFrame(data_dict)

df

# df_test1=pd.read_csv('/BERTLargeoutput.csv')
# df_test1 = df_test1.drop('index', axis=1)

# df_test2=pd.read_csv('/BiLSTMoutput.csv')
# df_test2 = df_test2.drop('index', axis=1)

# df_test3=pd.read_csv('/Keywordoutput.csv')
# df_test3 = df_test3.drop('index', axis=1)

"""####Try create df_combined which is used to create the dataframe of seeds and labels for each model"""

df_combined=df
print(len(df_combined))
# Define your desired index
new_index = [904727489, 42, 56789, 256, 32768, 1024, 2048, 4096, 8192, 116384,512, 16384, 65536, 131072, 262144, 1234, 98765, 55555, 888888, 6742]
# Set the index of df_combined to your desired index
df_combined = df_combined.set_index(pd.Index(new_index))

# Print the resulting DataFrame
df_combined

#auto complete classification results into my excel sheet
import pandas as pd
import os

# Set the path to the directory containing the CSV files
csv_dir = "."

# Create an empty DataFrame to hold the combined data
# df_combined = pd.DataFrame()

# Loop through each CSV file in the directory
for filename in sorted(os.listdir(csv_dir)):
    if filename.endswith(".csv"):
        print(filename)
        # Extract the column name and index from the filename
        # index_value,column_name = filename.split(".")[0].split("_")[1].split(" ")
        index_value=filename.split(".")[0].split("_")[1].split(" ")[0]
        index_value = int(index_value)
        # print(index_value)
        # for index in new_index1:
        #     index_value=index
        # if index_value==512:
        #     print("++++++++++++++=")
        #     print(index_value)
        column_name = filename.split(".")[0].split("_")[1].split(" ")[1:]
            # print(column_name)
        if len(column_name) == 2:
          column_name = " ".join([column_name[0], column_name[1]])
          # print("len 2",column_name)
        else:
          column_name=column_name[0]
          # print("len 1",column_name)


            # Read the CSV file into a DataFrame
            # df = pd.read_csv(os.path.join(csv_dir, filename), index_col=0)

            # Rename the columns to include the column name
            # df = df.rename(columns={col: f"{column_name}_{col}" for col in df.columns})

            # Read the CSV file into a DataFrame and set the index to include the column name
        df = pd.read_csv(os.path.join(csv_dir, filename), index_col=[0])

        # Add the data to the combined DataFrame using the index value as the row index

        #df_combined.loc[index_value, column_name] = df
        # Set the value of the cell at the given row and column to the data from the CSV file
        try:
          df_combined.at[index_value, column_name] = df
        except:
          continue

# Sort the rows of the combined DataFrame by index value
#df_combined = df_combined.sort_index()

# Reset the index to the original index name (if desired)
df_combined.index.name = "index"

# Print the resulting DataFrame
#print(df_combined)

df_combined.index.unique()

df_combined['Overall'].iloc[1]

df_combined.to_csv('BERToutput.csv')

df_combined

def compute_precision_recall(df, column_name):
    precisionlst=[]
    recalllst=[]
    fscorelst=[]

    for i in range(0,20):
        # define the classification report as a string
        report_str = df[column_name][i]

        # split the report into lines
        print(column_name)
        lines = report_str.split('\n')
        print(lines)
        # extract precision and recall for the true class
        #true_class_row = [line for line in lines if line.startswith('1 ')][0]
        print(line.strip() for line in lines if line.strip().startswith('1.0 ') or line.strip().startswith('1'))
        #true_class_row = [line.strip() for line in lines if line.strip().startswith('1 ')][0]
        true_class_row = [line for line in lines if line.strip().startswith('1.0 ') or line.strip().startswith('1')][0]
    
        precision = float(true_class_row.split()[1])
        precisionlst.append(precision)
        recall = float(true_class_row.split()[2])
        recalllst.append(recall)
        fscore= float(true_class_row.split()[3])
        fscorelst.append(fscore)

        # print("Precision for {}[{}] is {:.2f}".format(column_name, i, precision))
        # print("Recall for {}[{}] is {:.2f}".format(column_name, i, recall))
        # print("Fscore for {}[{}] is {:.2f}".format(column_name, i, fscore))

    print(precisionlst)
    df_boxplot.loc[0:19, column_name]=precisionlst
    print(recalllst)
    df_boxplot.loc[20:39, column_name]=recalllst
    print(fscorelst)
    df_boxplot.loc[40:59, column_name]=fscorelst
    print('____________')

def Average(lst):
    return sum(lst) / len(lst)

for col in df.columns:
  compute_precision_recall(df, col)

df_boxplot = pd.DataFrame(columns=['Overall','Measurement','Data','Time Constraint','Temperature','Mass','Size','Non-labelData','LabelData','Convention'],index=range(60))

df_boxplot

df_boxplot.to_csv('df_boxplotBERTbase.csv')

df_BERTbase=pd.read_csv('df_boxplotBERTbase.csv')
df_BERTlarge=pd.read_csv('df_boxplotBERTlarge.csv')
# df_ALBERT=pd.read_csv('df_boxplotALBERT.csv')
# df_RoBERTa=pd.read_csv('df_boxplotRoBERTa.csv')
# df_BiLSTM=pd.read_csv('df_boxplotBiLSTM.csv')
# df_Keyword=pd.read_csv('df_boxplotKeyword.csv')

df_BERTbase = df_BERTbase.drop('Unnamed: 0', axis=1)
df_BERTlarge = df_BERTlarge.drop('Unnamed: 0', axis=1)
# df_ALBERT = df_ALBERT.drop('Unnamed: 0', axis=1)
# df_RoBERTa = df_RoBERTa.drop('Unnamed: 0', axis=1)
# df_BiLSTM = df_BiLSTM.drop('Unnamed: 0', axis=1)
# df_Keyword = df_Keyword.drop('Unnamed: 0', axis=1)

# df_FDA=pd.read_csv('df_boxplotFDA.csv')
# df_nonFDA=pd.read_csv('df_boxplotnonFDA.csv')
# df_FDA = df_FDA.drop('Unnamed: 0', axis=1)
# df_nonFDA = df_nonFDA.drop('Unnamed: 0', axis=1)

"""**statistical tests**"""

#wilcoxcon significance test
from scipy.stats import ranksums

def wilcoxcon(lst1,lst2):

  # perform the Wilcoxon rank-sum test
  statistic, p_value = ranksums(lst1, lst2)
  # print the test results
  # print("Wilcoxon rank-sum test:")
  return(p_value)

#asymetric vargha delany significance test 
def Average(lst):
    return sum(lst) / len(lst)
def a12(lst1,lst2,rev=True):
      if Average(lst1) < Average(lst2):
        rev=False

      more = same = 0.0
      for x in lst1:
          for y in lst2:
              if   x==y : same += 1
              elif rev     and x > y : more += 1
              elif not rev and x < y : more += 1
      res = (more + 0.5*same)  / (len(lst1)*len(lst2))
      if   0.71 <res :
        description = 'Large'
      elif 0.64 <res <=0.71:
        description = 'Medium'
      elif 0.56 <res <= 0.64:
        description = 'Small'
      elif res <= 0.56:
        description = 'negligible'

      if rev==False:
        res=1-res
        if res<0.29:
          description = 'Large'
        elif 0.29<=res<0.36:
          description = 'Medium'
        elif 0.36<=res<0.44:
          description = 'Small'
        elif 0.44<=res:
          description = 'negligible'
      return res, description

df_statistic = pd.DataFrame(columns=['Measurement','Temperature','Mass','Size','Time Constraint','Non-labelData','labelData','Data','Overall','Convention'],index=range(3))
print(df_statistic)

for col in columns:
  print(col)
  for i in range(3):
    if i==0:
      p_value=wilcoxcon(df_BERTbase[col].iloc[:20], df_BERTlarge[col].iloc[:20])
      effect_size,effect_size1=a12(df_BERTbase[col].iloc[:20], df_BERTlarge[col].iloc[:20],True)

      # Write the concatenated value to a specific cell in the dataframe
    if i==1:
      p_value=wilcoxcon(df_BERTbase[col].iloc[20:40], df_BERTlarge[col].iloc[20:40])
      effect_size,effect_size1=a12(df_BERTbase[col].iloc[20:40], df_BERTlarge[col].iloc[20:40],True)

    if i==2:
      p_value=wilcoxcon(df_BERTbase[col].iloc[40:60], df_BERTlarge[col].iloc[40:60])
      effect_size,effect_size1=a12(df_BERTbase[col].iloc[40:60], df_BERTlarge[col].iloc[40:60],True)

      
    cell_value = f'p-value: {p_value}, Effect size: {effect_size}, {effect_size1}'
    df_statistic.loc[i,col] = cell_value

df_statistic.to_csv("statBERTbase-BERTlarge.csv")

import os
import shutil

# create directory if it doesn't already exist
if not os.path.exists("/BERT-large"):
    os.makedirs("/BERT-large")

# loop through all files in the current directory
#move reports to their dir
for file in os.listdir("."):
    if file.startswith("BERT") and os.path.isfile(file):
        shutil.move(os.path.join('.', file), os.path.join("/BERT-large", file))