## Imports
import os
import random
import torch
import time
import pandas as pd
import numpy as np
from aenum import Enum, extend_enum
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')
from collections import Counter
from sklearn.metrics import classification_report,f1_score
from nltk import word_tokenize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.model_selection import train_test_split #StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
from sklearn.decomposition import PCA
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
import optuna
os.environ['PYTHONHASHSEED']=str(0)

"""## Basic Parameters"""

embed_size = 300 # how big is each word vector
max_features = 12000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750 # max number of words in a question to use
batch_size = 16 # how many samples to process at once
debug = 0

"""## Config"""

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

class Sampling(Enum):
  NoSampling = 1
  UnderSampling = 2
  OverSampling = 3

config = Config(
    seed=32768,
    sampling = Sampling.NoSampling, #Sampling.UnderSampling, Sampling.NoSampling, Sampling.OverSampling
)

clazz = 'Measurement' # class to train classification on

config_data = Config(
    label_column = clazz,
)

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

    return seed

set_seed(config.seed)

"""##Load Data"""

def load_data(filename):
    df=pd.read_excel(filename)
    return df

df1 = load_data('Data/SFCR.xlsx')
df2= load_data('Data/Annotation.xlsx')
df=df1.append(df2)

df = pd.concat([df])[['Statement',config_data.label_column]]
df[config_data.label_column] = df[config_data.label_column].fillna(0)
df = df.dropna()

"""##Creating training and validation sets"""

def split_dataframe(df, train_size = 0.80, random_state = config.seed):
    # split data into training and validation set
    df_trn, df_valid = train_test_split(df, stratify = df[config_data.label_column], train_size = train_size, random_state = random_state)
    # apply sample strategy
    sizeOne = sum(df_trn[config_data.label_column] == 1)
    sizeZero = sum(df_trn[config_data.label_column] == 0)
    major_label = 0
    minor_label = 1
    if sizeOne > sizeZero:
      major_label = 1
      minor_label = 0
    if config.sampling == Sampling.UnderSampling:
      df_trn = undersample(df_trn, major_label, minor_label)
    elif config.sampling == Sampling.OverSampling:
      df_trn = oversample(df_trn, major_label, minor_label)
    return df_trn, df_valid

#data, sep_test = split_dataframe(data)
sep_test=df[len(df1):]
data=df[:len(df1)]

"""### Get the Number of instances in each class"""

data = data.sample(frac=1, random_state = config.seed).reset_index(drop=True)
sep_test=sep_test.sample(frac=1,random_state=config.seed).reset_index(drop=True)
# config.num_labels = data[config_data.label_column].nunique()

def create_label_indices(df):
    #prepare label
    labels = ['not_' + config_data.label_column, config_data.label_column]
  
    #create dict
    labelDict = dict()
    for i in range (0, len(labels)):
        labelDict[i] = labels[i]
    return labelDict

label_indices = create_label_indices(df)
print(label_indices)

"""### Oversampling/UnderSampling"""

def undersample(df_trn, major_label, minor_label):
  sample_size = sum(df_trn[config_data.label_column] == minor_label)
  majority_indices = df_trn[df_trn[config_data.label_column] == major_label].index
  random_indices = np.random.choice(majority_indices, sample_size, replace=False)
  sample = df_trn.loc[random_indices]
  sample = sample.append(df_trn[df_trn[config_data.label_column] == minor_label])
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

def oversample(df_trn, major_label, minor_label):
  minor_size = sum(df_trn[config_data.label_column] == minor_label)
  major_size = sum(df_trn[config_data.label_column] == major_label)
  multiplier = major_size//minor_size
  sample = df_trn
  minority_indices = df_trn[df_trn[config_data.label_column] == minor_label].index
  diff = major_size - (multiplier * minor_size)     
  random_indices = np.random.choice(minority_indices, diff, replace=False)
  sample = pd.concat([df_trn.loc[random_indices], sample], ignore_index=True)
  for i in range(multiplier - 1):
    sample = pd.concat([sample, df_trn[df_trn[config_data.label_column] == minor_label]], ignore_index=True)
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = config.seed)
  print(df_trn[config_data.label_column].value_counts())
  return df_trn

# ov = oversample(df, 0.0, 1.0 )
# un= undersample(df, 0.0, 1.0 )

"""##Creating training and Validation sets"""

#train, test = split_dataframe(un)
train, test = split_dataframe(data)

#full_train = un.reset_index(drop=True) # Contains full training data, to be used after hyper parameter tuning
full_train=data.reset_index(drop=True)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
sep_test = sep_test.reset_index(drop=True)

print("Length of training data after oversampling:",len(train))
print("Length of validation data after oversampling:",len(test))

"""##Assigning training , test and validation datasets to variables"""

full_train_X = full_train['Statement'] # Full train to be used only after hyper parameter tuning
full_train_Y= full_train[config_data.label_column]

train_X = train['Statement']
train_Y= train[config_data.label_column]

val_X = test['Statement']
val_Y=test[config_data.label_column]

test_X = sep_test['Statement']
test_Y= sep_test[config_data.label_column]

"""##Tokenizing the initial training and validation data for hyper parameter tuning"""

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)

# Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)

"""##Encoding labels for initial training, validation and test data for hyper parameter tuning"""

le = LabelEncoder()
train_Y = le.fit_transform(train_Y.values)
val_Y = le.transform(val_Y.values)

le.classes_

"""## Load Glove Embeddings

"""
import zipfile

zip_file = "path to glove.840B.300d.txt.zip"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("./")

## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):
    EMBEDDING_FILE = './glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

#missing entries in the embedding are set using np.random.normal so we have to seed here too
if debug:
    embedding_matrix = np.random.randn(12000,300)
else:
    embedding_matrix = load_glove(tokenizer.word_index)

"""Load glove embeddings for the corresponding tokenizer"""

np.shape(embedding_matrix)

"""## Pytorch Model - BiLSTM"""

class BiLSTM(nn.Module):
    
    def __init__(self, max_features, embed_size, hidden_size, drp, n_layers, n_classes):
        super(BiLSTM, self).__init__()
        # self.hidden_size = 64
        # drp = 0.1
        # n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*4 , hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(hidden_size, n_classes)


    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

"""## Train and validation function"""

def train_and_validate(params):
  n_epochs = params['n_epochs']
  hidden_size = params['hidden_size']
  drop_rate = params['drop_rate']
  n_layers = params['n_layers']
  n_classes = len(le.classes_)
  learning_rate = params['learning_rate']

  model = BiLSTM(max_features, embed_size, hidden_size, drop_rate, n_layers, n_classes)
  loss_fn = nn.CrossEntropyLoss(reduction='sum')
  model.cuda()
  # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
  optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])


  # Load train and test in CUDA Memory
  x_train = torch.tensor(train_X, dtype=torch.long).cuda()
  y_train = torch.tensor(train_Y, dtype=torch.long).cuda()
  x_cv = torch.tensor(val_X, dtype=torch.long).cuda()
  y_cv = torch.tensor(val_Y, dtype=torch.long).cuda()

  # Create Torch datasets
  train = torch.utils.data.TensorDataset(x_train, y_train)
  valid = torch.utils.data.TensorDataset(x_cv, y_cv)


  # Create Data Loaders
  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
  valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

  train_loss = []
  valid_loss = []

  for epoch in range(n_epochs):
      start_time = time.time()
      # Set model to train configuration
      model.train()
      avg_loss = 0.  
      for i, (x_batch, y_batch) in enumerate(train_loader):
          # Predict/Forward Pass
          y_pred = model(x_batch)
          # Compute loss
          loss = loss_fn(y_pred, y_batch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          avg_loss += loss.item() / len(train_loader)
      
      # Set model to validation configuration -Doesn't get trained here
      model.eval()        
      avg_val_loss = 0.
      val_preds = np.zeros((len(x_cv),len(le.classes_)))
      
      for i, (x_batch, y_batch) in enumerate(valid_loader):
          y_pred = model(x_batch).detach()
          avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
          # keep/store predictions
          val_preds[i * batch_size:(i+1) * batch_size] =F.softmax(y_pred).cpu().numpy()
      
      # Check Accuracy
      val_accuracy = sum(val_preds.argmax(axis=1)==val_Y)/len(val_Y)
      train_loss.append(avg_loss)
      valid_loss.append(avg_val_loss)
      elapsed_time = time.time() - start_time 
      print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))
  f_score = f1_score(val_Y, val_preds.argmax(axis=1), average='micro')
  #print(evaluate(model))
  return f_score

"""##Evaluation Function"""

def evaluate(model):

  x_test = torch.tensor(test_X, dtype=torch.long).cuda()
  y_test = torch.tensor(test_Y, dtype=torch.long).cuda()
  test = torch.utils.data.TensorDataset(x_test, y_test)
  test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
  test_preds = np.zeros((len(x_test),len(le.classes_)))
  
  avg_test_loss = 0
  for i, (x_batch, y_batch) in enumerate(test_loader):
      y_pred = model(x_batch).detach()
      # avg_test_loss += loss_fn(y_pred, y_batch).item() / len(test_loader)
      # keep/store predictions
      test_preds[i * batch_size:(i+1) * batch_size] =F.softmax(y_pred).cpu().numpy()
  
  # Check Accuracy
  # print(len(test_Y))
  # print(test_preds.argmax(axis=1))
  test_accuracy = sum(test_preds.argmax(axis=1)==test_Y)/len(test_Y)
  print(classification_report(test_Y, test_preds.argmax(axis=1), target_names=[label_indices[0],label_indices[1]]))
  
  return test_accuracy

def save_checkpoint(filename, epoch, model):
    state = {
        'epoch': epoch,
        'model': model,
        #'optimizer': optimizer,
        }
    torch.save(state, filename)

"""## Train on Full Data function"""

def train_on_full_data(params):
  n_epochs = params['n_epochs']
  hidden_size = params['hidden_size']
  drop_rate = params['drop_rate']
  n_layers = params['n_layers']
  learning_rate = params['learning_rate']
  n_classes = len(le.classes_)

  model = BiLSTM(max_features, embed_size, hidden_size, drop_rate, n_layers, n_classes)
  loss_fn = nn.CrossEntropyLoss(reduction='sum')
  model.cuda()
  # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
  optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])


  x_train = torch.tensor(full_train_X, dtype=torch.long).cuda()
  y_train = torch.tensor(full_train_Y, dtype=torch.long).cuda()
  train = torch.utils.data.TensorDataset(x_train, y_train)
  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

  train_loss = []

  for epoch in range(n_epochs):
      start_time = time.time()
      # Set model to train configuration
      model.train()
      avg_loss = 0.
      
      for i, (x_batch, y_batch) in enumerate(train_loader):
          # Predict/Forward Pass
          y_pred = model(x_batch)
          # Compute loss
          loss = loss_fn(y_pred, y_batch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          avg_loss += loss.item() / len(train_loader)

      train_loss.append(avg_loss)
      elapsed_time = time.time() - start_time 
      
      print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, elapsed_time))
  file_name = 'saved_weights.pt'
  
  save_checkpoint(file_name, epoch, model)
  return evaluate(model)

"""## Hyper Parameter Optimization using Bayesian Optimization"""

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 2e-5, 1e-2),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD","AdamW","RMSprop"]),
              'n_layers': trial.suggest_int("n_layers", 1, 4),
              'drop_rate':trial.suggest_float("drop_rate", 0.1, 0.5),
              'hidden_size': trial.suggest_int("hidden_size", 48, 128),
              'n_epochs': trial.suggest_int("n_epochs", 10, 50),
              }
    
    
    f_score = train_and_validate(params)

    return f_score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=35)
best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

"""##Declaring the best parameters"""

best_trial

"""##Tokenizing the Full Training Data"""

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(full_train_X))
full_train_X = tokenizer.texts_to_sequences(full_train_X)
test_X = tokenizer.texts_to_sequences(test_X)

# Pad the sentences 
full_train_X = pad_sequences(full_train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

"""##Encoding the labels for the Full Training """

le = LabelEncoder()
full_train_Y = le.fit_transform(full_train_Y.values)
test_Y = le.transform(test_Y.values)

"""##Laod Glove for the corresponding tokenizer"""

# missing entries in the embedding are set using np.random.normal so we have to seed here too
if debug:
    embedding_matrix = np.random.randn(12000,300)
else:
    embedding_matrix = load_glove(tokenizer.word_index)

"""##Training and testing the model with the best parameters """

train_on_full_data(best_trial.params)

"""##Fine-tuning the model on the current best parameeres without Hyperparameter Optimization"""

best_trial={
'learning_rate': 0.00040981421211932936,
'optimizer': 'AdamW',
'n_layers': 3,
'drop_rate': 0.3457129079784031,
'hidden_size': 85,
'n_epochs': 11
}
train_on_full_data(best_trial)
