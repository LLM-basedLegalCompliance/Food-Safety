## Imports
import os
import time
import numpy as np
import pandas as pd
import torch
from enum import Enum
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support,f1_score
from sklearn.utils.multiclass import unique_labels
from pytorch_transformers import AdamW
from fastprogress import master_bar, progress_bar
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import random

os.environ['PYTHONHASHSEED']=str(0)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def get_memory_usage():
    return torch.cuda.memory_allocated(device)/1000000

def get_memory_usage_str():
    return 'Memory usage: {:.2f} MB'.format(get_memory_usage())

cuda_available = torch.cuda.is_available()
if cuda_available:
    curr_device = torch.cuda.current_device()
    print("device:",torch.cuda.get_device_name(curr_device))
device = torch.device("cuda" if cuda_available else "cpu")

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
    model_name="albert-base-v1", 
    bs=16, 
    max_seq_len= 512, 
    seed=512,
    sampling = Sampling.NoSampling, #Sampling.UnderSampling, Sampling.NoSampling, Sampling.OverSampling
)

clazz="Measurement"

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
    torch.backends.cudnn.deterministic = True
    
    return seed

set_seed(config.seed)

"""To import the dataset, first we have to either load the data set from zenodo (and unzip the needed file) or connect to our Google drive (if data should be loaded from gdrive). To connect to our Google drive, we have to authenticate the access and mount the drive.

## Data
"""

def load_data(filename):
  df=pd.read_excel(filename)

  return df

df1 = load_data('Data/SFCR.xlsx')
df2= load_data('Data/Annotation.xlsx')
df=df1.append(df2)

df = pd.concat([df])[['Statement',config_data.label_column]]
df[config_data.label_column] = df[config_data.label_column].fillna(0)
df = df.dropna()

#@title Create the dictionary that contains the labels along with their indices. This is useful for evaluation and similar. {display-mode: "form"}
def create_label_indices(df):
    #prepare label
    labels = ['not_' + config_data.label_column, config_data.label_column]
  
    #create dict
    labelDict = dict()
    for i in range (0, len(labels)):
        labelDict[i] = labels[i]
    return labelDict

label_indices = create_label_indices(df)

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

def split_dataframe(df, train_size = 0.80, random_state =config.seed):
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

sep_test=df[len(df1):]
df=df[:len(df1)]

df=df.sample(frac=1,random_state=config.seed).reset_index(drop=True)
sep_test=sep_test.sample(frac=1,random_state=config.seed).reset_index(drop=True)

config.num_labels = df[config_data.label_column].nunique()
print('config.num_labels',config.num_labels)

print('df.shape',df.shape)
print(df[config_data.label_column].value_counts())

#ov = oversample(df, 0.0, 1.0 )
#un= undersample(df, 0.0, 1.0 )
#train, test = split_dataframe(ov)
train, test=split_dataframe(df)

#full_train = ov.reset_index(drop=True) # Contains full training data, to be used after hyper parameter tuning
full_train=df.reset_index(drop=True)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
sep_test = sep_test.reset_index(drop=True)

class ALBERTInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, label_id=None):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        

def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    #print(example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(text)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        label_id = label

        input_items.append(
            ALBERTInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label_id))
      
    return input_items

full_train_X = full_train['Statement'] # Full train to be used only after hyper parameter tuning
full_train_Y= full_train[config_data.label_column]

train_X = train['Statement']
train_Y= train[config_data.label_column]

val_X = test['Statement']
val_Y=test[config_data.label_column]

test_X = sep_test['Statement']
test_Y= sep_test[config_data.label_column]

"""## Create and train the learner/classifier

"""
def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask, labels=label_ids)
        loss = outputs[0]
        logits = outputs[1]

        y_pred = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(y_pred)
        correct_labels += list(label_ids)
        
        eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    print("Eval loss", eval_loss)
    print("Classification Report: ")
    print(classification_report(correct_labels, predicted_labels))
    clsf_report = pd.DataFrame(classification_report(correct_labels,predicted_labels,output_dict=True)).transpose()
    
    value=str(config.seed)+" "+clazz
    file_name = f"BERT Classification Report_{value}.csv"
    # Write classification report to file
    clsf_report.to_csv(file_name, index=True)

    return eval_loss,correct_labels, predicted_labels

bert_tok = AlbertTokenizer.from_pretrained(config.model_name)

train_features = convert_examples_to_inputs(train_X, train_Y, config.max_seq_len, bert_tok, verbose=0)
test_features = convert_examples_to_inputs(test_X, test_Y, config.max_seq_len, bert_tok, verbose=0)
val_features = convert_examples_to_inputs(val_X, val_Y, config.max_seq_len, bert_tok, verbose=0)
full_train_features = convert_examples_to_inputs(full_train_X, full_train_Y, config.max_seq_len, bert_tok, verbose=0)

def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader

train_loader = get_data_loader(train_features, config.max_seq_len, config.bs, shuffle=False)
valid_loader = get_data_loader(val_features, config.max_seq_len, config.bs, shuffle=False)
test_loader = get_data_loader(test_features, config.max_seq_len, config.bs, shuffle=False)
full_train_loader = get_data_loader(full_train_features, config.max_seq_len, config.bs, shuffle=False)

def train_and_validate(params):
  n_epochs = params['n_epochs']
  learning_rate = params['learning_rate']

  model = AlbertForSequenceClassification.from_pretrained(config.model_name)

  model.cuda()
  optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

  train_loss = []
  valid_loss = []

  for epoch in range(n_epochs):
      start_time = time.time()
      # Set model to train configuration
      model.train()
      avg_loss = 0.
      
      for i, batch in enumerate(train_loader):
          batch = tuple(t.cuda() for t in batch)
          input_ids, input_mask, label_ids = batch
          outputs = model(input_ids, attention_mask=input_mask, labels=label_ids)
          loss = outputs[0]
          logits = outputs[1]
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          avg_loss += loss.mean().item()
      
      # Set model to validation configuration -Doesn't get trained here
      model.eval()        
      avg_val_loss = 0.
      val_preds = np.zeros(len(val_X))
      
      for i, batch in enumerate(valid_loader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
          outputs = model(input_ids, attention_mask=input_mask,labels=label_ids)

        y_pred = np.argmax(outputs[1].to('cpu'), axis=1)
        avg_val_loss += outputs[0].mean().item()
        # keep/store predictions
        val_preds[i * config.bs:(i+1) * config.bs] = y_pred
      
      # Check Accuracy
      val_accuracy = sum(val_preds==val_Y)/len(val_Y)
      train_loss.append(avg_loss)
      valid_loss.append(avg_val_loss)
      elapsed_time = time.time() - start_time 
      print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))
  f_score = f1_score(val_Y, val_preds, average='micro')
  return f_score

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 2e-5, 2e-4),
              'optimizer': trial.suggest_categorical("optimizer", ["AdamW","SGD"]),
              'n_epochs': trial.suggest_int("n_epochs", 10, 35),
              }
    f_score = train_and_validate(params)

    return f_score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)
best_trial = study.best_trial

"""# **Assuming that the following is the best set of parameters**

"""
def save_checkpoint(filename, epoch, model):
    state = {
        'epoch': epoch,
        'model': model,
        }
    torch.save(state, filename)

"""# **Train on full dataset after reaching best parameters**

"""
def train_on_full_data(params):
  n_epochs = params['n_epochs']
  learning_rate = params['learning_rate']

  model = AlbertForSequenceClassification.from_pretrained(config.model_name)
  
  model.cuda()
  optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

  train_loss = []

  for epoch in range(n_epochs):
      start_time = time.time()
      # Set model to train configuration
      model.train()
      avg_loss = 0.
      
      for i, batch in enumerate(full_train_loader):
          batch = tuple(t.cuda() for t in batch)
          input_ids, input_mask, label_ids = batch
          outputs = model(input_ids, attention_mask=input_mask, labels=label_ids)
          loss = outputs[0]
          logits = outputs[1]
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          avg_loss += loss.mean().item()
          torch.cuda.empty_cache()

      train_loss.append(avg_loss)
      elapsed_time = time.time() - start_time 
      
      print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, elapsed_time))

  file_name = "albert_model"
  save_checkpoint(file_name, epoch, model)

  return evaluate(model, test_loader)

print(train_on_full_data(best_trial.params))

current_best_params = {'learning_rate': 4.197010547755772e-05,
                       'optimizer': 'AdamW',
                       'n_epochs': 1}
print(train_on_full_data(current_best_params))
