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
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import random

cuda_available = torch.cuda.is_available()
if cuda_available:
    curr_device = torch.cuda.current_device()
    print("device:",torch.cuda.get_device_name(curr_device))
device = torch.device("cuda" if cuda_available else "cpu")
os.environ['PYTHONHASHSEED']=str(0)

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
    model_name="bert-base-cased", # bert_large_cased
    bs=16, 
    max_seq_len= 512, 
    seed=512,
    sampling = Sampling.NoSampling, #Sampling.UnderSampling, Sampling.OverSampling

 )
clazz = "Measurement" # class to train classification on

config_data = Config(
    label_column = clazz,
    model_name= 'FoodSafetyLM'
)

load_from_gdrive = False # True, if you want to use Google Drive; else, False

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

def load_data(filename):
  df=pd.read_excel(filename)
  
  return df

df1 = load_data('Data/SFCR.xlsx')
df2= load_data('Data/Annotation.xlsx')
df=df1.append(df2)

df[config_data.label_column] = df[config_data.label_column].fillna(0)
df = pd.concat([df])[['Statement',config_data.label_column]]
df = df.dropna()

def create_label_indices(df):
    #prepare label
    labels = ['not_' + config_data.label_column, config_data.label_column]
  
    #create dict
    labelDict = dict()
    for i in range (0, len(labels)):
        labelDict[i] = labels[i]
    return labelDict

label_indices = create_label_indices(df)

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

sep_test=df[len(df1):]
data=df[:len(df1)]

data=data.sample(frac=1,random_state=config.seed).reset_index(drop=True)
sep_test=sep_test.sample(frac=1,random_state=config.seed).reset_index(drop=True)

"""##Assigning training , test and validation datasets to variables"""

train, test = split_dataframe(data)

full_train = data.reset_index(drop=True) # Contains full training data, to be used after hyper parameter tuning
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
sep_test = sep_test.reset_index(drop=True)

full_train_X = full_train['Statement'] # Full train to be used only after hyper parameter tuning
full_train_Y= full_train[config_data.label_column]

train_X = train['Statement']
train_Y= train[config_data.label_column]

val_X = test['Statement']
val_Y= test[config_data.label_column]

test_X = sep_test['Statement']
test_Y= sep_test[config_data.label_column]

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id=None):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    
    return input_items

bert_tok = BertTokenizer.from_pretrained(config.model_name)

train_features = convert_examples_to_inputs(train_X, train_Y, config.max_seq_len, bert_tok, verbose=0)
test_features = convert_examples_to_inputs(test_X, test_Y, config.max_seq_len, bert_tok, verbose=0)
val_features = convert_examples_to_inputs(val_X, val_Y, config.max_seq_len, bert_tok, verbose=0)
full_train_features = convert_examples_to_inputs(full_train_X, full_train_Y, config.max_seq_len, bert_tok, verbose=0)

def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader

train_loader = get_data_loader(train_features, config.max_seq_len, config.bs, shuffle=False)
valid_loader = get_data_loader(val_features, config.max_seq_len, config.bs, shuffle=False)
test_loader = get_data_loader(test_features, config.max_seq_len, config.bs, shuffle=False)
full_train_loader = get_data_loader(full_train_features, config.max_seq_len, config.bs, shuffle=False)

"""## Train and validation function"""

def train_and_validate(params):
  n_epochs = params['n_epochs']
  learning_rate = params['learning_rate']

  model = BertForSequenceClassification.from_pretrained(config.model_name)
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
          input_ids, input_mask, segment_ids, label_ids = batch
          outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
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
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
          outputs = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)

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

"""### Hyper Parameter Optimization using Bayesian Optimization"""

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

"""##Declaring the best parameters

"""

best_trial

def save_checkpoint(filename, epoch, model):
    state = {
        'epoch': epoch,
        'model': model,
        #'optimizer': optimizer,
        }
    torch.save(state, filename)

"""##Evaluation Function"""

def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)
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

    return correct_labels, predicted_labels

"""## Train on Full Data function"""

def train_on_full_data(params):
  n_epochs = params['n_epochs']
  learning_rate = params['learning_rate']

  model = BertForSequenceClassification.from_pretrained(config.model_name)
  model.cuda()
  optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

  train_loss = []
  #print("Model", model)
  for epoch in range(n_epochs):
      start_time = time.time()
      # Set model to train configuration
      model.train()
      avg_loss = 0.
      
      for i, batch in enumerate(full_train_loader):
          batch = tuple(t.cuda() for t in batch)
          input_ids, input_mask, segment_ids, label_ids = batch
          outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
          loss = outputs[0]
          logits = outputs[1]
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          avg_loss += loss.mean().item()
      train_loss.append(avg_loss)
      elapsed_time = time.time() - start_time 
      
      print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                  epoch + 1, n_epochs, avg_loss, elapsed_time))
  file_name = 'saved_weights.pt'
  
  save_checkpoint(file_name, epoch, model)
  return evaluate(model, test_loader)

print(train_on_full_data(best_trial.params))

"""##Fine-tuning the model on the current best parameeres without Hyperparameter Optimization"""

current_best_params = {'learning_rate': 4.197010547755772e-05,
                       'optimizer': 'AdamW',
                       'n_epochs': 1}
print(train_on_full_data(current_best_params))

"""##Prediction"""

def evaluate_entire_data(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []
    
    start_time = time.time()
    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids)
        
     
        logits = outputs[0]
        # print("Logits ", logits)
        y_pred = np.argmax(logits.to('cpu'), axis=1)
        
        predicted_labels += list(y_pred)
    predicted_labels = np.array(predicted_labels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")
        
    return predicted_labels

list_input = df2['Statement'].tolist()
test_data = list_input

path = 'saved_weights.pt'
# start_time = time.time()
checkpoint = torch.load(path)
model = checkpoint.get("model")
# elapsed_time = time.time() - start_time
# print("Time to load model: ", elapsed_time, " seconds")

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id=None):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_test_data(examples, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
  
    #print(example_labels)
    for (ex_index, text) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(text)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

        
    return input_items

def get_test_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader

bert_tok = BertTokenizer.from_pretrained(config.model_name,)

test_features = convert_test_data(test_data, config.max_seq_len, bert_tok, verbose=0)
test_loader = get_test_loader(test_features, config.max_seq_len, config.bs, shuffle=True)
model.cuda()
print(evaluate_entire_data(model,test_loader))

"""####method for prediction [this method is slower compared to the first one but provides good chance for individual observations]"""

class Prediction:
    def __init__(self):
        path = 'saved_weights.pt'
        checkpoint = torch.load(path,map_location='cpu')
        self.predictor = checkpoint.get('model')
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        #self.tag = checkpoint.get("id_map")

    def predict(self,text):
        input_ids = self.tokenizer.encode(text)
        if len(input_ids) > config.max_seq_len:
            input_ids = input_ids[:config.max_seq_len]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (config.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == config.max_seq_len
        assert len(input_mask) == config.max_seq_len

        t_input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        t_input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
          outputs = self.predictor(t_input_ids, attention_mask=t_input_mask)
        print(outputs)
        logits = outputs[0]
        #print("Logits ", logits)
        pred = np.argmax(logits.to('cpu'), axis=1)
        return pred

pred = Prediction()
list_input = ["Poultry Carcasses  with weight between 1.8 Kg to 3.6 kg must be continuously chilled to reach less than 4°C in an additional time of 4 hours.",
              "The licence holder must submit to the veterinary inspector a written protocol for each all poultry products which will be crust frozen outlining the type and temperature of the chilling procedure/refrigerant.",
              "obtain a document that sets out multiple information.",
              "(2) Any person who sells a food at retail, other than a restaurant or other similar enterprise that sells the food as a meal or snack, must prepare and keep documents that include the information specified in paragraphs (1)(a), (c) and (d)."]

for item in list_input:
    pred_val = pred.predict(item)
    print(pred_val)

