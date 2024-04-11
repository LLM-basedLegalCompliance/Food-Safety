## Imports
import spacy, nltk, os, re, en_core_web_md
from os.path import join
import numpy as np
from string import punctuation
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# from nltk.stem import WordNetLemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
nltk.download(['punkt','stopwords','wordnet','omw-1.4'])
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
nlp = en_core_web_md.load()
import random
from enum import Enum

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
    seed=42,
    sampling = Sampling.NoSampling, #Sampling.UnderSampling,Sampling.OverSampling

 )
clazz = "Measurement" # class to train classification on

config_data = Config(
    label_column = clazz,
)
def load_data(filename):

  df=pd.read_excel(filename)
  
  return df

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**31)
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)             

    return seed

set_seed(config.seed)

df1 = load_data('Data/SFCR.xlsx')
df2= load_data('Data/Annotation.xlsx')
df=df1.append(df2)

df[config_data.label_column] = df[config_data.label_column].fillna(0)
df = pd.concat([df])[['Statement',config_data.label_column]]
df = df.dropna()

sep_test=df[len(df1):]
data=df[:len(df1)]

data=data.sample(frac=1,random_state=config.seed).reset_index(drop=True)
sep_test=sep_test.sample(frac=1,random_state=config.seed).reset_index(drop=True)

data1 = data['Statement']
data1

"""### Keyword Method 1 - Using WikidoMiner"""

def stemlemma(text):
  return ' '.join([stemmer.stem(wordnet_lemmatizer.lemmatize(word)) for word in word_tokenize(text.lower())])

"""Optional"""

def buildTFIDFvector(docs,use_ngrams=True,ngrams=4):
  if use_ngrams:
    vectorizer = TfidfVectorizer(ngram_range=(1,ngrams),min_df=0,stop_words=stopwords.words('english'))
  else:
    vectorizer = TfidfVectorizer(min_df=0,stop_words=stopwords.words('english'))
  vectors = vectorizer.fit_transform(docs)
  return pd.DataFrame(vectors.todense().tolist(), columns=vectorizer.get_feature_names_out())

def buildTFIDF(domains,files,use_ngrams=True,ngrams=3):
  docs={}
  for d in domains:
    docs[d]=stemlemma(' '.join([files[doc] for doc in domains[d]]))
  return buildTFIDFvector(list(docs.values()),use_ngrams=use_ngrams,ngrams=ngrams)

def getTFIDFscore(q,id,tfidf):
  score=0
  for t in q.split():
    if t in tfidf[q].columns:
      score+=tfidf[q][t][id]
  return score

def getAllNPsFromSent(sent,include_nouns=False):
    npstr=[]
    chunks = list(sent.noun_chunks)
    for i in range(len(chunks)):
        np=chunks[i]
        if len(np)==1:
            if np[0].pos_!="NOUN":
                continue
        if np.text.lower() not in npstr:
            npstr.append(np.text.lower())      
        if i < len(chunks)-1:
            np1=chunks[i+1]
            if np1.start-np.end==1:
                if sent.doc[np.end].tag_=="CC":
                    newnp = sent[np.start:np1.end]
                    if newnp.text.lower() not in npstr:
                        npstr.append(newnp.text.lower())
    if include_nouns:
        for t in sent:
            if "subj" in t.dep_ and t.pos_=="NOUN": 
                if t.text.lower() not in npstr:
                    npstr.append(t.text.lower())
    return npstr

def getTopK(di,K=50):
  tempdf=pd.DataFrame.from_dict(di,columns=["tfidf"], orient='index')
  return list(tempdf.sort_values(by=['tfidf'],ascending=False)[:K].index)

def getKeywords(doc,nlp,include_nouns=False,tfidf=[],K=None): # K: free parameter
  keywords=[]
  for s in doc.split('\n'):
    s=nlp(s)
    keywords.extend([n.text for n in list(s.ents)])
    keywords.extend(list(getAllNPsFromSent(s,include_nouns)))
  
  keywords=list(set(keywords))

  keywords_wn={}

  if len(tfidf)>0:
    tfidf_threshold=np.mean([t for t in tfidf if t>0])

  # print("Reached line 19")
  for k in keywords:
    keyword=' '.join([word for word in word_tokenize(k) if not word.lower() in stopwords.words('english')])
    if not wn.synsets(keyword) and keyword.replace(' ','').isalpha() and not keyword.isupper() and not np.array([k.isupper() for k in [ky[:-1] for ky in keyword.split()]]).any():
      keyword=keyword.lower()
      if len(tfidf)>0:
        if stemlemma(keyword) in tfidf.index:# and len(keyword)>2:
          if tfidf[stemlemma(keyword)]>tfidf_threshold:
            if keyword not in keywords_wn:
              keywords_wn[keyword]=tfidf[stemlemma(keyword)]
            else:
              keywords_wn[keyword]=max(keywords_wn[keyword],tfidf[stemlemma(keyword)])
      else:
        keywords_wn[keyword]=0

  if K and len(tfidf)>0:
    return getTopK(keywords_wn,K=K)

  else:
    return list(keywords_wn.keys())

keywords_set1={}
corpora={}
id = 0
for doc in data1:
  keywords = getKeywords(doc,nlp,include_nouns=True,K=50) # extract keywords
  keywords_set1[id]=keywords
  id = id + 1

keywords_set1

with open("keywordset1.json", "w") as file:
    json.dump(keywords_set1, file)

"""##Creating Train data"""

df_train=data

"""##Converting list of keywords into a line"""

for i in range(len(df_train['Statement'])):
  line = ''
  for j in keywords_set1[i]:
    if line == '':
      line = j
    else:
      line = line +' '+ j
  df_train['Statement'][i] = line

df_train

print(df_train.shape)
print(df_train[config_data.label_column].value_counts())

"""##Remove records for which no Keywords were extracted """

df_train = df_train[df_train['Statement'] !='']
df_train = df_train.reset_index(drop=True)

len(df_train)

"""##Get train Class Distribution"""

num_labels = df_train[config_data.label_column].nunique()

print(df_train.shape)
print(df_train[config_data.label_column].value_counts())

"""##Oversample/ Undersample """

def undersample(df_trn, major_label, minor_label):
  sample_size = sum(df_trn[label_column] == minor_label)
  majority_indices = df_trn[df_trn[label_column] == major_label].index
  random_indices = np.random.choice(majority_indices, sample_size, replace=False)
  sample = df_trn.loc[random_indices]
  sample = sample.append(df_trn[df_trn[label_column] == minor_label])
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = seed)
  print(df_trn[label_column].value_counts())
  return df_trn

def oversample(df_trn, major_label, minor_label):
  minor_size = sum(df_trn[label_column] == minor_label)
  major_size = sum(df_trn[label_column] == major_label)
  multiplier = major_size//minor_size
  sample = df_trn
  minority_indices = df_trn[df_trn[label_column] == minor_label].index
  diff = major_size - (multiplier * minor_size)     
  random_indices = np.random.choice(minority_indices, diff, replace=False)
  sample = pd.concat([df_trn.loc[random_indices], sample], ignore_index=True)
  for i in range(multiplier - 1):
    sample = pd.concat([sample, df_trn[df_trn[label_column] == minor_label]], ignore_index=True)
  df_trn = sample
  df_trn = df_trn.sample(frac=1, axis=0, random_state = seed)
  print(df_trn[label_column].value_counts())
  return df_trn

#ov = oversample(tr_df, 0.0, 1.0 )
#ov = ov.reset_index(drop=True)
# x_train = ov['Statement']
# y_train = ov['Measurement']

# x_test = te_df['Statement']
# y_test = te_df['Measurement']

x_train= df_train['Statement']
y_train = df_train[config_data.label_column]

x_test= sep_test['Statement']
y_test = sep_test[config_data.label_column]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(x_train).toarray()
X_test = vectorizer.transform(x_test).toarray()

len(X_test)

"""##Training and Evaluating using Rndom Forest Classifier"""

classifier = RandomForestClassifier(random_state=config.seed)
classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report
y_pred = classifier.predict(X_test)
classification_report = classification_report(y_test, y_pred)
print(classification_report)
