# Replication Package for EMSE2024: Classifying Requirements-related Provisions in Food-safety Regulations: An LLM-based Approach


## Content description

```bash
.
├── Code
│   ├── RQ1
│   │   ├── BERT
│   │   └── GPT
│   ├── RQ2
│   └── Supplementary Code
├── Data
└── Evaluation Results
    ├── RQ1
    │   └── dfboxplots
    └── RQ2
        └── dfboxplots
```
        
* Code: implementations of all the elements discussed in the paper. The folder is divided into three subfolders: RQ1, RQ2, and Supplementary Code.

    * RQ1: is divided into two subfolders: BERT and GPT.
        * BERT: contains code related to the BERT-based Language Models implemented for answering RQ1. The prerequisite packages are listed in the requirement.txt file.
        *  GPT: contains code related to GPT-3.5 Language Model implemented for answering RQ1. The prerequisite packages are listed in the requirement.txt file.
    * RQ2: contains code for the BiLSTM and Keyword Search baselines and implementation details of these baselines. The prerequisite packages are again listed in the requirement.txt file. 
    * Supplementary Code: contains auxilliary scripts that support the main analyses presented in the paper. The scripts implement 1) the calculation of summary statistics for food-safety provisions (retrieving the content of SFCR and FSRG URLs and deriving various statistics on the number of sentences found in them), 2) the claculation of Accuracy metrics for classification and significance testing (taking classification reports as input and generating a dataframe with Precision, Recall, and F-score values for each label, followed by statistical significance testing between model pairs), 3) boxplot visualization (based on the Accuracy calculation dataframes).

* Evaluation Results: contains two subfolders named RQ1 and RQ2. 
    * RQ1: contains Precision, Recall, F1-Score, and F2-score for different BERT variants, along with BERT variants hyperparameters used in the experiments, statistical significance tests for comparing BERT base against other variants. The dfboxplots subfolder contains the dataframes used for creating boxplots. We also provide details of results for FDA and Non-FDA part of our test data.
    
    * RQ2: contains Precision, Recall, F1-Score, and F2-score for baselines, BiLSTM hyperparameters used in the experiment, statistical significance tests comparing BERT and GPT against baselines. Like in RQ1 the dfboxplots subfolder contains the dataframes used for creating boxplots.
    
* Data: contains our datasets including qualitative data derived from our qualitative coding as well as data curated from third-parties and used as test data in our evaluation (with and without paragraph-level text). This folder further contains the keywordsets used for classifying scarce labels. We also provide the prliminery sheets of our Grounded Theory (Open Coding), and the annotation protocol for third party annotators.

### Execution Instructions

* Create a python environment with the packages listed in: FSR/Code/RQ1/BERT/requirement.txt or FSR/Code/RQ1/GPT/requirement.txt
* Open the environment and proceed to FSR main folder FSR/Code/RQ1/BERT or FSR/Code/RQ1/GPT
* Execute the code BERTbase.py
* For GPT experiments, set your OpenAI key in the GPT.py
* Execute the code GPT.py
  
* For baseline experiments, create a python environment with the packages listed in: FSR/Code/RQ2/requirement.tx
* For BiLSTM experiments you need to download glove.840B.300d and update zip_file = "path to glove.840B.300d.txt.zip" in BiLSTM.py 

## Version History

Initial Release

## Acknowledgments
Redacted for double-anonymous review
