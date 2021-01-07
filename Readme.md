

## Citations

Please note that part of the pre-processing involves sentiment polarity dictionaries which were created by:

William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky
Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. ArXiv preprint (arxiv:1606.02820). 2016. 

Download the sentiment polarity dictionaries via

[Sentiment scores for frequent words](https://nlp.stanford.edu/projects/socialsent/files/socialsent_hist_freq.zip)
[Sentiment scores for adjectives](https://nlp.stanford.edu/projects/socialsent/files/socialsent_hist_adj.zip)

Save the unzipped files in the polarity directory to get the following structure
```
automl_vs_hyperdrive/
│
└── data/
  └── polarity_data
            ├── socialsent_hist_adj 
            │      └── adjectives
            │            └── * many_tsv_files
            │
            └── socialsent_hist_freq
                   └── frequent_words
                         └── * many_tsv_files
```

## Repository setup


### 1. Create a virtual environment

```
conda create --name automl_vs_hyperdrive python=3.7
```

### 2. Install requirements.txt
```
pip install -r requirements.txt
```
### 3. Execute 
```
python setup.py develop
```