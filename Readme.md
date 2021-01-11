# Sentiment classification for Trip Advisor Hotel Reviews

## Dataset description

The [Kaggle Trip Advisor Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) Dataset comprises 

> _20491_ Hotel Reviews 
> 
> Rating from _1* (worst) to 5* (best rating)_

The Dataset was modified to facilitate the task the Rating column was binned to 3 columns

> * Negative - class 0 (Rating 1* & Rating 2*)
> * Neutral - class 1 (Rating 3*)
> * Positive - class 2 (Rating 4* & 5*)
> 
> This thus becomes a **Sentiment classification task**

However there is a [Class imbalance](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/class_imbalance.png)

To compare the Feature engineering of azureml AutoML with hand-crafted engineering
a [preprocessing script](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/rating_ml_modules/scripts/preprocessing/featurize_dataframe.py) was implemented:

For each hotel review the preprocessing script generates the following features:

> * Ia. Spacy Transformer embedding [en_trf_robertabase_lg](https://spacy.io/models/en-starters#en_trf_robertabase_lg)
> * Ib. Alternatively a character n-gram TFIDF embedding was used
> * II. Review text gength based feature _short_review, long_review_
> * III. Sentiment polarity dictionaries for adjectives/frequent words were mapped to the texts. Minimum, mean and maximum polarity wer determined per review text: _min_adj, max_adj, mean_adj, min_freq_w, max_freq_w, mean_freq_w_
> * IV. LDA topic vectors were fitted for each text. Hyperparameter search for topic coherence yielded 30 topics as an optimal number of topics

This feature-engineered Dataset was used for **Hyperparameter tuning with azureml Hyperdrive**


## Citations & Required Downloads

### Download A - Polarity Dictionaries
> **Please note that part of the pre-processing involves sentiment polarity dictionaries which were created by:**
> 
> _William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky_
> _Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. ArXiv preprint (arxiv:1606.02820). 2016._ 
>
> Download the sentiment polarity dictionaries via:
>
> [Sentiment scores for frequent words](https://nlp.stanford.edu/projects/socialsent/files/socialsent_hist_freq.zip)
>
> [Sentiment scores for adjectives](https://nlp.stanford.edu/projects/socialsent/files/socialsent_hist_adj.zip)
>
####Save the unzipped files in the polarity directory to get the following structure:
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

### Download B - Dataset
> Please note that the modeling is based on the **Kaggle Trip Advisor Reviews Dataset** see citation bellow
> 
> _Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223._
> 
> [Kaggle Trip Advisor Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)
####Save the unzipped files in the polarity directory to get the following structure:
```
automl_vs_hyperdrive/
│
└── data/
  └── datasets
            └── socialsent_hist_adj 
            │      └── adjectives
            │            └── * many_tsv_files
            │
            └── socialsent_hist_freq
                   └── frequent_words
                         └── * many_tsv_files
```

## Repository setup/ Installation steps


* I. Create a virtual environment

```
conda create --name automl_vs_hyperdrive python=3.7
```
* II. Activate your conda environment
```
activate automl_vs_hyperdrive
```
* III. Install requirements.txt
```
pip install -r requirements.txt
```
* IV. Execute setup.py 
```
python setup.py develop
```
* V. Download  _en_trf_robertabase_lg_ spacy model
```
python -m spacy download en_trf_robertabase_lg
```
* VI. Download _en_core_web_md_ spacy model
```
python -m spacy download en_core_web_md
```
### Access to the Dataset
In our workspace the Dataset is:
* 1. manually loaded into the 
Notebook workspace of Azure Machine Learning studio
* 2. uploaded into the azure default datastore    
* 3. Loaded as a Tabular Dataset    
* 4. Registered as a Dataset object    

``` 
hotel_review_dataset = pd.read_csv(filepath_2_dataset)
hotel_review_dataset.to_csv("data/review_dataset.csv", index=False)
datastore = workspace.get_default_datastore()
datastore.upload(src_dir="data", target_path="data")
dataset_training = Dataset.Tabular.from_delimited_files(path = [(datastore, ("data/review_dataset.csv"))])
dataset_training = dataset_training.register(workspace=workspace, name="hotel-review-data", description="Hotel Review Dataset")
```
## Automated ML
The following parameters were set for the AutoML Training Configuration:
* _experiment_timeout_minutes_: was set to prevent the experiment from running for long timer periods with high cost
* _max_concurrent_iterations_: was set to 4 since only 4 compute target nodes are available for paralle child runs
* _primary_metric_: was set to AUC_weighted since this includes a balance between false positive and true positive rate
* _n_cross_validations_: 5 crossvalidations were selected, since this results in a more robust mean/std estimation for each model

* _enable_early_stopping_: to prevent unproductive runs which lead to no improvement and costs
* _compute_target_: needs to be define to perform the AutoML computations
* _task_: needs to be classification since the label column is defining separate classes
* _training_data_: corresponds to the training set
* _label_column_: corresponds to the target/label column defining the separate classes
* _debug_log_: defined to enable detailed logging of automl errors

### Results
The best selected AutoML models were a Voting Ensemble and Random Forest with:

[Best AutoML Model](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/BestAutoMLModel.PNG)
> Voting Ensemble Accuracy: _0.7352_
> RandomForest Accuracy: _0.7352_

* The auto-selected Features were pre-processed by a MinMaxScaler
* The comparison with the Hyperparameter tuning results with AutoML revealed that the feature engineering is a major advantage towards the automated preprocessing thus the results can be improved if the features are provided to AutoML
* Much better results might be obtainable by using a pre-trained neural model e.g. a pre-trained bert-base-uncased 

[AutoML RunDetails partI](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/AutoMLRunDetails1.PNG)
[AutoML RunDetails partII](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/AutoMLRunDetails2.PNG)
[AutoML RunDetails partIII](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/AutoMLRunDetails3.PNG)

> A big pitfall of using Accuracy as a metric to be optimized was that the best 
> model learned to perfectly classify 100% of all examples of class 2 while it failed for 100% of the 
> cases for classes 0 and 1 [Confusion Matrix](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/AutoMLClassImbalance.PNG)
> This can be seen by the confusion matrix 
> In general the Dataset is highly class imbalanced and e.g. an **F1 score or weighted AUC** would have been a better metric
> Also stratified sampling, upsampling, downsampling might help with the class imbalance

## Hyperparameter Tuning
In the previous azureml experiments with AutoML mostly [LightGBM](https://lightgbm.readthedocs.io/en/latest/) was the best performing model.

Also it is widely reported that Gradient Boosting Machines are winning in many Kaggle competition unless competing against a suitable pretrained neural net with sufficient finetuning

To limit the training time I decided against training a neural net and for the well established [XGBoost library](https://xgboost.readthedocs.io/en/latest/)

The following hyperparameters were provided to the XGBoost Model
>* '--max-depth': "How deep are individual trees growing during one round of boosting"
>* '--min-child-weight': "Minimum sum of weight for all observations in a child. Controls overfitting"
>* '--gamma': "Gamma corresponds to the minimum loss reduction required to make a split."
>* '--subsample': "What fraction of samples are randomly sampled per tree.")
>* '--colsample-bytree': "What fraction of feature columns are randomly sampled per tree."
>* '--reg-alpha': "L1 regularization of the weights. Increasing the values more strongly prevents overfitting."
>* '--eta': "Learning rate for XGBoost.")
>* '--seed':  "Random seed."
>* '--num-iterations': "Number of fitting iterations"

The parameters for RandomSearch were selected according to my prior experience and the excellent [XGBoost Tuning Blog Post](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

* XGBoost is very sensitive to tree-centered parameters such as _max_depth, min_child_weight, subsample, colsample_bytree, gamma_

* Also the regularization parameter alpha was tuned 

For Hyperparameter tuning a Random Grid was defined:
>* uniform distribution of the _subsample & gamma parameters_
>* loguniform distribution of _reg-alpha parameter_
>* discrete choice for _max-depth, min-child-weight, colsample-bytree parameters_

``` 
parameter_sampling_grid = RandomParameterSampling(
     {
      "--max-depth": choice(3,4,5,6),
      "--min-child-weight": choice(1,2,3,4,5),
      "--colsample-bytree": uniform(0.8, 1.0),
      "--subsample": uniform(0.7, 1.0),
      "--gamma": uniform(0, 0.4),
      "--reg-alpha": loguniform(-5,-1)
     }
)
``` 

An early termination _BanditPolicy_ was used with an _evaluation interval of 2 and a slack_factor of 0.1_
``` 
from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1)
``` 
* evaluation_interval 2 ==> every 2nd iteration it is checked, whether the termination criterion is met
* slack_factor 0.1 ==> if the evaluation result is 10% worse than the current optimum, the parameter search is aborted

> The _Accuracy score_ was defined as a primary criterion to be maximized during parameter search
> However besides the primary metric, also the weighted F1 score as logged 

The hyperparameter tuning was performed via the [train.py](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/rating_ml_modules/scripts/machine_learning/train.py) script


### XGBoost Hyperparameter Tuning Results

For some plots of the Hyperparameter Search RunDetails see:
* [RandomSearch Parameter Grid Plot](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/HyperparamRun.PNG)
* [Accuracy Plot for Hyperparameter Search](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/HyperparamRun1.PNG)
* [Log of Hyperparameter value configurations with the corresponding scores](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/HyperparamRunWithHyperparams.PNG)
* [Parameter Search RunDetail Logs](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/HyperparamRun2.PNG)

* [Best Model](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/BestHyperDriveModel.PNG)
> Test set accuracy: _0.9189_
> Test set weighted F1 score: _0.9080_

Hyperparameters of the best model:

>* --max-depth 3
>* --min-child-weight 2
>* --gamma 0
>* --subsample 0.9
>* --colsample-bytree 0.8
>* --reg-alpha 1E-05
>* --eta 0.2
>* --seed 42
>* --num-iterations 20
>* --colsample-bytree 0.9501132077820976
>* --gamma 0.16945015198714986 
>* --max-depth 6 
>* --min-child-weight 2
>* --reg-alpha 0.06308908942969567
>* --subsample 0.7683197302311903


## Model Deployment
The best XGBoost model as selected by hyperparameter Randomsearch was deployed as a Webservice.
[Successfull deployment of the best XGBoost model](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/BestModelDeployedAsEndpoint.PNG)

[Endpoint of deployed model](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/DeployedHyperparamModel.PNG)

[Application Insights Logs for best model](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/LoggingEnabledByApplicationInsights.PNG)

[Example Request against best model endpoint](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/ExampleRequestBestModel.PNG)

The scoring script [score.py](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/rating_ml_modules/scripts/machine_learning/score.py) was defined in such a way, that: 

_Negative_ was returned for _class:0_

_Neutral_ was returend for _class:1_

_Positive_ was returned for _class:2_

``` 
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

inference_config = InferenceConfig(entry_script="score.py",
                                   environment=myenv)
service_name = 'xgboost-review-classification'
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[best_model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print("scoring URI: " + service.scoring_uri)
``` 
Application Insights was enabled for logging the status of the Deployed Webservice:
``` 
service.update(enable_app_insights=True)
``` 
An example request was done with the following code:
``` 
import requests
import json

headers = {'Content-Type':'application/json'}
test_sample = json.dumps({'data': [
    positive_example,
    neutral_example,
    negative_example
]})
test_sample = bytes(test_sample, encoding = 'utf8')
service_url = "http://06df2eb2-6456-4d1d-ae18-0470e3d3e11b.southcentralus.azurecontainer.io/score"
response = requests.post(service_url, test_sample, headers=headers)
print("prediction:", response.text)
``` 
## Conclusion
All in all the Hyperparameter Search for an XGBoost model was the most successful with the engineered features.
However the class imbalance made was still not resolvable and the feature engineering required a lot of exploratory 
data analysis. 
Finetuning a pre-trained neural model such as bert-base-uncased, distilbert.... might lead to even better performance but will
also require more training time and more costly GPU resources. 

But the combination of polarity dictionaries, transformer embeddings and topic modeling was a very fruitful representation and 
yielded some reasonable results in particular the test set **F1_score: 0.908**

## How to improve the project in the future
#### 1. Setup an azureml Pipeline

From an azureml perspective a future improvement definitely would be to create 2 azureml Pipelines which include the text preprocessing step followed either by AutoML or Hyperdrive and a final custom evaluation step.
This would enable the identification of a new model if additional training data is available and e.g. a _domain shift_ of the review data is observed over the time. 
An advantage would be that this Pipeline could be easily triggered via an endpoint if the Pipeline is deployed by Publishing.   

#### 2. Deal with class imbalance
##### 2.1 Stratified sampling:
Another improvement of the current approach would be to deal better with the class imbalance e.g. via
stratified sampling
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
```
##### 2.2 Upsampling
Alternatively a random boostrapping upsampling of the underrepresented Negative and Neutral samples could be performed. 
##### 2.3 Different binning of ratings to sentiment categories
A. Furthermore, it has to be considered to bin the classes differently based on the original Rating column the Positive class into 2 classes e.g. 4 = Positive; 5 = Perfect
* class 0 - Negative (1*&2*)
* class 1 - Neutral (3*)
* class 2 - Positive (4*)
* class 3 - Perfect (5*)

B. It could even be considered to merge the ratings 1*, 2*, 3* into a single class and convert the problem into a _binary sentiment classification problem_:
* class 0 - Negative (1*,2*,3*)
* class 1 - Positive (4*, 5*)

Actually the distribution of the example Ratings in the TSNE embedding space for Topic embeddings, Roberta embeddings and TFIDF embeddings show that the current classes (Negative and Neutral) are strongly overlapping which would suggest this approach.
See the next section in the Readme.md about _Extensive Exploratory Data-Analysis and Feature engineering_

#### 3. AutoML with engineered features
For sure it would be a fair comparison to perform the AutoML with the engineered features and not just the raw review texts. Also neural methods should be enabled for AutoML training. 

#### 4. Test a Range of Neural Network architectures
To further improve the performance of the sentiment classifier I would test a range of state of the art neural text classification network architectures which worked very well for some text classification tasks which I was conducting.
Importantly it needs to be considered to perform Hyperdrive based parameter search these models on a GPU compute cluster.
Also the training and evaluation code needs to be refactored to a _train.py_ script to enable hyperdrive based parameter search. 
An important step for training/finetuning those neural text classification methods will be the implementation of DataLoaders which provide mini-batches to the models during the training and evaluation.

##### 4.1 Finetune a _bert-base-uncased_ for sequence classification model
* [Train a BERT base text classifier with a Jupyter Notebook](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=p9MCBOq4xUpr)
* [Here is an excellent introduction to transformer models and a walkthrough for the Jupyter Notebook](https://www.youtube.com/watch?v=x66kkDnbzi4)
A bert-base-uncased transformer model from huggingface can be finetuned on the sequence sentiment classification task.
However the num_labels argument has to be set to 3. I previously have successfully trained some high performance text classification models with a customized version of this
notebook in the past.
```
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 3, # The number of output labels--3 for 3 classes
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
```
##### 4.2 Finetune a _DistilBert_ for sequence classification model
[How to train a DistilBert classifier](https://www.sunnyville.ai/fine-tuning-distilbert-multi-class-text-classification-using-transformers-and-tensorflow/)
A TFDistilBertForSequenceClassification transformer model from the Huggingface transformers can be finetuned on the sentiment classification task

##### 4.3 Train a flair _TextClassifier_ model
[FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md)
With flair a TextClassifier can be trained by embedding the hotel reviews with concatenated 
FlairEmbeddings ('news-forward' + 'news-backward') and WordEmbeddings('glove') with a single LSTM 
DocumentRNNEmbeddings layer. 

##### 4.4 Train a flair _CNN_ text classifier with concatenated topic and text length features
(Convolutional Neural Networks for Sentence Classification)[https://arxiv.org/pdf/1408.5882.pdf]
[Implementation of the CNN with PyTorch in Github](https://github.com/cezannec/CNN_Text_Classification/blob/master/CNN_Text_Classification.ipynb)
This widely used CNN text classification architecture as introduced by Yoon Kim 2015 needs to be trained on the Hotel Review Text classification task.
As an input embedding matrix for the CNN I would rather use a sequence of the transformer embeddings as generated as input features in my current text classfier with the Roborta model.
As in my current text classification approach I would truncate the texts to a maximum lenght as defined by the review length distribution.
Also I would extend the Github Implementation for a concatenate the final flattened CNN representation topic model embeddings and text length features and use a softmax layer for the final classification. 

The model should be exported in [ONNX format](ttps://docs.microsoft.com/de-de/windows/ai/windows-ml/get-onnx-model) to enable cross-platform compatibility:
h and e.g. facilitate on edge deployment. 


## Extensive Exploratory Data-Analysis and Feature engineering

A lot of feature engineering was performed prior to the Hyperparameter Tuning. 
For this purpose also a Topic modeling and Random Forest Classification for identifying the 
feature imporatance of the engineered features was performed with the following 
Jupyter notebook:
> [Exploratory Data Analysis and Feature Engineering](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/jupyter_notebooks/exploratory_data_analysis.ipynb)

A tiny fraction of the gained insights is presented here:
##### Exploratory Data Analysis
One interesting feature for the classification task comprises a topic model vector. With the topics being fitted on the documents via 
**Latent-Dirichlet-Allocation (LDA)** based on the lemmatized tokens and just maintaining tokens with the part-of-speech-tags (POS) ADV, ADV, NOUN and VERB. 
To decide on the number of topics a Grid Search was performed with the _topic coherence score as a target metric_.
> * [Topic Model Coherence Score Grid Search - 30 Topics were selected for the embedding](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/coherence_scores.png)

Another feature comprises the _minimum, maximum and mean_ **frequent word and adjective polarity scores** which are obtained by matching the Adjectives and frequent words from 
polarity dictionaries with the corresponding documents and aggregating the polarity scores. The plot indicates increasing mean adjective polarity scores for classes 1 and 2 compared to class 0.

> * [Mean Adjective Polarity Score - positive reviews tend to have higher scores](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/class_distribution_mean_adj.png)

To assess how well the different embeddings (document topic vectors, roberta document embeddings, TFIDF document embeddings) are separating the different classes, 2D plots of the dimension reduced TSNE embeddings were visualized. 
Those plots indicate that all of those embeddings show at least some subspace being occupied by the Negative and Neutral hotel reviews.

> * [Roberta Transformer Embedding TSNE - there are somewhat distinct subspaces for negative reviews](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/roberta_vector_tsne.png)
> * [Topic Vector TSNE visualization - there are somewhat distinct subspaces for negative reviews](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/topic_vector_tsne.png)
> * [TFIDF vector TSNE visualization - there are somewhat distinct subspaces for negative reviews](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/tfidf_vector_tsne.png)

##### Sanity check of engineered features
As a sanity check for how well suited the engineered features are for sentiment classification a simple Random Forest classifier was fitted to the training data and the confusion matrix and classification report were 
evaluated for this classification model with its default fitting parameters. Both the confusion matrix and the classification report indicate that the 
RF model is at least reasonably capable of predicting the labels for class 0 and 2 correctly while it totally fails to predict class 1. 
The weighted average F1 score 0.79 and the accuracy score 0.84 for this very crude classification attempt which is a good starting point.
> * [confusion matrix: RandomForest default parameter sanity check are the engineered features reasonable](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/RF_roberta_embedding_confusion_matrix.PNG)
> * [classification report: RandomForest default parameter sanity check are the engineered features reasonable](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/RF_roberta_embedding_metrics.PNG)

Also the feature importance of the engineered features was evaluated. This indicates that in particular the polarity scores and Roberta embedding are mainly contributing to the classification. 

> * [feature importance plot: RandomForest default parameter sanity check are the engineered features reasonable](https://github.com/chiemenz/automl_vs_hyperdrive/blob/master/RF_roberta_embedding_Feature_Importance.PNG)

All in all this pointed out that a traditional machine learning approach with those engineered features is worth trying

## VIDEO Summary
[Video Summary of the Project](https://www.loom.com/share/ed1c914c3e954cc4967407fb1b450dc8)



