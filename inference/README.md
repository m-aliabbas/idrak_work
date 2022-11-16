(The Notebook contains the experiment details of Inference API)


**TinyBertInfernce API**
Idrak is using Tiny Bert Model for the classification of texts. TinyBert out perform the classical ML *(Random Forest, SGD, LightGBM, XGBoost, Linear Regression, and SVM)* as well as other Bert Varients *(MobileBert, Distilt Bert and Bert-Base-Uncased)* . Also, Tiny Bert is occupying 50 MB space on disk while other Bert Models are taking more than 400 MB space. 
Moreover, We have initialy 4 datasets and 5 classifier for each dataset. 
The detail of datasets are following
1.   Actual Text (Human Written Data)
2.   Data from Current Transcriptor
3.   Data from Transcriptor with Decoder (Beam)
4.   Data from Transcriptor without Decoder (Greedy)


And, the detail of classifier are following

1.  Hello, (Answering Machine and DNC ) `NUM_CLASSES=2`
2.  Intro, (Answering Machine, Busy, DNC, Greetings, sorry greetings, Greet back, Spanish, Other) `NUM_CLASSES=8`
3.  Pitch, (Busy, DNC, Spanish, Other, Not interested, Positive, Negative, BOT) `NUM_CLASSES=8`
4.  Yes No without Age Sheet (Positive, Negative, DNC, Other, Not interested) `NUM_CLASSES=5`
5.  Yes No with Age Sheet (Positive, Negative, DNC, Other, Not interested) `NUM_CLASSES=5`
This API will take a text string and return a dictionary containing probabilities of different classes `prob` and class label `class`

Mounting the Google Drive . You can ommit this line if you are running envoirnment other than Google Drive.

**Dataset:** https://drive.google.com/drive/folders/1YDvc7E_QYwlhaxMGxk0E7YI1tNeixgiF?usp=share_link 

==> data/datasetX/classifierX_train.csv 

==> data/datasetX/classifierX_train_aug.csv 

==> data/datasetX/classifierX_test.csv

**Models:** https://drive.google.com/drive/folders/1RbchTgviRCxcQ1wjniX79mM9VtxWSJ_o?usp=share_link 

**Checkpoint Path:**

==> model/tinybert_report/best_datasetX_classifier_X.ckpt

**Model History**

==> model/tinybert_report/datasetX_classifier_X_history.csv

**Wrong Predictions Record**

==> model/tinybert_report/datasetX_classifier_X_invalid_predictions.csv

**Classification Report**

==> model/tinybert_report/datasetX_classifier_X_report.json

**Instructions:**

The constructor of IdrakTinyBertInfrence Requires 3 parameters.
```
from inference import IdrakTinyBertInference #Import IdrakTinyBertInference API
dataset=1 #The dataset on which model trained
classifier=3 #The Classifier
model_dirpath='models' #Directory Where Model is stored
itbf=IdrakTinyBertInference(classifier=classifier,dataset=dataset,drive_folder=model_dirpath) #Object of InfernceClass
```
1: `classifier`  expected input any number between 1 to 5. (This para donated the classifier intro, hello etc

2: `dataset` expected input any number between1 to 4 
This para about the dataset on which model is trained. 

3: `drive_folder` path of drive where trained models are stored.

For prediction you need to pass a text string to IdrakTinyBertInference object `predict` function as
```
result=itbf.predict(text)
print(result)
```
**expected output**
**Expected Output:**

`prob` : probabilities of each class

`class`: class numaric label

`class_label`: class label in human readable format
```
{'prob': array([0.15899259, 0.05700221, 0.12287708, 0.63530654, 0.02582158],
       dtype=float32), 'class': 3, 'class_label': 'other'}
```
