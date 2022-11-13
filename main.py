"""
The constructor of IdrakTinyBertInfrence Requires 3 parameters.

1: `classifier`  expected input any number between 1 to 5. (This para donated the classifier intro, hello etc

2: `dataset` expected input any number between1 to 4 
This para about the dataset on which model is trained. 

3: `drive_folder` path of drive where trained models are stored.

"""
from inference import IdrakTinyBertInference #Import IdrakTinyBertInference API
dataset=1 #The dataset on which model trained
classifier=3 #The Classifier
model_dirpath='models' #Directory Where Model is stored
itbf=IdrakTinyBertInference(classifier=classifier,dataset=dataset,drive_folder=model_dirpath) #Object of InfernceClass

#### For Prediction 

text='A Quick brown Fox jumps over the lazy dog.'
result=itbf.predict(text)

#### Output
print(result)