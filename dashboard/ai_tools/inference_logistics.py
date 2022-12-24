import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,BertModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBarBase
import warnings
from sklearn.utils.extmath import softmax
from torchmetrics import Accuracy
import re


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

class InferenceModel(pl.LightningModule):
    '''
    This is PyTorch model class having forward and other method requird for trianing,validation, testing and prediciton.
    Arguments:
    n_classes(int): Number of classes to predict
    model_path(str): pretrained models path 
    '''
    def __init__(self, n_classes: int,model_path=None):
        '''
        Making the Bert Base model from Hugging face repo
        '''
        super().__init__()

        # return_dict=True
        self.bert = BertModel.from_pretrained(model_path, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_state_output=output.last_hidden_state[:,0,:]
        output = self.classifier(last_state_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


class IdrakTinyBertInference:
    '''
    Class for Inferncing using Bert Model
    It will load checkpoint and label df and
    predict on the base of it.
    '''
    def __init__(self,datapath='',model_path='prajjwal1/bert-tiny',batch_size=1,checkpoint_path='',df_label_path=''):
        '''
        model_path(str): the path of tiny bert
        batch_size(int): 1 for prediction 
        classifier(int): classifier label
        dataset(int): dataset label
        drive_folder(str): the path of pretrained checkpoint
        '''
        self.text=''
        #Class Maps for Class value to Labels
        df_label=pd.read_json(df_label_path)
        self.label_dict=df_label.to_dict()[0]
        #Generated Text Point Path from classifier and dataset value passed
        self.checkpoint_path=checkpoint_path
        self.LABEL_COLUMNS=len(df_label)
        self.model_path=model_path
        self.MAX_TOKEN_COUNT=71
        self.BATCH_SIZE=1
        #Defining Bert Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = InferenceModel(n_classes=self.LABEL_COLUMNS,model_path=self.model_path)
        self.model=self.model.load_from_checkpoint(self.checkpoint_path,model_path=self.model_path,n_classes=self.LABEL_COLUMNS)
        self.save_model_name=''
    def predict(self,text):
        '''
        prediction function. This function get a text string from Infernce object and return the 
        predicted class, probibilities and class labels using trianed pytorch model
        '''
        self.text=text
        print(self.text)
        prediction=self.prep_data()
        prediction=prediction[1].cpu().detach().numpy()
        y_pred=prediction[0].argmax()
        prob=softmax(prediction)[0]
        class_label=self.label_dict[y_pred]
        result={'prob':prob,'class':y_pred,'class_label':class_label}
        return result
    def prep_data(self):
        #function to make dataframe
        self.text=self.cleanify()
        encoding = self.tokenizer.encode_plus(self.text,add_special_tokens=True,max_length=112,return_token_type_ids=False,padding="max_length",truncation=True,return_attention_mask=True,return_tensors='pt')
        input_ids,attention_mask=encoding['input_ids'],encoding['attention_mask']
        prediction=self.model(input_ids,attention_mask)
        return prediction
    def eval_model(self):
        pass
    
    def cleanify(self):
        #function to clean text
        '''
        This is inner function. It will first remove the unwantted symbols from text
        using regular expression. Then Keep the numbers, alphabets, and question mark 
        '''
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') #compile regulare expression for removing symbols
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z ?]') #compile regulare expression to keep wanted data
        text=str(self.text)
        text = text.lower() #making text to lower case
        text = REPLACE_BY_SPACE_RE.sub(' ', text)  #applying 1 and 2nd mentioned re
        text = BAD_SYMBOLS_RE.sub(' ', text)
        return text
    def __repr__(self):
        return 'IdrakTinyBertInference(num_class={},trained_model_path={})'.format(self.LABEL_COLUMNS, self.checkpoint_path)    
    def __str__(self):
        return 'IdrakTinyBertInference Trained over {}'.format(self.checkpoint_path)

