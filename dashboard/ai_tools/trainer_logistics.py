#------------------------------- Libraries -----------------------------------
#
import os
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import uuid
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel,BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertModel, DistilBertConfig,DistilBertTokenizer
from transformers import MobileBertConfig,MobileBertModel,MobileBertTokenizer
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torchmetrics import F1Score, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report,accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import warnings
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging
from torchmetrics import Accuracy

#------------------------ Configurations -------------------------------------
#
RANDOM_SEED = 321
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02",
                        "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
pl.seed_everything(RANDOM_SEED)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
InteractiveShell.ast_node_interactivity = "all"
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

model_path='prajjwal1/bert-tiny'
tokenizer = AutoTokenizer.from_pretrained(model_path)



#------------------------- Dataset Createder----------------------------------
#

class CallCenterDataset(Dataset):
    #------------------------- Dataset Createder------------------------------
    '''
    
    CallCenterDataset is inherited from PyTorch Dataset Object.
    This class is responsible for making dataset read for Bert Training from 
    dataframe. Also Trainer will read item by item rows from the dataframe. 
    
    '''
    #------------Constructor--------------------------------------------------
    #
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, 
                 max_token_len: int = 40):
        '''
        args:
            data: pandas dataframe having column cleaned.text and class
            ---------- cleaned.text: for cleaned input for bert
            ---------- class : column for labels must be numeric labels starts from zero
            tokenizer: Tiny Bert Tokenizer from Hugging Face
            max_token_len: proposed token length use for padding of low length tokens
            
        '''
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
    
    def __len__(self):
        '''
        return the legnth of dataframe
        '''
        return len(self.data)

    def __getitem__(self, index: int):
        
        '''
        This function will return each item from dataframe using index. with
        it will return the tokenize version with attention masks.
        
        args:
            index (int): integer input for class labels
        
        returns:
            dict (python dictionary):
                the dictionary consits of 
                comment_text: cleaned text from dataframe
                input_ids: tokenize comment text
                attention_mask: attention mask of input text
                labels: the label of class row
        '''
        
        data_row = self.data.iloc[index]
        comment_text = data_row.cleaned_text
        labels = data_row['class']
        encoding = self.tokenizer.encode_plus(
          comment_text,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return dict(
          comment_text=comment_text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=labels
        )

#------------------ CallCenterData Module ------------------------------------
#
class CallCenterDataModule(pl.LightningDataModule):
    
    '''
    This class is inherited from Pytorch lighting module. This is responsible
    for making training , validation and testing loaders. It create the batches 
    also shuffle them.
    '''
    #------------------- Constructor------------------------------------------
    #
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, 
                 max_token_len=64):
        '''
        
        args: 
        train_df (dataframe): training dataframe
        test_df  (dataframe): testing dataframe
        tokenizer (huggingface tokenizer): tiny bert tokenizer
        bacth_size (int) : proposed batchsize
        
        '''
        
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        
    #------------------------- Setup function --------------------------------
    #
    
    def setup(self, stage=None):
        
        '''
        Setting up CallCenterDataset Object for training ; validation and 
        testing. 
        '''
        
        self.train_dataset = CallCenterDataset(
          self.train_df,
          self.tokenizer,
          self.max_token_len

        )

        self.test_dataset = CallCenterDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len
        )
        #testing and validation object df are same 
        self.valid_dataset = CallCenterDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len
        )
    #-------------------- training loader ------------------------------------
    #
    def train_dataloader(self):
        
        '''
        Dataloader for training dataset.
        
        returns: 
              dataloader (pytorch lighting dataloader object) :
              ------- shuffle=True will shuffle it.
        '''
        
        print('Train loader Called')
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=4
        )
    #--------------------------- validation loader ---------------------------
    #
    def val_dataloader(self):
        
        '''
        Dataloader for validation dataset.
        
        returns: 
              dataloader (pytorch lighting dataloader object) :
              ------- shuffle=True will shuffle it.
        '''
        
        print('Valid loader Called')
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=0
        )
    #----------------------------- testing loader ----------------------------
    #
    def test_dataloader(self):
        
        '''
        Dataloader for testing dataset.
        
        returns: 
              dataloader (pytorch lighting dataloader object) :
              ------- shuffle=True will shuffle it.
        '''
        
        print('Test loader Called')
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=0
        )
        
        
#------------------- CallCenterTagger-----------------------------------------
#
class CallCenterTagger(pl.LightningModule):
    
    '''
    
    This class extened from Pytorch lighting. It will contain the logic for
    PyTorch Models. Its Different layers and Classification metrics using
    torch metrics. 
    
    '''
    
    #-------------------- contructor------------------------------------------
    #
    
    def __init__(self, n_classes: int,model_path=None,n_training_steps=None, 
                 n_warmup_steps=None,learning_rate=0.02):

        super().__init__()
        '''
        Constructor initalizing differnt arguments.

        args: 
            n_classes (int): number of classes on which we are gonna train
            model_path(str): the path of base model i.e. tiny_bert huggingface
                             repo path.
            n_training_steps(int): total training steps
            n_warmup_steps(int): total number of warmup stemps use for scheduler
                                 of learning rate. 
            learning_rate (float16): learning rate for model training.

        '''

        #------------ making bert tokenizer from hugging face repo----------------
        #
        self.bert = BertModel.from_pretrained(model_path, return_dict=True)
        print('Number of classes,',n_classes)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.learning_rate=learning_rate

        #-------------- loss function --------------------------------------------
        #
        self.criterion = nn.CrossEntropyLoss()

        #------------------ metrics ---------------------------------------------
        #
        task="multiclass"
        # if n_classes<=2:
        #     task='binary'
        
        self.train_f1 = F1Score(num_classes=n_classes,average="micro",task=task)
        self.train_acc=Accuracy(task=task,num_classes=n_classes)
        self.val_f1=F1Score(num_classes=n_classes,average="micro",task=task)
        self.val_acc=Accuracy(task=task,num_classes=n_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        '''
        forward function traing the model.
        
        args:
            input_ids (trochtensor): tokenize text input
            attention_mask (torchtensor): attention masks of tokenize text 
                                           input.
            labels: class labels of training batch
            
        returns:
            loss (torchtensor) : the training loss
            output (torchtensor): predicted output for training batch
        '''
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_state_output=output.last_hidden_state[:,0,:]
        output = self.classifier(last_state_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    #------------------training step---------------------------
    def training_step(self, batch, batch_idx):
        '''
        
        1) getting batch and ides of batch
        2) making prediction on batch
        3) calculating accuracy and f1scores
        4) logging metrics
        5) making progress bar
        
        args: 
            batch (torchtensor)
            batch_idx (torchtensor)
            
        returns:
            dict (python dictionary)
            ----- loss: loss for step prediction
            ----- outputs: predicted labels
        '''
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        y_pred=outputs
        # y_pred = self.softmax(outputs)
        acc = self.train_acc(y_pred, labels)
        f1 = self.train_f1(y_pred, labels)

        self.log("train_accuracy", acc)
        self.log("train_f1", f1)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    #-------------------------- validation steps -----------------------------
    #
    def validation_step(self, batch, batch_idx):
        '''
        
        1) getting batch and ides of batch
        2) making prediction on batch
        3) calculating accuracy and f1scores
        4) logging metrics
        5) making progress bar
        
        args: 
            batch (torchtensor)
            batch_idx (torchtensor)
            
        returns:
            dict (python dictionary)
            ----- loss: loss for step prediction
            ----- outputs: predicted labels
        '''
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(input_ids, attention_mask, labels)
        y_pred = outputs
        # y_pred = self.softmax(outputs)
        acc=self.val_acc(y_pred,labels)
        f1=self.val_f1(y_pred, labels)

        self.log("valid_accuracy", acc)
        self.log("valid_f1", f1)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    #-----------------testing step--------------------------------------------
    #
    def test_step(self, batch, batch_idx):
        '''
        
        1) getting batch and ides of batch
        2) making prediction on batch
        3) calculating accuracy and f1scores
        4) logging metrics
        5) making progress bar
        
        args: 
            batch (torchtensor)
            batch_idx (torchtensor)
            
        returns:
            dict (python dictionary)
            ----- loss: loss for step prediction
            ----- outputs: predicted labels
        '''
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(input_ids, attention_mask, labels)
        y_pred = outputs
        # y_pred = self.softmax(outputs)
        acc=self.val_acc(y_pred,labels)
        f1=self.val_f1(y_pred, labels)

        self.log("test_accuracy", acc)
        self.log("test_f1", f1)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    #-------------------- for prediction on target data-----------------------
    #
    def predict_step(self,batch,batch_idx):
        
        '''
        
        1) getting batch and ides of batch
        2) making prediction on batch
        3) calculating accuracy and f1scores
        4) logging metrics
        5) making progress bar
        
        args: 
            batch (torchtensor)
            batch_idx (torchtensor)
            
        returns:
            dict (python dictionary)
            ----- loss: loss for step prediction
            ----- outputs: predicted labels
        '''
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(input_ids, attention_mask, labels)
        y_pred = outputs
        # y_pred = self.softmax(outputs)
        acc=self.val_acc(y_pred,labels)
        f1=self.val_f1(y_pred, labels)

        # self.log("test_accuracy", acc)
        # self.log("test_f1", f1)
        # self.log("test_loss", loss, prog_bar=True, logger=True)
        return y_pred
    
    
    #---------------- after successful completion of training epoch ----------
    #
    def training_epoch_end(self, outputs):
        '''
            calcualting the accuracy and f1 scores
            
            args: 
            
                outputs (torchtensor): predicted ouptus
        '''
        #---------------- concating the labels and predictions ----------------
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        #----------------- calculating metrices -------------------------------
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        print('Train Accuracy: ',train_accuracy)
        print('Train F1: ',train_f1)
        #----------------- logging-------------------------------------------
        self.log("epoch_train_accuracy", train_accuracy)
        self.log("epoch_train_f1", train_f1)
        
    #---------------- after successful completion of validtion epoch ----------
    #
    def validation_epoch_end(self, outputs):
        
        val_accuracy = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        print('Valid Accuracy: ',val_accuracy)
        print('Valid F1: ',val_f1)
        # log metrics
        self.log("epoch_val_accuracy", val_accuracy)
        self.log("epoch_val_f1", val_f1)
        self.val_acc.reset()
        self.val_f1.reset()
        
    #------------------- model hyperparameter configuration ------------------
    #
    def configure_optimizers(self):
        '''
        configuring learning_rate ; scheduler and other things for finetunnings.
        '''
        LEARNING_RATE=self.learning_rate
        
        #getting layers paranters
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        #setting differnt hyper paramters like weight_decay for differet layers f
        #finetuning
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
            ]
        
        #----------------------------AdamW optimizer from huggingface---------
        #
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, 
                          correct_bias=False)
        
        return dict(
          optimizer=optimizer,
          # lr_scheduler=dict(
          #   scheduler=scheduler,
          #   interval='step'
          # )
        )
    
        #-------------------------- if we have to use Schduler ---------------
             # scheduler = get_linear_schedule_with_warmup(
            #   optimizer,
            #   num_warmup_steps=self.n_warmup_steps,
            #  num_training_steps = -1
            # )

                # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                #factor=0.1,patience=5,verbose=True)

            # return dict(
            #   optimizer=optimizer,
            #   lr_scheduler=dict(
            #     scheduler=scheduler,
            #     monitor = "train_loss"
            #   )
            # )
#----------------------- My Module -------------------------------------------
#

class MyModule(pl.LightningDataModule):
    
    '''
    This class is inherited from Pytorch lighting module. This is responsible
    for making prediction dataloader . It create the batches 
    also shuffle them.
    '''
    
    #---------------------------- Constructor --------------------------------
    #
    def __init__(self,test_df, tokenizer, batch_size=8, max_token_len=64):
    
        super().__init__()
        self.batch_size = batch_size
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len


        self.test_dataset = CallCenterDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len)
        self.predict_dataset = CallCenterDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len)
        
    #--------------------- Test Loader --------------------------------------
    #
    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=0
        )
    #---------------------------- Prediction Loader---------------------------
    #
    def predict_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=0
        )
        
#--------------------------- Tagger Logic ------------------------------------
#
class IdrakTinyBertClassifier:
    '''
    This class will train the TinyBert Model. It will load training and
    testing dataset, drop_na, metrics reports, records of invalids 
    prediction of models. 
    
    '''
    #------------------------- constructor -----------------------------------
    def __init__(self,datapath,model_path='prajjwal1/bert-tiny',
                 num_epochs=100,batch_size=16,model_name='',
                 learning_rate=0.02,
                 drive_folder='',
                     checkpoint_name='best_checkpoint'):
        '''

        1) Reading the training and validation datafrom paths
        2)

        args: 
             datapath (str) : path of dataset where the dataset is stored localy
             model_path(str): the base path of tiny bert hugging face repo
             num_epochs(int): number of training epochs
             batch_size (int): the batch size
             drive_folder (str): where the generated results, checkpoints and 
                                 other model generated stuff will be stored.
            checkpoint_names(str): the name of final_checkpoint i.e best 

        '''
        #drive folder
        drive_folder=checkpoint_name
        
        if not os.path.exists('models'):
            os.mkdir('models')
            print('models directory created')
            
        if not os.path.exists('models/{}'.format(drive_folder)):
            os.mkdir('models/{}'.format(drive_folder))
            print('models/{} created'.format(drive_folder))
        drive_folder='models/{}'.format(drive_folder)
        
        #reading data
        self.df_train_org=pd.read_csv(datapath+'_train.csv')
        self.df_train=pd.read_csv(datapath+'_train_aug.csv')
        self.df_test=pd.read_csv(datapath+'_test.csv')
        
        #self_labels
        
        #droping na
        self.df_train.dropna(inplace=True)
        self.df_test.dropna(inplace=True)

        #appending best in user given checkpoint name
        self.checkpoint_name='best_'+checkpoint_name

        #Getting length of label column for initialing n_classes in bert mode;
        self.LABEL_COLUMNS=len(self.df_train['class'].value_counts())
        self.column_name=self.df_train['class_labels'].unique()

        #path of log directory
        self.log_dir_name=self.my_random_string(6)
        self.log_dir=f'lightning_logs/{self.log_dir_name}'
        if not os.path.exists(f'lightning_logs/{self.log_dir_name}'):
            os.mkdir(f'lightning_logs/{self.log_dir_name}')
        self.log_dir=f'lightning_logs/{self.log_dir_name}'
        #initalizing various parameters
        self.drive_folder=drive_folder
        self.learning_rate=learning_rate
        self.scores={}
        self.model_name=model_name
        self.acc=0
        self.f1=0
        self.recal=0
        self.prec=0
        self.confusion_mat=[[]]
        self.clf_report=''
        self.y_pred=''
        self.model_name=model_name
        self.model_path=model_path
        self.MAX_TOKEN_COUNT=31
        self.N_EPOCHS=num_epochs
        self.num_warmup_steps=0
        self.BATCH_SIZE=batch_size
        self.df_model_history=pd.DataFrame()
        self.label_df=pd.DataFrame()
        
        #downloading tokenizer
        self.barow()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.get_max_token_count()

        #calculating hyper paramters for scheuduler
        GRADIENT_ACCUMULATION_STEPS = 1
        WARMUP_PROPORTION = 0.01
        MAX_GRAD_NORM = 5
        num_train_steps = int(len(self.df_train) / self.BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * self.N_EPOCHS)
        num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
        self.num_warmup_steps=num_warmup_steps

        # Dataframe to Dataset conversion
        self.train_dataset = CallCenterDataset(self.df_train,self.tokenizer,
                                               max_token_len=self.MAX_TOKEN_COUNT)
        self.data_module= CallCenterDataModule(self.df_train,self.df_test,tokenizer,
                                               batch_size=self.BATCH_SIZE,
                                               max_token_len=self.MAX_TOKEN_COUNT)

        #Downloading the Models and Initiating the CallCenterTagger Object

        self.model = CallCenterTagger(n_classes=self.LABEL_COLUMNS,
                                      model_path=self.model_path,
                                      n_warmup_steps=self.num_warmup_steps,
                                      n_training_steps=-1,
                                      learning_rate=self.learning_rate)
        self.model_x = CallCenterTagger(n_classes=self.LABEL_COLUMNS,
                                        model_path=self.model_path,
                                        n_warmup_steps=self.num_warmup_steps,
                                        n_training_steps=-1,
                                        learning_rate=self.learning_rate)

        self.save_model_name=''
        
            
        #--------------------------------Defining callbacks-----------------------
        #
        #for saving check points we are only saving the best based on val_loss 
        self.checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",filename=self.checkpoint_name,save_top_k=2,verbose=True,monitor="val_loss",mode="min")
        #call back for logging
        self.logger = TensorBoardLogger("lightning_logs", name=self.log_dir_name)
        #call back for earlystoping if no imporvement in model in 7 epochs
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=7)

        #------------------------------------ initiating the trainer -------------
        #
        self.trainer = pl.Trainer(logger=self.logger,
                                  callbacks=[self.early_stopping_callback,
                                             self.checkpoint_callback],
                                  max_epochs=self.N_EPOCHS,accelerator="gpu")
        #------------------ data module for testing/prediction--------------------
        #
        self.dm=dm=MyModule(test_df=self.df_test,tokenizer=self.tokenizer,batch_size=1,max_token_len=self.MAX_TOKEN_COUNT)
        #---------------------- driver-------------------------------------------
        #
        self.make_label_df()
        self.train_model()
        self.eval_model()
        self.record_history()
        self.invalid_predictions()
        self.make_report()
    #---------------------------- random string ------------------------------
    #
    def my_random_string(self,string_length=10):
        '''
        Returns a random string of length string_length.
        '''
        random = str(uuid.uuid4()) # Convert UUID format to a Python string.
        random = random.upper() # Make all characters uppercase.
        random = random.replace("-","") # Remove the UUID '-'.
        return random[0:string_length] # Return the random string.

    #--------------------- make label df -------------------------------------
    #
    def make_label_df(self):
        '''
        Map the Human and Machie Label for the dataset
        '''
        df = self.df_train
        label_dict={ df['class'].unique()[i]:df['class_labels'].unique()[i] for i in range(len(df['class_labels'].unique()))}
        self.label_df=pd.DataFrame().from_dict(label_dict,orient ='index')
        
    #----------------------------- borrow function ---------------------------
    #
    def barow(self):
        '''
        incase of answering maching getig some labels from orignal train to test
        '''
        temp_df=self.df_train_org[self.df_train_org['class_labels']=='answering_machine'][:5]
        self.df_test=pd.concat([self.df_test,temp_df])
        self.df_test=self.df_test.sample(frac=1)
    #------------ Getting Maximum Token Count and adding 12 withit------------
    #
    def get_max_token_count(self):
        '''
        getting maximum token count by iterating over whole dataset 
        and using max function of PyTorch
        '''
        token_counts = []
        for _, row in self.df_train.iterrows():
            token_count = len(self.tokenizer.encode(
                row["cleaned_text"], 
                max_length=512, 
                truncation=True
              ))
            token_counts.append(token_count)
        max_tokens=max(token_counts)+12
        self.MAX_TOKEN_COUNT=max_tokens
    #------------------ Starting Training -----------------------------------
    #
    def train_model(self):
        '''
        training model using pytorch lighting training regime
        '''
        self.trainer.fit(self.model, self.data_module)
    #------------------ Custom Evaluation ------------------------------------
    #
    def eval_custom(self,df_test=None,checkpoint_path=''):
        
        '''
        1) Getting Testing Datafarme class labels as y_test
        2) Load Model from checkpoint
        3) Make Predictions using Predict Step
        4) Calculate Metrices
        5) Update Metrices 
        
        args:
            checkpoint_path(str): the path of checkpoint you have made after 
                                  training
            df_test (dataframe) : if you want to put custom data to test
            
        '''
        checkpoint_name=os.path.basename(checkpoint_path)
        
        drive_folder=checkpoint_name[:-4]
        
        if not os.path.exists('models'):
            os.mkdir('models')
            print('models directory created')
            
        if not os.path.exists('models/{}'.format(drive_folder)):
            os.mkdir('models/{}'.format(drive_folder))
            print('models/{} created'.format(drive_folder))
        drive_folder='models/{}'.format(drive_folder)
        self.drive_folder=drive_folder
        
        if df_test is None:
            df_test=self.df_test
        
        self.y_test=df_test['class'].values
        self.model_x=self.model_x.load_from_checkpoint(checkpoint_path,
                                                       model_path=self.model_path,
                                                       n_classes=self.LABEL_COLUMNS)
        
        p=self.trainer.predict(self.model_x,datamodule=self.dm) #predicting
        self.y_pred=[p[i].argmax().cpu().item() for i in range(len(p))] #doing argmax operation
        self.df_test['y_pred']=self.y_pred
        
        #-------------- Calculating Metrics ----------------------------------
        self.acc=accuracy_score(y_pred=self.y_pred,y_true=self.y_test)
        self.prec=precision_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.recall=recall_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.f1=f1_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.confusion_mat=confusion_matrix(self.y_test, self.y_pred)
        self.clf_report=classification_report(self.y_test, self.y_pred,target_names=self.column_name)
        
        self.scores['Precision']=self.prec
        self.scores['Recalll']=self.recall
        self.scores['Accuracy']=self.acc
        self.scores['F1Score']=self.f1
        self.scores['ConfusionMatrix']=self.confusion_mat
        self.scores['ClassificationReport']=self.clf_report
        
        self.record_history()
        self.invalid_predictions()
        self.make_report()
        
        print('Congrats!..... Everything Goes Successful')
        
    #--------------------- Running Evaluation COde----------------------------
    #
    def eval_model(self):
        
        '''
        1) Getting Testing Datafarme class labels as y_test
        2) Load Model from checkpoint
        3) Make Predictions using Predict Step
        4) Calculate Metrices
        5) Update Metrices 
        '''
        
        self.y_test=self.df_test['class'].values
        self.model_x=self.model_x.load_from_checkpoint("checkpoints/{}.ckpt".
                                                       format(self.checkpoint_name),
                                                       model_path=self.model_path,
                                                       n_classes=self.LABEL_COLUMNS)
        
        p=self.trainer.predict(self.model_x,datamodule=self.dm) #predicting
        self.y_pred=[p[i].argmax().cpu().item() for i in range(len(p))] #doing argmax operation
        self.df_test['y_pred']=self.y_pred
        
        #-------------- Calculating Metrics ----------------------------------
        self.acc=accuracy_score(y_pred=self.y_pred,y_true=self.y_test)
        self.prec=precision_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.recall=recall_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.f1=f1_score(y_pred=self.y_pred,y_true=self.y_test,average='weighted')
        self.confusion_mat=confusion_matrix(self.y_test, self.y_pred)
        self.clf_report=classification_report(self.y_test, self.y_pred,target_names=self.column_name)
        
        self.scores['Precision']=self.prec
        self.scores['Recalll']=self.recall
        self.scores['Accuracy']=self.acc
        self.scores['F1Score']=self.f1
        self.scores['ConfusionMatrix']=self.confusion_mat
        self.scores['ClassificationReport']=self.clf_report
    #------------------ to record model history dataframe --------------------
    #
    def record_history(self):
        '''
        read logdir and update model training history dataframe against
        each epochs (specified steps) 
        It uses TensorBoard
        '''
        list_of_version=glob.glob(f'{self.log_dir}/*')
        latest_version = max(list_of_version, key=os.path.getctime)
        #reading 
        self.event_accumulator =EventAccumulator(latest_version)
        self.event_accumulator.Reload()
        
        events1 =  self.event_accumulator.Scalars('epoch_train_accuracy')
        events2 =  self.event_accumulator.Scalars('epoch_val_accuracy')
        events3 =  self.event_accumulator.Scalars('epoch_train_f1')
        events4 =  self.event_accumulator.Scalars('epoch_val_f1')
        events5 =  self.event_accumulator.Scalars('epoch_train_f1')
        events6 =  self.event_accumulator.Scalars('epoch_val_f1')
        
        x = [x.step for x in events1]
        y = [x1.value for x1 in events1]
        z = [x2.value for x2 in events2]
        a = [x3.value for x3 in events3]
        b = [x4.value for x4 in events4]
        c = [x5.value for x5 in events5]
        d = [x6.value for x6 in events6]
        #making dataframe
        self.df_model_history = pd.DataFrame({"step": x, "train_acc": y,'valid_acc':z,'train_f1':a,'valid_f1':b,'train_loss':c,'valid_losss':d})
    #----------------- Recording invalid Predictions--------------------------
    #
    def invalid_predictions(self):
        '''
        Keeping record of invalid prediction of model
        where y_true is not equal to y_pred
        '''
        self.invalid_pred_df=self.df_test[self.df_test['class']!=self.df_test['y_pred']]
    #---------------------- Function to Save CSVs and Jsong of Model ---------
    #
    def make_report(self):
        '''
        Saving Model Performance and Generated Dataframes 
        to Disk for Analysis and Future Reference
        '''
        self.invalid_pred_df.to_csv(self.model_name+'_invalid_predictions.csv')
        self.model_report=pd.DataFrame([self.scores])
        self.model_report.to_json(self.model_name+'_report.json')
        self.df_model_history.to_csv(self.model_name+'_history.csv')
        self.label_df.to_json(self.model_name+'_class_labels.json')
        invld_pre_copy=self.model_name+'_invalid_predictions.csv'
        report_copy=self.model_name+'_report.json'
        history_copy=self.model_name+'_history.csv'
        label_copy=self.model_name+'_class_labels.json'
        
        # model_checkpoint_name="/content/checkpoints/{}.ckpt".format(self.checkpoint_name)
        try:
            #---------------- Coping the Data---------------------------------
            #
            # os.popen('cp {} {}'.format(model_checkpoint_name,self.drive_folder))
            os.popen('cp {} {}'.format(invld_pre_copy,self.drive_folder))
            os.popen('cp {} {}'.format(report_copy,self.drive_folder))
            os.popen('cp {} {}'.format(history_copy,self.drive_folder))
            os.popen('cp {} {}'.format(label_copy,self.drive_folder))
        except Exception as e:
            print(e)
            

