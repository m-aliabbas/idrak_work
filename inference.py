"""Importing the Modules"""
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,BertModel
from transformers import AdamW
import pytorch_lightning as pl
from torchmetrics import F1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
from sklearn.utils.extmath import softmax
from torchmetrics import Accuracy
import re
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

class LitProgressBar(ProgressBarBase):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = False

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
#Pytorch lighting Requires a Dataset consiting of dataframe for prediction
#So creating the dataset and dataloader
class CallCenterDataset(Dataset):
  '''
  Dataset for Bert Processing
  '''
  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    max_token_len: int = 40
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
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
    
class CallCenterTagger(pl.LightningModule):
  '''
  This is PyTorch model class having forward and other method requird for trianing,validation, testing and prediciton.
  Arguments:
  n_classes(int): Number of classes to predict
  model_path(str): pretrained models path 
  '''
  def __init__(self, n_classes: int,model_path=None,n_training_steps=None, n_warmup_steps=None,learning_rate=0.02):
    super().__init__()

    # return_dict=True
    self.bert = BertModel.from_pretrained(model_path, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
       
    self.softmax = nn.Softmax(dim=1)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
     
    self.criterion = nn.CrossEntropyLoss()
    self.train_f1 = F1Score(num_classes=n_classes,average="micro")
    self.train_acc=Accuracy()
    self.val_f1=F1Score(num_classes=n_classes,average="micro")
    self.val_acc=Accuracy()
    self.learning_rate=learning_rate
  def forward(self, input_ids, attention_mask, labels=None):
    
    output = self.bert(input_ids, attention_mask=attention_mask)
    last_state_output=output.last_hidden_state[:,0,:]
   
    output = self.classifier(last_state_output)

    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    y_pred=outputs
    acc = self.train_acc(y_pred, labels)
    f1 = self.train_f1(y_pred, labels)

    self.log("train_accuracy", acc)
    self.log("train_f1", f1)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    loss, outputs = self(input_ids, attention_mask, labels)
    y_pred = outputs
    acc=self.val_acc(y_pred,labels)
    f1=self.val_f1(y_pred, labels)

    self.log("valid_accuracy", acc)
    self.log("valid_f1", f1)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    loss, outputs = self(input_ids, attention_mask, labels)
    y_pred = outputs
    acc=self.val_acc(y_pred,labels)
    f1=self.val_f1(y_pred, labels)

    self.log("test_accuracy", acc)
    self.log("test_f1", f1)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss
  def predict_step(self,batch,batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    loss, outputs = self(input_ids, attention_mask, labels)
    y_pred = outputs
    acc=self.val_acc(y_pred,labels)
    f1=self.val_f1(y_pred, labels)

    return y_pred

  def training_epoch_end(self, outputs):
    
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    train_accuracy = self.train_acc.compute()
    train_f1 = self.train_f1.compute()
    print('Train Accuracy: ',train_accuracy)
    print('Train F1: ',train_f1)
    self.log("epoch_train_accuracy", train_accuracy)
    self.log("epoch_train_f1", train_f1)
    self.train_acc.reset()
    self.train_f1.reset()


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
  
  def configure_optimizers(self):
    LEARNING_RATE=self.learning_rate
    param_optimizer = list(self.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)

    # scheduler = get_linear_schedule_with_warmup(
    #   optimizer,
    #   num_warmup_steps=self.n_warmup_steps,
    #  num_training_steps = -1
    # )

    return dict(
      optimizer=optimizer,
    )
    
class InferenceDataModue(pl.LightningDataModule):
  '''
  Module for Data loading. This module bind the text data and tokenizer on them.
  test_df(dataframe): having column cleaned_text and class 
  batchsize(int): 1 for prediction
  max_token_len(int): Maximum length of token for tokenizer
  '''
  def __init__(self,test_df, tokenizer, batch_size=1, max_token_len=64):
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
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
  def predict_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )


class IdrakTinyBertInference:
  def __init__(self,datapath='',model_path='prajjwal1/bert-tiny',batch_size=1,classifier=1,dataset=1,model_name='',drive_folder='/content/gdrive/MyDrive/idraak/model/tinybert_report',checkpoint_name='best_checkpoint'):
    '''
    model_path(str): the path of tiny bert
    batch_size(int): 1 for prediction 
    classifier(int): classifier label
    dataset(int): dataset label
    drive_folder(str): the path of pretrained checkpoint
    '''
    self.df_test=pd.DataFrame()
    self.text=''
    self.classifier=classifier
    self.dataset=dataset
    self.drive_folder=drive_folder
    #Class Maps for Class value to Labels
    self.classifier_1_labelmaps={0:'answering_machine',1:'dnc'}
    self.classifier_2_labelmaps={0:'answering_machine',1:'busy',2:'dnc',3:'greetback',4:'greeting',5:'other',6:'sorry_greeting',7:'spanish'}
    self.classifier_3_labelmaps={0:'bot',1:'busy',2:'dnc',3:'negative',4:'not_intrested',5:'other',6:'positive',7:'spanish'}
    self.classifier_4_labelmaps={0:'dnc',1:'negative',2:'not_intrested',3:'other',4:'positive'}
    self.classifier_5_labelmaps={0:'dnc',1:'negative',2:'not_intrested',3:'other',4:'positive'}
    self.classifiers_meta={1:{'labels':self.classifier_1_labelmaps,'NUM_CLASSES':2},
                  2:{'labels':self.classifier_2_labelmaps,'NUM_CLASSES':8},
                  3:{'labels':self.classifier_3_labelmaps,'NUM_CLASSES':8},
                  4:{'labels':self.classifier_4_labelmaps,'NUM_CLASSES':5},
                  5:{'labels':self.classifier_5_labelmaps,'NUM_CLASSES':5}
                  }
    #Generated Text Point Path from classifier and dataset value passed
    print(self.classifier,self.dataset)    
    self.checkpoint_path='{}/best_dataset{}_classifier_{}.ckpt'.format(self.drive_folder,self.dataset,self.classifier)
    self.checkpoint_name='best_'+checkpoint_name
    self.LABEL_COLUMNS=self.classifiers_meta[classifier]['NUM_CLASSES']
    self.log_dir = "lightning_logs/IDRAK/version_0"
    self.drive_folder=drive_folder
    self.model_name=model_name
    self.model_path=model_path
    self.MAX_TOKEN_COUNT=71
    self.BATCH_SIZE=batch_size
    #Defining Bert Tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = CallCenterTagger(n_classes=self.LABEL_COLUMNS,model_path=self.model_path,n_warmup_steps=0,n_training_steps=-1,learning_rate=0)
    print(self.checkpoint_path)
    self.model=self.model.load_from_checkpoint(self.checkpoint_path,model_path=self.model_path,n_classes=self.LABEL_COLUMNS)
    self.save_model_name=''
    self.checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",filename=self.checkpoint_name,save_top_k=1,verbose=True,monitor="val_loss",mode="min")
    self.logger = TensorBoardLogger("lightning_logs", name="IDRAK")
    self.bar = LitProgressBar()
    self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)
    self.trainer = pl.Trainer(logger=self.logger,callbacks=[self.early_stopping_callback,self.checkpoint_callback,self.bar],max_epochs=0)
    self.dm=InferenceDataModue(test_df=self.df_test,tokenizer=self.tokenizer,batch_size=1,max_token_len=self.MAX_TOKEN_COUNT)
    
  def predict(self,text):
    '''
    prediction function. This function get a text string from Infernce object and return the 
    predicted class, probibilities and class labels using trianed pytorch model
    '''
    self.text=text
    prediction=self.prep_data()
    prediction=prediction[0].cpu().detach().numpy()
    y_pred=prediction[0].argmax()
    prob=softmax(prediction)[0]
    class_label=self.classifiers_meta[self.classifier]['labels'][y_pred]
    result={'prob':prob,'class':y_pred,'class_label':class_label}
    return result
  def prep_data(self):
    #function to make dataframe
    self.text=self.cleanify()
    self.df_test['cleaned_text']=[self.text]
    self.df_test['class']=[1] #dummy label
    self.df_test['class_labels']=['xx']
    # print(self.df_test)
    self.dm= self.dm=InferenceDataModue(test_df=self.df_test,tokenizer=self.tokenizer,batch_size=1,max_token_len=self.MAX_TOKEN_COUNT)
    p=self.trainer.predict(self.model,datamodule=self.dm)
    return p
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
