'''
    Copyright IdrakAI ltd.
    This module is developed to clean the dataframe.
    Developed by M. Ali Abbas
    maliabbas@gmail.com
'''
## ------------------------ Importing the Libraries --------------------------
#

import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import numpy as np
import math

## ------------------------ Idrak Text Cleaner Class --------------------------
#
class IdrakTextCleaner():
  def __init__(self,file_path,sheet_list,save_file_path,column_to_keep=[],wanted_text_col='Text',label_col='Audio Tag',new_text_col='actual_english_cleaned',new_label_col='class',selected_class=[]):
    '''
      This class receive the Excel file and Convert the Excel File to csv.
      Excel file is first cleaned. Labels are attached with excel file.
    '''
    self.label_class_mapping={1:'positive',2:'negative',3:'answering_machine',4:'busy',5:'dnc',6:'greetings',7:'dnq',8:'not_intrested',9:'spanish',10:'other',11:'X',12:'X',13:'already_insured',14:'bot',15:'sorry_greeting',16:'greetback'}
    self.file_path=file_path
    self.file_type=file_path.name.split(".")[-1]
    self.n_sheets=sheet_list
    self.column_to_keep=column_to_keep
    self.save_file_path=save_file_path
    self.wanted_text_col=wanted_text_col
    self.label_col=label_col
    self.new_label_col=new_label_col
    self.new_text_col=new_text_col
    self.selected_class=selected_class
    self.df=pd.DataFrame()
    self.df_wanted=pd.DataFrame()
    if self.file_type=='xls' or self.file_type=='xlsx':
        self.read_excel()
    elif self.file_type=='csv':
        self.read_csv()
    else:
      raise RuntimeError("Please use Excel or CSV file")
    self.cleaner()
    self.number_to_label()
    self.selector()
    self.label_encode()
    self.shuffle()
    self.save_csv()
  def read_csv(self):
    self.df=pd.read_csv(self.file_path)
  def read_excel(self):
    dfs=[]
    for sheet_name in self.n_sheets:
      df=pd.read_excel(self.file_path,sheet_name=sheet_name)
      dfs.append(df)
    self.df=pd.concat(dfs)
  def save_csv(self):
    self.df_wanted.to_csv(self.save_file_path+'.csv',index=False)
  def shuffle(self):
    self.df_wanted=self.df_wanted.sample(frac=1)
  def cleanify(self,text):
    '''
    This is inner function. It will first remove the unwantted symbols from text
    using regular expression. Then Keep the numbers, alphabets, and question mark 
    '''
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;\|]') #compile regulare expression for removing symbols
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z]') #compile regulare expression to keep wanted data
    text=str(text)
    text = text.lower() #making text to lower case
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  #applying 1 and 2nd mentioned re
    text = BAD_SYMBOLS_RE.sub(' ', text)
    return text
  def cleaner(self):
    self.df_wanted=self.df
    # if len(self.column_to_keep)>0:
    self.df_wanted=self.df_wanted[self.column_to_keep]
    self.df_wanted[self.new_label_col]=self.df_wanted[self.label_col]
    self.df_wanted[self.wanted_text_col]=self.df_wanted[self.wanted_text_col]
    self.df_wanted.dropna(inplace=True)
    self.df_wanted[self.new_text_col]=self.df_wanted[self.wanted_text_col].apply(self.cleanify)

    self.df_wanted=self.df_wanted[self.df_wanted[self.new_label_col]<17]
    self.df_wanted=self.df_wanted.drop_duplicates(keep='last')
    
    self.df_wanted[self.new_label_col]=self.df_wanted[self.new_label_col].astype(int)
    self.df_wanted=self.df_wanted[self.df_wanted[self.new_label_col]!=12]
    
    self.df_wanted=self.df_wanted.drop([self.wanted_text_col,self.label_col],axis=1)
    
  def selector(self):
    '''
    Select the sheet and df based on it
    '''
    if self.selected_class!=[]:
      try:
        selected_dfs=[]
        for i in self.selected_class:
          df=self.df_wanted[self.df_wanted['class_labels']==i]
          selected_dfs.append(df)
        self.df_wanted=pd.concat(selected_dfs)
      except Exception as e:
        print('Following Error in Selector \n \t',e)
  def label_encode(self):
    label_encoder = preprocessing.LabelEncoder()
    self.df_wanted[self.new_label_col]= label_encoder.fit_transform(self.df_wanted['class_labels'])
  def number_to_label(self):
    self.df_wanted['class_labels']=self.df_wanted[self.new_label_col].map(self.label_class_mapping).fillna('X')

