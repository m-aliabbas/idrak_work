# Importing the libs
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from ai_tools.idrak_text_cleaner import IdrakTextCleaner

#-------------------- Class Responsible for Loading of Excel/Csv Files -------
#
class FileLoader:
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        Constructor of FileLoader 
        - Showing the Page Head Header 
        - &
        - information related to file uploading 
        '''
        # self.uploaded_file = None
        self.files = None
        self.file_name = ''
        self.df=None
        self.sheets_name=[]
        self.selected_sheets=[]
        self.is_cleaned=False
        # show title
        st.subheader('Load the files')
        st.info(' In this panel you can take a look on your Excel Sheet or CSV\
                and get an idea about which sheets/columns to Use.')
        
    # ------------------------- Loading File ---------------------------------
    #
    def load_file(self):
        '''
        Using Streamlit File Uploader
        '''
        self.uploaded_file = st.file_uploader('Chose a csv or excel file', \
            type=['csv','xlsx','xls'], accept_multiple_files=False)
        
    # -------------------- show selection menu--------------------------------
    #
    def show_selection_menu(self,file_type):
        st.info(' Before Cleaning the Data You Need to select the followings: \
                  \n \t 1. Columns to Keep in Data, \n \t 2. Text Column i.e X, \n \t 3. Label \
                  Column i.e Y')
        
        
        if file_type=='.xlsx' or file_type=='.xls':
            self.selected_sheets = st.multiselect(
                    'Please Select the Sheets You Want to Includes?',
                    self.sheets_name,
                    self.sheets_name
                    )
        
        columns_names=list(self.df.columns)
       
        self.selected_columns = st.multiselect(
                    'Please Select the Columns You Want to Includes?',
                    columns_names,
                    columns_names
                    )
        self.text_column = st.selectbox(
                        "Please Select the Text Column",
                        self.selected_columns
                    )
        self.label_column = st.selectbox(
                        "Please Select the Label Column",
                        self.selected_columns
                    )
            
    # ------------------------ Cleaner Function ------------------------------
    #
    def cleaner(self):
        try:
            itc=IdrakTextCleaner(file_path=self.uploaded_file,column_to_keep=self.selected_columns,sheet_list=self.selected_sheets,save_file_path='temp.csv',selected_class=[],label_col=self.label_column,wanted_text_col=self.text_column,new_text_col='cleaned_text')
            self.df=itc.df_wanted
            self.is_cleaned=True
            if st.button('save the file'):
                
        except Exception as e:
            print('Following Error Occured \n {}'.format(e))    
        
    #-------------------- Display the Updated Dataframe ----------------------
    #
    def show_df(self):
        self.container.empty()
        self.container.write(self.df)
    # -------------------------  display the file ----------------------------
    #
    def show_file(self):
        '''
        
        Get the list of files
        read the dataframe from CSV one by one
        
        '''
        if self.is_cleaned==False:
            if self.uploaded_file:
                self.file_name = self.uploaded_file.name
                file_type = os.path.splitext(self.file_name)[1]
                print(file_type)
                if file_type == '.csv':
                    self.df = pd.read_csv(self.uploaded_file)
                    
                elif file_type == '.xlsx' or file_type == '.xls':
                    df_info=pd.ExcelFile(self.uploaded_file)
                    self.sheets_name = df_info.sheet_names
                    
                    current_selected_sheet = st.selectbox(
                            "Select a sheet to take a look?",
                            self.sheets_name
                        )
                    
                    self.df = pd.read_excel(self.uploaded_file,sheet_name=current_selected_sheet) 
                    
                else:
                    pass
                
                self.show_selection_menu(file_type)
                if st.button('Clean the Data'):
                    self.cleaner()
        else:
            if st.button('Reload File'):
                self.is_cleaned=False
                self.show_file()
        if self.df is not None:
            self.container=st.empty()
            self.show_df()
    # ------------------------------------------------------------------------
    # 
    def Run(self):
        '''
        Driver function
        '''
        self.load_file()
        self.show_file()


# --------------------------------- Main Function ----------------------------
#
def main():
    obj = FileLoader()
    obj.Run()
