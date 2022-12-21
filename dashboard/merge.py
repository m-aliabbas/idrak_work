# Importing the libs
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from ai_tools.idrak_text_cleaner import IdrakTextCleaner
import io
import plotly.express as px
import plotly.figure_factory as ff
import streamlit_nested_layout
from sklearn.preprocessing import LabelEncoder

#-------------------- Class Responsible for Merge CSV File ---------- -------
#
class Merge:
    '''
    Merge The Splitted Dataframes 
    '''
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        This will manage the session initial state wthat will be used for 
        displaying the records
        '''
        
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df=None
        # self.uploaded_file = None
        self.file_name = ''
        self.df=st.session_state.df
        self.uploaded_file=False
        st.subheader('Load the files')
        st.info('Please upload multiple .csv files to merge. Merging will be\
            based on class_labels column. :) ')
        
    # ------------------------- Loading File ---------------------------------
    #
    def load_file(self):
        '''
        Using Streamlit File Uploader
        '''
        self.uploaded_files = st.file_uploader('Choose .csv or excel files', \
            type=['csv'], accept_multiple_files=True)
    
    #-----------------------  merge logic ------------------------------------
    #
    
    def merge_files(self):
        '''
        By Clicking merge Button it will read uploaded files one by one
        and make a bigger csv of them.
        '''
        

        fname = st.text_input('save file name',value='merge')
        btn_merge = st.button('Merge it')
        if btn_merge:
            try:
                for i, f in enumerate(self.uploaded_files):
                    print(f' Working a {i}')
                    if i == 0:
                        data = pd.read_csv(f)
                    else:
                        # loading the data frame
                        df = pd.read_csv(f)
                        # concating the data
                        data = pd.concat([data, df])
                LE = LabelEncoder()
                data['class']=LE.fit_transform(data['class_labels'])
                data.to_csv(f'./data/{fname}.csv', index=False)
                st.success('File Saved')
            except Exception as e:
                print(f'Error as  {e}')


    
    # ------------------------------------------------------------------------
    #
    def Run(self):
        '''
        Driver function
        '''
        self.load_file()
        self.merge_files()


# --------------------------------- Main Function ----------------------------
#
def main():
    obj = Merge()
    obj.Run()
