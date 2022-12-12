# Importing the libs
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#-------------------- Class Responsible for Loading of Excel/Csv Files -------
class FileLoader:
    def __init__(self):
        
        '''
        Constructor of FileLoader 
        - Showing the Page Head Header 
        - &
        - information related to file uploading 
        '''
        # self.uploaded_file = None
        self.files = None

        # show title
        st.subheader('Load the files')
        st.success('By Loading the files (one can load multiple files), it  will remove the blank lines'
                 ', fix the columns, fix the headers, and save the file in data folder')
    # ------------------------- Loading File ---------------------------------
    def load_files(self):
        '''
        Using Streamlit File Uploader
        '''
        self.uploaded_file = st.file_uploader('Chose a csv file', type='csv', accept_multiple_files=True)

    # -------------------------  display the file ----------------------------
    def show_files(self):
        '''
        
        Get the list of files
        read the dataframe from CSV one by one
        
        '''
        st.markdown('***')
        for f in self.uploaded_file:
            with st.expander(f.name):
                df = pd.read_csv(f)
                df = df.iloc[1:, :2]
                df.columns = ['text', 'label']

                st.write('Blanks in data :', df['text'].isnull().sum())

                df['text'].replace('', np.nan, inplace=True)
                df.dropna(subset=['text'], inplace=True)

                df.to_csv(f'./data/{f.name}',index=False)
                fig, ax = plt.subplots()
                ax = pd.value_counts(df.iloc[:,1]).plot.bar()
                # st.write(pd.value_counts(df.iloc[:,1]))
                st.pyplot(fig)
                # st.write(df.iloc[:, 1].unique().count())
                st.write(f'Size = {df.shape}')
                # st.write(df)


    # ============================================
    #  Run
    def Run(self):
        self.load_files()
        self.show_files()


# ============================================
def main():
    obj = FileLoader()
    obj.Run()
