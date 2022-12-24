# Importing the libs
#
import streamlit as st
import pandas as pd
import numpy as np
from ai_tools.inference_logistics import IdrakTinyBertInference
import glob


#----------------- Class Responsible for Analyzing the results ---------------
#
class Evaluate:
    '''
    
    Evaluate the Model on Single Text or A CSV Files
    
    '''
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        This will manage the session initial state wthat will be used for 
        displaying the records
        '''
        
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'predicted_df' not in st.session_state:
            st.session_state.predicted_df=pd.DataFrame()
        # self.uploaded_file = None
        self.file_name = ''
        self.uploaded_checkpoint_file=False
        self.uploaded_label_file=False
        self.uploaded_csv_file=False
        self.model_names=None
        self.predicted_df=pd.DataFrame()
        st.subheader('Evaluate the Model')
        st.info('You need to Upload Checkoint and Label file of respective '
                'model to perform evaluation'
                )
            
    #---------------- Upload Checkpoint and label df -------------------------
    #
    
    def upload_files(self):
        if st.session_state.model is None:
            self.uploaded_checkpoint_file = st.file_uploader("Choose checkpint: ")
            self.uploaded_label_file = st.file_uploader('Choose label json: ')  
        
          
    #------------------------ Load the Model ---------------------------------
    #
    
    def load_model(self):
        '''
        Load the Bert Model
        '''
        if self.uploaded_checkpoint_file and self.uploaded_label_file:
            if st.session_state.model == None :
                with st.spinner('Model is loading wait'):
                    model=IdrakTinyBertInference(checkpoint_path=self.uploaded_checkpoint_file,df_label_path=self.uploaded_label_file)
                    st.session_state.model=model
                st.success('Done!')
                
                
    #--------------------------- Show Prediction Box -------------------------
    #
    def show_prediction_box(self):
        if st.session_state.model:
            with st.expander('Evaluate on Text Strings'):
                text=st.text_area('Please write sometext: ')
                btn_predict=st.button('Predict')
                if btn_predict: #if pressed
                    #query to ai for inference
                    prediction=st.session_state.model.predict(text) 
                    #showing the results
                    st.success('Class Name: '+prediction['class_label'])
                    st.success('Class Number: '+str(prediction['class']))
                    st.success('Probs: '+str(prediction['prob']))
    @st.cache
    def convert_df(self,df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv()
    #---------------------- Show File Prediction -----------------------------
    #
    def show_file_prediction_box(self):
        '''
        Get a text file.
        Do prediction on each item
        add prediction labels etc
        '''
        if st.session_state.model:
            with st.expander('Evaluate on Text File '):
                st.info('Please Upload a File Having text column')
                
                uploaded_file = st.file_uploader("File for Prediction ")
                
                if uploaded_file:
                    # reading csv
                    temp_df=pd.read_csv(uploaded_file)
                    texts=list(temp_df.text.values)
                
                    predictions=[st.session_state.model.predict(text) for text in texts]
                    classes=[prediction['class'] for prediction in predictions]
                    class_labels=[prediction['class_label'] for prediction in predictions]
                    probs =[prediction['prob'] for prediction in predictions]
                    
                    temp_df['predicted_class_number']=classes
                    temp_df['predicted_class_labels']=class_labels
                    temp_df['predicted_probs']=probs
                    
                    st.dataframe(temp_df)
    
                    
                    st.download_button(
                        label="Download data as CSV",
                        data=self.convert_df(temp_df),
                        file_name='idrak_dashboard_predicted.csv',
                        mime='text/csv',
                    )
                    
                    
                    
                    
    # ------------------------------------------------------------------------
    #
    def Run(self):
        '''
        Driver function
        '''
        self.upload_files()
        self.load_model()
        self.show_prediction_box()
        self.show_file_prediction_box()


# --------------------------------- Main Function ----------------------------
#
def main():
    obj = Evaluate()
    obj.Run()
