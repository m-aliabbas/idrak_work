# Importing the libs
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from ai_tools.idrak_text_cleaner import IdrakTextCleaner
import io
import glob
import plotly.express as px
import plotly.figure_factory as ff

#----------------- Class Responsible for Analyzing the results ---------------
#
class Analyzer:
    '''
    Analyze the results of differnt model. For every model there will be a 
    directory with model name. These directories will contains the 3 files/
    1: modelname_test_history.csv
    2: modelname_test_invalid_predictions.csv
    3: modelname_test_report.json 
    
    If the directory did not contains these 3 files it will populate some err.
    
    '''
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        This will manage the session initial state wthat will be used for 
        displaying the records
        '''
        
        if 'df_history' not in st.session_state:
            st.session_state.df_history = None
        if 'df_invalid_pred' not in st.session_state:
            st.session_state.df_invalid_pred=None
        if 'df_report' not in st.session_state:
            st.session_state.df_report=None
        if 'label_dict' not in st.session_state:
            st.session_state.label_dict={}
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model=None
        
        # self.uploaded_file = None
        self.file_name = ''
        self.df=st.session_state.df
        self.uploaded_file=False
        self.model_names=None
        st.subheader('Select the Model')
        st.info('You need to select the model. With each model there will be \n'
                'directory in models. And directory will contains \n'
                 '1: modelname_test_history.csv \n'
                 '2: modelname_test_invalid_predictions.csv \n'
                ' 3: modelname_test_report.json '
                )
            
    # ------------------------- Get the Models name --------------------------
    #
    def get_model_names(self):
        '''
        models names 
        '''
        
        trained_models=glob.glob('models/*/')
        trained_models_names=[i[7:].replace('/','') for i in trained_models]
        trained_model_db={trained_models_names[i]:trained_models[i] 
                          for i in range(len(trained_models))}
        self.model_names = trained_model_db
    
    #---------------------- select model menu --------------------------------
    #
    def select_models_menu(self):
        '''
        Showing menu to select the models
        '''
        self.get_model_names()
        selected = st.selectbox('Please select the Model'
                                 ,self.model_names.keys())
        st.session_state.selected_model=selected
    
    #------------------------ read files -------------------------------------
    #
    def read_files(self):
        '''
        read history, invalid prediction and report of models
        '''
        
        #getting path of selected model
        model_path=self.model_names[st.session_state.selected_model]
        
        #reading the files
        #
        df_history = pd.read_csv(f'{model_path}{st.session_state.selected_model}_history.csv')
        df_invalid_pred=pd.read_csv(f'{model_path}{st.session_state.selected_model}_invalid_predictions.csv')
        df_report=pd.read_json(f'{model_path}{st.session_state.selected_model}_report.json')
        df_labels=pd.read_json(f'{model_path}{st.session_state.selected_model}_class_labels.json')
        label_dict=df_labels.to_dict()[0]
        df_invalid_pred['predicted_labels']=df_invalid_pred['y_pred'].map(label_dict)
        
        #updating the session state
        st.session_state.df_history=df_history
        st.session_state.df_report=df_report
        st.session_state.df_invalid_pred=df_invalid_pred
        st.session_state.label_dict=label_dict
        
        
    #-------------------- show invalid predictons ----------------------------
    #
    
    def show_invalid_prediction(self):
        '''
        Show the invalid prediction CSV
        '''
        
        with st.expander("See Invalid Prediction By Models"):
            st.write(f"Invalid prediction by {st.session_state.selected_model}")
            column_to_show=['cleaned_text','class_labels','predicted_labels']
            st.dataframe(st.session_state.df_invalid_pred[column_to_show])
            
    #----------------------- show confusion matrix ---------------------------
    #
    def show_confusion_matrix(self,cm=[[]],labels=[],col=''):
        '''
        Display the confusion matrix using plotly
        '''
        x,y=labels,labels
        z_text = [[str(y) for y in x] for x in cm]
        fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text)
        fig.update_layout(title_text='Confusion matrix',autosize=False,width=500,
                          height=500,)
        fig.update_layout(margin=dict(t=50, l=200)) #adding margin from top and left
        # fig['data'][0]['showscale'] = True
        with st.expander('Confusion Matrix'):
            st.plotly_chart(fig, use_container_width=True)
            
    #---------------------- show classification report -----------------------
    #
    
    def show_classification_report(self):
        '''
        Get classification report from df_report and display it in expander
        '''
        with st.expander('Classification Report '):
            classification_report=st.session_state.df_report['ClassificationReport'].values[0]
            st.text(classification_report)
            
    #-------------------- Show Metrics ---------------------------------------
    #
    
    def show_metries(self):
        
        '''
        It will displau Accuracy, Precesion, Recall and F1 Scores
        '''
        
        with st.expander('Show Metrices'):
            column_to_show=['Accuracy','Precision','Recalll','F1Score']
            df=st.session_state.df_report[column_to_show]
            st.dataframe(df)
            
            
    #----------------------- show training history ---------------------------
    #
    
    def show_training_history(self):
        '''
        Plotting the Training History 
        and Writing Training History
        '''
        try:
            with st.expander('Show History'):
                with st.expander("Plot Training History"):
                    fig = px.line(st.session_state.df_history[['train_acc','valid_acc']])
                    fig.update_traces(textposition="bottom right")
                    st.plotly_chart(fig)
                with st.expander("Show training history"):
                    st.dataframe(st.session_state.df_history)
        except Exception as e:
            print(f'Error {e} in show training history')
            
    #---------------------  show model report --------------------------------
    #
    def show_model_report(self):
        '''
        This will show the model report ; Confusion Matrix; Classification 
        report
        '''
        try:
            with st.expander('Model Report'):
                df=st.session_state.df_report
                cm=df['ConfusionMatrix'].values[0]
                self.show_metries()
                self.show_confusion_matrix(cm=cm,labels=list(st.session_state.label_dict.values())
                                           ,col='')
                self.show_classification_report()
        except Exception as e:
            print(f'Error in show Model Report {e}')
    #-----------------------  showing Expanders ------------------------------
    #
    def show_expeander(self):
        '''
        This will show to the result of model reterived from the model created
        files 
        '''
        self.show_model_report()
        
        self.show_training_history()
        
        self.show_invalid_prediction()
        
    
    # ------------------------------------------------------------------------
    #
    def Run(self):
        '''
        Driver function
        '''
        
        self.select_models_menu()
        self.read_files()   
        self.show_expeander()


# --------------------------------- Main Function ----------------------------
#
def main():
    obj = Analyzer()
    obj.Run()
