# Importing the libs
#
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.figure_factory as ff
import streamlit_nested_layout


#-------------------- Class Responsible for Training Bert Model------ s-------
#
class Trainer:
    '''
    Helps in splitting the dataframe; based on label columns
    '''
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        This will manage the session initial state wthat will be used for 
        splitting
        '''
        
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df=None
        # self.uploaded_file = None
        self.file_name = ''
        self.df=st.session_state.df
        self.uploaded_file=False
        st.subheader('Spliting the Columns')
        st.info('Please load a dataframe to split it'
                'For Everyclass a seprate file wll be stored'
                )
        
    #------------------ Show Plotly BarChart---------------------------------
    #
    
    def show_barchart(self,col=''):
        '''
        Getting the Dataframe from Session;
        Counting the Number of Elements of Each Class
        Renaming the columns :p
        reseting the index
        plotly magic
        '''
        df_inv=st.session_state.df
        df_inv_groups=df_inv.groupby(by=['class_labels']).count()[['class']]
        df_inv_groups=df_inv_groups.rename(columns={'class':'class','class':'counts'})
        df_inv_groups=df_inv_groups.reset_index()
        bar_chart=px.bar(df_inv_groups,x='class_labels',y='counts',color_discrete_sequence =['#F63366']*len(df_inv_groups),
                template= 'plotly_white')
        bar_chart.update_layout(title_text='Data Distribution in Classes',autosize=True,width=700,height=700,)
        st.plotly_chart(bar_chart)
    
    #---------------------------- split function ----------------------------
    #
    def split_file(self,dir_name):
        '''
        split the dataframe based on column_labels
        '''
        data_tem =  st.session_state.df
        
        label_c =data_tem['class_labels'].unique()
        if not os.path.exists('data'):
            os.mkdir('data')
        dir_name=f'./data/{dir_name}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print(f'{dir_name} created')
        for i in label_c:
            temp = data_tem[data_tem['class_labels']==i]
            fname = dir_name+'/'+i+'.csv'
            temp.to_csv(fname)
            print(f'{fname} saved...')
        st.success('Saved')
    #-------------------- Display the Updated Dataframe ----------------------
    #
    def show_df(self):
        st.empty()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(st.session_state.temp_df)
        with col2:
            with st.expander("Bar Chart"):
                self.show_barchart()
            with st.expander('Split Data'):
                dir_name=st.text_input(label = "It will create a Dir name inwhich datasplit:  ",)
                if dir_name:
                    split_btn = st.button('Split',key=2)
                    # button pressed
                    if split_btn:
                        self.split_file(dir_name=dir_name)
    # ------------------------- Loading File ---------------------------------
    #
    def load_file(self):
        '''
        Using Streamlit File Uploader
        '''
        self.uploaded_file = st.file_uploader('Chose a csv or excel file', \
            type=['csv'], accept_multiple_files=False)
    # -------------------------  display the file ----------------------------
    #
    def show_file(self):
        '''
        
        Show the Loaded file using Streamlit File Uploader object and Pandas
        
        '''
        
        
        if self.uploaded_file and st.session_state.df is None:
            self.file_name = self.uploaded_file.name
            file_type = os.path.splitext(self.file_name)[1]
        
            if file_type == '.csv':
                try:
                    self.df = pd.read_csv(self.uploaded_file)
                    columns=self.df.columns
                    
                    if ('cleaned_text' not in self.df.columns) and ('class' not in self.df.columns):
                        raise("Please Do the Data Cleaning First From Load and Clean Data Tab ")
                    else:
                        st.session_state.df=self.df
                        st.session_state.temp_df=self.df
                except Exception as e:
                    print('Following Error Occured: \n {}'.format(e))
                    st.write('Some Error Occured Please Check the Console.')
            else:
                pass
        
        if st.session_state.df is not None:
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
    obj = Split()
    obj.Run()
