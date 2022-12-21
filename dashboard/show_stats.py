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


#-------------------- Class Responsible for Loading of Excel/Csv Files -------
#
class ShowStats:
    '''
    This class will describe the classwise distribution of datasets.
    It will show Charts based on Plotly
    and 
    Other Numarical Stats
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
        st.info('This is EDA tool.  You will load a file and it will display\
            Pie Chart and Bar Chart for better understanding of data')

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
        
    #---------------------- Show Plotly PieChart-----------------------------
    
    def show_piechart(self,col=''):
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
        pie_chart=px.pie(df_inv_groups,names='class_labels',values='counts',
                template= 'plotly_white')
        pie_chart.update_layout(title_text='Data Distribution in Classes',autosize=True,width=700,height=700,)
        st.plotly_chart(pie_chart)
    
    # ------------------------- Loading File ---------------------------------
    #
    def load_file(self):
        '''
        Using Streamlit File Uploader
        '''
        self.uploaded_file = st.file_uploader('Chose a csv or excel file', \
            type=['csv'], accept_multiple_files=False)
    #-------------------- Display the Updated Dataframe ----------------------
    #
    def show_df(self):
        st.empty()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(st.session_state.temp_df)
        with col2:
            
            temp_df=st.session_state.temp_df
            buf = io.StringIO()
            temp_df.info(buf=buf)
            s = buf.getvalue()
            with st.expander("Stats Info"):
                col1_1, col1_2, col1_3 = st.columns([2, 1, 1])
                st.header('Stats Info')
                with col1_1:
                    st.subheader('Dataframe Info')
                    st.text(s)
                with col1_2:
                    st.subheader('Dataframe Shape')
                    st.write(temp_df.shape)
                
                with col1_3:
                    st.subheader('Dataframe Class Counts')
                    st.write(st.session_state.df['class_labels'].value_counts())
            with st.expander("Bar Chart"):
                self.show_barchart()
            with st.expander("Pie Chart"):
                self.show_piechart()
        
    #--------------------------- Save Dataframe ------------------------------
    #
    
    def convert_df(self):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return st.session_state.df.to_csv(index=False)
        
    # ----------------------- Column Selection Menu -------------------------
    #
    def show_column_selection_menu(self):
        
        columns_names1=list(st.session_state.df.columns)
        selected_columns1 = st.multiselect(
                    'Please Select the Columns You Want to Includes?',
                    columns_names1,
                    columns_names1, 
                    on_change=self.show_df
                    )
        # st.session_state['temp_df']=st.session_state.df[columns_names1]
        print(selected_columns1)
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
            columns_names1=list(st.session_state.df.columns)
            selected_columns1 = st.multiselect(
                        'Please Select the Columns You Want to Includes?',
                        columns_names1,
                        columns_names1
                        )
            print(selected_columns1)
            st.session_state.temp_df=st.session_state.df[selected_columns1]
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
    obj = ShowStats()
    obj.Run()
