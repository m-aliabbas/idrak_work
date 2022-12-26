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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import numpy as np
from sklearn.preprocessing import LabelEncoder

#-------------------- augmentors initializations ----------------------------
#
syn_aug = naw.SynonymAug(aug_src='wordnet')
ran_aug = naw.RandomWordAug(action="swap")
split_aug = naw.SplitAug()

aug_list={'syn_aug':syn_aug,'random_aug':ran_aug,'split_aug':split_aug}

#-------------------- Class Responsible for Loading of Excel/Csv Files -------
#
class FixColumn:
    '''
    This class will
    1) Split Train Test Data
    2) Helps in Adding/Droping Samples in Training Data based on
        Augmentation / Sklearn Copying Approch
    3) Visaualization
    4) Support of Downloading Splitted Data
    '''
    #----------------------------Constructor ---------------------------------
    #
    def __init__(self):
        
        '''
        This will manage the session initial state wthat will be used for 
        displaying the records and managing the splits
        '''
        
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df=None
        if 'train_df' not in st.session_state:
            st.session_state.train_df=None
        if 'test_df' not in st.session_state:
            st.session_state.test_df=None
        if 'train_aug_df' not in st.session_state:
            st.session_state.train_aug_df=None
        # self.uploaded_file = None
        self.file_name = ''
        self.df=st.session_state.df
        self.uploaded_file=False
        st.subheader('Load the Cleaned files')
        st.info('You can Select the Classes from Dataframe, UpSample/ \
            Augment Them,Downsample them as need of training')

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
        df_inv=st.session_state.train_aug_df
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
        df_inv=st.session_state.train_aug_df
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
        
        with st.expander('DataFrame'):
            st.write(st.session_state.train_aug_df)
            st.write(st.session_state.train_aug_df.shape)
        with st.expander("Bar Chart"):
            self.show_barchart()
        
            
    #---------------------- UpSample Logic -----------------------------------
    #
    def upsample(self,upsample_class_label,val):
        '''
        Upsample Logic:
        1) Getting a Class labels to Upsample and Number of Elements to Samples
        2) Do Upsampling using Sklearn
        3) concat orignla df with upsample
        4) Randomize Dataframe
        args:
            upsample_class: class to upsample
            val: number of value to upsample
        
        '''
        df_min =  st.session_state.train_aug_df[ st.session_state.train_aug_df['class_labels'] == upsample_class_label]
        up_samp = resample(df_min,
                           
                               replace=True,  # sample with replacement
                               n_samples=val,  # to match majority class
                               random_state=123)  # reproducible results
        st.session_state.train_aug_df= pd.concat([ st.session_state.train_aug_df, up_samp])
        
        st.session_state.train_aug_df.sample(frac=0.1)
        # self.show_df()
        
    #---------------------- Downsample Logic -----------------------------------
    #
    def downsample(self,downsample_class_label,val):
        '''
        Downsample Logic:
        1) Getting a Class labels to Downsample and Number of Elements to Samples
        2) Do Upsampling using Sklearn
        3) concat orignla df with Downsmple
        4) Randomize Dataframe
        args:
            upsample_class: class to Downsample
            val: number of value to Downsample
        
        '''
        df_max =  st.session_state.train_aug_df[ st.session_state.train_aug_df['class_labels'] == downsample_class_label]
        down_samp = resample(df_max,
                           
                               replace=True,  # sample with replacement
                               n_samples=val,  # to match majority class
                               random_state=123)  # reproducible results
        
        st.session_state.train_aug_df = st.session_state.train_aug_df[ st.session_state.train_aug_df['class_labels'] != downsample_class_label]
        st.session_state.train_aug_df = pd.concat([ st.session_state.train_aug_df, down_samp])
        st.session_state.train_aug_df.sample(frac=0.1)
    
    #------------------ Augmentation Logic ----------------------------------
    #
    def augment_text(self,aug_list={},aug_class_label='',val=100):
        '''
        This function will augment texgt based on provided augmentor lists.
        Steps:
        1) Get a dataframe and extract the specified class from it
        2) Extract differnt Columns
        3) Itterate over each text and augment it 
        4) Make New Dataframe from Augmented
        5) Check for Column not present in augment 
        6) Randomly Droping rows
        7) Append with Parent dataframe
        8) Randomize
        
        augs:
            aug_class_label: class to augment string
            val: how much samples to augment
            aug_list: list of augmentors
        '''

        temp_df=st.session_state.train_aug_df[st.session_state.train_aug_df['class_labels']==aug_class_label]
        class_labels=temp_df.iloc[0].class_labels
        class_number=temp_df.iloc[0]['class']
        texts=list(temp_df['cleaned_text'].values)
        augmented_texts=[]
        for text in texts:
            for aug in aug_list:
                augmented_texts+=aug_list[aug].augment(text)
        class_labels_list=len(augmented_texts)*[class_labels]
        class_numbers_list=len(augmented_texts)*[class_number]
        temp_df1=pd.DataFrame()
        temp_df1['class_labels']=class_labels_list
        temp_df1['class']=class_numbers_list
        temp_df1['cleaned_text']=augmented_texts
        
        #Check for columns whose are not in augmented dataframe
        labels_not_in_temp=list(set(temp_df.columns)-set(temp_df1.columns))
        #filling those columns rows with "Augmented Row" Text
        
        for label in labels_not_in_temp:
            temp_df1[label]=['augmented_row']*len(augmented_texts)
        print(len(temp_df1))
        if len(temp_df1)<=val:
            val=len(temp_df1)
        #randomly selecting the rows from it
        randomly_select=list(set(np.random.randint(len(temp_df1), size=val)))
        temp_df1=temp_df1.iloc[randomly_select]
        st.session_state.train_aug_df=st.session_state.train_aug_df.append(temp_df1)
        #shuffling the dataframe
        st.session_state.train_aug_df=st.session_state.train_aug_df.sample(frac=0.1)

    
    #--------------------------- Save Dataframe ------------------------------
    #
    
    def convert_df(self,df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index=False)
    
    #-------------------------- make directory in contents -------------------
    #
    def make_dir(self,dir_name):
        dir_path = f'contents/data/{dir_name}/'
        
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
    #-------------------------------Save file to Contents/data ---------------
    #
    
    def save_df(self,df,dir_name,file_name):
        '''
        Save the dataframe to contents/data for AI Training
        '''
        dir_path = f'contents/data/{dir_name}/'
        file_path= dir_path+file_name
        df.to_csv(file_path,index=False)
        
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
                        st.session_state.train_df,st.session_state.test_df=train_test_split(st.session_state.df,test_size=0.2)
                        st.session_state.train_aug_df=st.session_state.train_df
                except Exception as e:
                    print('Following Error Occured: \n {}'.format(e))
                    st.write('Some Error Occured Please Check the Console.')
            else:
                pass
        
        if st.session_state.df is not None:
            
            class_labels=list(st.session_state.train_aug_df.class_labels.unique())
            print(class_labels)
            selected_classes = st.multiselect(
                        'Please Select the Classes: ',
                        class_labels,
                        class_labels
                        )
            
            LE = LabelEncoder()
            temp=st.session_state.train_aug_df
            print(temp)
            temp=temp[temp['class_labels'].isin(selected_classes)]
            temp['class'] = LE.fit_transform(temp['class_labels'])
            st.session_state.train_aug_df=temp
            
            temp1=st.session_state.train_df
            temp1=temp1[temp1['class_labels'].isin(selected_classes)]
            temp1['class'] = LE.fit_transform(temp1['class_labels'])
            st.session_state.train_df=temp1
            
            temp2=st.session_state.test_df
            temp2=temp2[temp2['class_labels'].isin(selected_classes)]
            temp2['class'] = LE.fit_transform(temp2['class_labels'])
            st.session_state.test_df=temp2
            
            self.show_df()
            
            with st.expander('Augment'):
                class_labels=list(st.session_state.train_aug_df.class_labels.unique())
                option = st.selectbox(
                    "Select class to Augment: ",
                    class_labels
                )
                val = st.number_input('Number of Elements in Augments',min_value=100,max_value=1500)
                #button
                if st.button('Augment'):
                    self.augment_text(aug_class_label=option,aug_list=aug_list,val=val)
            
            with st.expander('Upsample'):
                class_labels=list(st.session_state.train_aug_df.class_labels.unique())
                option = st.selectbox(
                    "Select class to Upsample: ",
                    class_labels
                )
                ###########################
                val = st.number_input('Number of Upsamples',min_value=100,max_value=1500)
                #button
                if st.button('Upsample'):
                    self.upsample(option,val)
            
            with st.expander('Downsample'):
                class_labels=list(st.session_state.train_aug_df.class_labels.unique())
                option = st.selectbox(
                    "Select class to Downsample: ",
                    class_labels
                )
                val = st.number_input('Number of Downsamples',min_value=100,max_value=1500)
                #button
                if st.button('Downsample'):
                    self.downsample(option,val)
            
            with st.expander('Download Data'):
                file_name=st.text_input(label = "Please Enter Save File_Name: ",)
                train_csv=self.convert_df(st.session_state.train_df)
                train_aug_csv=self.convert_df(st.session_state.train_aug_df)
                test_csv=self.convert_df(st.session_state.test_df)
                if file_name:
                    self.make_dir(dir_name=file_name) # create a dir inside conents
                    print(f'{file_name} created')
                    file_name1='{}_train.csv'.format(file_name)
                    
                    self.save_df(df=st.session_state.train_df,dir_name=file_name,
                                 file_name=file_name1)
                    st.download_button(
                            label="Download training CSV",
                            data=train_csv,
                            file_name=file_name1,
                            mime='text/csv',
                        )
                    file_name1='{}_train_aug.csv'.format(file_name)
                    self.save_df(df=st.session_state.train_aug_df,dir_name=file_name,
                                 file_name=file_name1)
                    st.download_button(
                            label="Download Augmented CSV",
                            data=train_aug_csv,
                            file_name=file_name1,
                            mime='text/csv',
                        )
                    file_name1='{}_test.csv'.format(file_name)
                    self.save_df(df=st.session_state.test_df,dir_name=file_name,
                                 file_name=file_name1)
                    st.download_button(
                            label="Download test CSV",
                            data=test_csv,
                            file_name=file_name1,
                            mime='text/csv',
                        )
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
    obj = FixColumn()
    obj.Run()
