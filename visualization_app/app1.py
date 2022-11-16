import streamlit as st
st.set_page_config(
        page_title="Idrak Metrics Analyzer",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ai_tools.constant import DATADIR,classifiers_meta,dataset_names,classifier_names
from components import *


#Top Menu     

sidebar=st.sidebar
with sidebar:
    dataset = sidebar.selectbox(
        "Select a Dataset From List",
        dataset_names
    )
    classifer = sidebar.selectbox(
        "Select a Classifier From List",
        classifier_names
    )
model_name='tinybert_report'
#path of file having classifier report
file_path='{}/{}/{}_{}_report.json'.format(DATADIR,model_name,dataset_names[dataset],classifier_names[classifer])
#path of file having model training/evaluation history
history_file_path='{}/{}/{}_{}_history.csv'.format(DATADIR,model_name,dataset_names[dataset],classifier_names[classifer])
#invalid prediction csv path
invalidpredictions_file_path='{}/{}/{}_{}_invalid_predictions.csv'.format(DATADIR,model_name,dataset_names[dataset],classifier_names[classifer])
labels=classifiers_meta[classifier_names[classifer]]['labels'] #Class labels
#head section
make_head()

st.subheader('Model\'s Performace Report')
df=pd.read_json(file_path)
with sidebar:
    selected_columns=sidebar.multiselect('Please select some columns', df.columns)


df_history=pd.read_csv(history_file_path)

df_invalid_predictions=pd.read_csv(invalidpredictions_file_path)
df_invalid_predictions=df_invalid_predictions[['cleaned_text','class','y_pred','class_labels']]
df_invalid_predictions['class']=df_invalid_predictions['class'].map(labels)
df_invalid_predictions['y_pred']=df_invalid_predictions['y_pred'].map(labels)

if selected_columns:
    print(selected_columns)
    if 'ConfusionMatrix' in selected_columns:
        case_1=1
    if 'ClassificationReport' in selected_columns:
        case_1=2
    if ('ConfusionMatrix' in selected_columns) and ('ClassificationReport' in selected_columns):
        case_1=3
    if ('ConfusionMatrix' not in selected_columns) and ('ClassificationReport' not in selected_columns):
        case_1=0
    show_panel1(df=df,selected_columns=selected_columns,labels=labels,case_1=case_1)
else:
    df_to_show=df[df.columns[:-1]]
    st.dataframe(df_to_show)
    
with st.container():
    st.subheader('Model\'s Invalid Predictions')
    col1= st.columns((12))
    options = st.multiselect('Select Classes To Display',labels.values(),labels.values())
    labels_list=options
    df_invalid_predictions=df_invalid_predictions[df_invalid_predictions['class'].isin(labels_list)]
    show_panel2(df=df_invalid_predictions)

with st.container():
    st.subheader('Model\'s Training History')
    plot_history(df_history)